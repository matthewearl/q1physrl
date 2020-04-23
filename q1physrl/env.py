import dataclasses
import enum
from typing import Tuple

import gym.spaces
import numpy as np
import ray.rllib

from . import phys


__all__ = (
    'ActionToMove',
    'Config',
    'eval',
    'eval_coro',
    'get_obs_scale',
    'INITIAL_YAW_ZERO',
    'Key',
    'Obs',
    'PhysEnv',
    'SimpleConfig',
    'SimplePhysEnv',
    'SinglePhysEnv',
)


_DEFAULT_TIME_DELTA = np.float32(0.014)
_MAX_YAW_SPEED = np.float32(2 * 360)  # degrees per second
_INITIAL_STATE = {'z_pos': np.float32(32.843201),
                  'vel': np.array([0, 0, -12], dtype=np.float32),
                  'on_ground': np.bool(False),
                  'jump_released': np.bool(True)}
INITIAL_YAW_ZERO = np.float32(90)


class Key(enum.IntEnum):
    STRAFE_LEFT = 0
    STRAFE_RIGHT = enum.auto()
    FORWARD = enum.auto()
    JUMP = enum.auto()   # Not used if _AUTO_JUMP is true


class Obs(enum.IntEnum):
    TIME_LEFT = 0
    YAW = enum.auto()
    Z_POS = enum.auto()
    X_VEL = enum.auto()
    Y_VEL = enum.auto()
    Z_VEL = enum.auto()


@dataclasses.dataclass(frozen=True)
class Config:
    num_envs: int
    auto_jump: bool # Automatically jump when approaching the floor
    initial_yaw_range: Tuple[float, float] # Range of yaw values on the first step.
    max_initial_speed: float # Maximum speed along the ground plane on the first step.
    zero_start_prob: float # Probability of starting with zero speed at the first step.
    # Physics frame time should be 0.01388888 to match quake but 0.014 is here for backwards compatibility.
    time_delta: float = 0.014   
    action_range: float = _MAX_YAW_SPEED * _DEFAULT_TIME_DELTA
    time_limit: float = 5 # Episode length in seconds
    key_press_delay: float = 0.3    # Minimum time between consecutive presses of the same key.
    discrete_yaw_steps: int = -1    # -1 = continuous.  Does nothing if allow_yaw is false.
    speed_reward: bool = False  # reward speed rather than velocity in y dir.
    fmove_max: float = 800.  # units per second
    smove_max: float = 700.  # units per second
    hover: bool = False     # No gravity, fixed initial speed
    smooth_keys: bool = False  # Register half a key press on transitions
    allow_jump: bool = True  # If auto-jump is False, then permit jumping with an action
    allow_yaw: bool = True  # Have yaw dimension in action space.


class ActionToMove:
    """Convert a sequence of actions into a sequence of move commands"""
    _last_key_press_time: np.ndarray
    _last_keys: np.ndarray
    _yaw: np.ndarray

    def __init__(self, config: Config):
        self._config = config
        has_jump_action = not self._config.auto_jump and self._config.allow_jump
        self._num_keys = len(Key) if has_jump_action else len(Key) - 1

    @property
    def action_space(self):
        if not self._config.allow_yaw:
            yaw_action_space = []
        elif self._config.discrete_yaw_steps == -1:
            yaw_action_space = [gym.spaces.Box(low=-self._config.action_range, high=self._config.action_range,
                                               shape=(1,), dtype=np.float32)]
        else:
            yaw_action_space = [gym.spaces.Discrete(2 * self._config.discrete_yaw_steps + 1)]

        return gym.spaces.Tuple([*(gym.spaces.Discrete(2) for _ in range(self._num_keys)), *yaw_action_space])

    def _fix_actions(self, actions):
        # trainer.compute_actions() and trainer.train() return data in slightly different formats.
        return np.array([[np.ravel(x)[0] for x in a] for a in actions])

    def map(self, actions, z_vel, time_remaining):
        actions = self._fix_actions(actions)
        key_actions = actions[:, :self._num_keys].astype(np.int)

        max_yaw_delta = _MAX_YAW_SPEED * self._config.time_delta
        yaw_steps = self._config.discrete_yaw_steps

        if not self._config.allow_yaw:
            mouse_x_action = 0.
        elif yaw_steps == -1:
            mouse_x_action = actions[:, self._num_keys] * max_yaw_delta / self._config.action_range
        else:
            mouse_x_action = (actions[:, self._num_keys] - yaw_steps) * max_yaw_delta / yaw_steps

        # Rate limit key presses
        elapsed = (self._config.time_limit - time_remaining[:, None] >=
                   self._last_key_press_time + self._config.key_press_delay)
        keys = key_actions & (elapsed | self._last_keys)
        self._last_key_press_time = np.where(
            keys & ~self._last_keys,
            (self._config.time_limit - time_remaining[:, None]),
            self._last_key_press_time
        )

        # Register half a press if transitioning, see cl_input.c:CL_KeyState()
        if self._config.smooth_keys:
            smoothed_keys = (keys + self._last_keys) * 0.5
        else:
            smoothed_keys = keys

        self._last_keys = keys

        self._yaw = self._yaw + mouse_x_action
        #self._yaw = np.full((keys.shape[0],), 100.)
        strafe_keys = smoothed_keys[:, Key.STRAFE_RIGHT] - smoothed_keys[:, Key.STRAFE_LEFT]
        smove = np.float32(self._config.smove_max) * strafe_keys
        fmove = np.float32(self._config.fmove_max) * smoothed_keys[:, Key.FORWARD]
        if self._config.auto_jump:
            jump = z_vel <= 16
        elif self._config.allow_jump:
            jump = keys[:, Key.JUMP].astype(np.bool)
        else:
            jump = np.full((keys.shape[0]), False)

        return self._yaw, smove.astype(np.int), fmove.astype(np.int), jump.astype(np.bool)

    def vector_reset(self, yaw):
        self._last_key_press_time = np.full((self._config.num_envs, self._num_keys),
                                            -self._config.key_press_delay)
        self._last_keys = np.full((self._config.num_envs, self._num_keys), False)

        self._yaw = np.array(yaw)

    def reset_at(self, index, yaw):
        self._last_key_press_time[index] = -self._config.key_press_delay
        self._last_keys[index] = False
        self._yaw[index] = yaw


def get_obs_scale(config):
    return [config.time_limit, 90., 100, 200, 200, 200]


class SinglePhysEnv(gym.Env):
    def __init__(self, config):
        config = dict(config)
        config['num_envs'] = 1

        self._env = PhysEnv(config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def step(self, action):
        (obs,), (reward,), (done,), (info,) = self._env.vector_step([action])
        return obs, reward, done, info

    def reset(self):
        (obs,) = self._env.vector_reset()
        return obs


class PhysEnv(ray.rllib.env.VectorEnv):
    player_state: phys.PlayerState
    _yaw: np.ndarray
    _time_remaining: np.ndarray
    _step_num: int
    _zero_start: np.ndarray

    def _round_vel(self, v):
        # See sv_main.c : SV_WriteClientdataToMessage
        return (v / 16).astype(int) * 16

    def _round_origin(self, o):
        # See common.c : MSG_WriteCoord

        # We should not send a changed origin if the position has not changed by more than 0.1.
        # See `miss` variable in `SV_WriteEntitiesToClient`.
        return np.round(o * 8) / 8

    def _get_obs(self):
        ps = self.player_state

        t = self._time_remaining
        vel = self._round_vel(ps.vel)
        z_pos = self._round_origin(ps.z_pos)

        obs = np.concatenate([t[:, None], self._yaw[:, None], z_pos[:, None], vel], axis=1)
        return obs / self._obs_scale

    def _get_obs_at(self, index):
        ps = self.player_state
        t = self._time_remaining[index]
        z_pos = self._round_origin(ps.z_pos[index])
        vel = self._round_vel(ps.vel[index])
        obs = np.concatenate([t[None], self._yaw[index][None], z_pos[None], vel], axis=0)
        return obs / self._obs_scale

    def __init__(self, config):
        self._config = Config(**config)
        self.num_envs = self._config.num_envs

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(6,), dtype=np.float32)
        self.reward_range = (-1000 * self._config.time_delta, 1000 * self._config.time_delta)
        self.metadata = {}

        self._obs_scale = get_obs_scale(self._config)

        self._action_to_move = ActionToMove(self._config)
        self.action_space = self._action_to_move.action_space
        self._step_num = 0
        self.vector_reset()

    def vector_reset(self):
        self.player_state = phys.PlayerState(
            **{k: np.stack([v for _ in range(self.num_envs)]) for k, v in _INITIAL_STATE.items()})

        self._zero_start = np.random.random(size=(self.num_envs,)) < self._config.zero_start_prob

        self._yaw = np.where(self._zero_start,
                             INITIAL_YAW_ZERO,
                             np.random.uniform(*self._config.initial_yaw_range, size=(self.num_envs,)))
        self._time_remaining = np.where(self._zero_start,
                                        self._config.time_limit,
                                        np.random.uniform(self._config.time_limit, size=(self.num_envs,)))
        speed = np.where(self._zero_start,
                         0,
                         np.random.uniform(self._config.max_initial_speed, size=(self.num_envs,)))

        if self._config.hover:
            speed[:] = 320

        move_angle = np.random.uniform(2 * np.pi, size=(self.num_envs,))

        if self._config.hover:
            move_angle[:] = np.pi / 2

        self.player_state.vel[:, 0] = speed * np.cos(move_angle)
        self.player_state.vel[:, 1] = speed * np.sin(move_angle)

        self._action_to_move.vector_reset(self._yaw)

        return self._get_obs()

    def reset_at(self, index):
        for k, v in _INITIAL_STATE.items():
            getattr(self.player_state, k)[index] = v

        self._zero_start[index] = np.random.random() < self._config.zero_start_prob
        self._yaw[index] = (INITIAL_YAW_ZERO if self._zero_start[index] else
                            np.random.uniform(*self._config.initial_yaw_range))
        self._time_remaining[index] = (self._config.time_limit
                                       if self._zero_start[index]
                                       else np.random.uniform(self._config.time_limit))
        speed = 0 if self._zero_start[index] else np.random.uniform(self._config.max_initial_speed)

        move_angle = np.random.uniform(2 * np.pi)
        self.player_state.vel[index, 0] = speed * np.cos(move_angle)
        self.player_state.vel[index, 1] = speed * np.sin(move_angle)

        self._action_to_move.reset_at(index, self._yaw[index])

        return self._get_obs_at(index)

    def vector_step(self, actions):
        if self._config.hover:
            self.player_state.vel[:, 2] = 0
            self.player_state.z_pos[:] = 100

        z_vel = self.player_state.vel[:, 2]
        self._yaw, smove, fmove, jump = self._action_to_move.map(actions, z_vel, self._time_remaining)

        pitch = np.zeros((self.num_envs,), dtype=np.float32)
        roll = np.zeros((self.num_envs,), dtype=np.float32)
        button2 = jump
        time_delta = np.full((self.num_envs,), self._config.time_delta)

        inputs = phys.Inputs(yaw=self._yaw, pitch=pitch, roll=roll, fmove=fmove, smove=smove,
                             button2=button2, time_delta=time_delta)

        self.player_state = phys.apply(inputs, self.player_state)

        if self._config.speed_reward:
            reward = self._config.time_delta * np.linalg.norm(self.player_state.vel[:, :2], axis=1)
        else:
            reward = self._config.time_delta * self.player_state.vel[:, 1]

        self._time_remaining -= self._config.time_delta
        done = self._time_remaining < 0

        self._step_num += 1

        return self._get_obs(), reward, done, [{'zero_start': self._zero_start[i]} for i in range(self.num_envs)]

    def get_unwrapped(self):
        return []


@dataclasses.dataclass(frozen=True)
class SimpleConfig:
    num_envs: int
    time_limit: float = 5
    action_range: float = _MAX_YAW_SPEED * _DEFAULT_TIME_DELTA


_SIMPLE_INITIAL_STATE = {'z_pos': np.float32(50),
                         'vel': np.array([0, 320, 0], dtype=np.float32),
                         'on_ground': np.bool(False),
                         'jump_released': np.bool(True)}


class SimplePhysEnv(ray.rllib.env.VectorEnv):
    """An environment in which only yaw can be controlled, and the initial state doesn't change"""

    _yaw: np.ndarray
    _time_remaining: np.ndarray
    _player_state: phys.PlayerState

    def __init__(self, config):
        self._config = SimpleConfig(**config)
        self.num_envs = self._config.num_envs
        self.action_space = gym.spaces.Box(
                low=-self._config.action_range,
                high=self._config.action_range,
                shape=(1,),
                dtype=np.float32)
        self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self._player_state = phys.PlayerState(
            **{k: np.stack([v
                            for _ in range(self.num_envs)])
               for k, v in _SIMPLE_INITIAL_STATE.items()})

    @property
    def _obs_scale(self):
        return [self._config.time_limit, 90]

    def _get_obs(self):
        return np.stack([self._time_remaining, self._yaw], axis=1) / self._obs_scale

    def _get_obs_at(self, index):
        return np.array([self._time_remaining[index], self._yaw[index]]) / self._obs_scale

    def vector_reset(self):
        self._time_remaining = np.full((self.num_envs,), self._config.time_limit, dtype=np.float32)
        self._yaw = np.full((self.num_envs,), INITIAL_YAW_ZERO, dtype=np.float32)

        return self._get_obs()

    def reset_at(self, index):
        self._time_remaining[index] = self._config.time_limit
        self._yaw[index] = INITIAL_YAW_ZERO

        return self._get_obs_at(index)

    def vector_step(self, actions):
        max_yaw_delta = _MAX_YAW_SPEED * self._config.time_delta

        mouse_x_action = np.concatenate(actions) * max_yaw_delta / self._config.action_range
        self._yaw = self._yaw + mouse_x_action

        pitch = np.zeros((self.num_envs,), dtype=np.float32)
        roll = np.zeros((self.num_envs,), dtype=np.float32)
        fmove = np.full((self.num_envs,), 800., dtype=np.float32)
        smove = np.zeros((self.num_envs,), dtype=np.float32)
        button2 = np.full((self.num_envs,), False)
        time_delta = np.full((self.num_envs,), self._config.time_delta)

        inputs = phys.Inputs(yaw=self._yaw, pitch=pitch, roll=roll, fmove=fmove, smove=smove,
                             button2=button2, time_delta=time_delta)

        next_player_state = phys.apply(inputs, self._player_state)

        reward = self._config.time_delta * np.linalg.norm(next_player_state.vel[:, :2], axis=1)

        self._time_remaining -= self._config.time_delta
        done = self._time_remaining < 0

        if np.any(np.isnan(reward)):
            print(reward)
        return self._get_obs(), reward, done, [{} for i in range(self.num_envs)]

    def get_unwrapped(self):
        return []


