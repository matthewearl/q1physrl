# Copyright (c) 2020 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Quake 1 physics gym environment definitions.

Importing this module will register the 'Q1PhysEnv-v0' environment.  Alternatively, you can use the `PhysEnv` or
`VectorPhysEnv` classes directly.

"""



import dataclasses
import enum
from typing import Optional, Tuple

import gym.spaces
import numpy as np

from . import phys


__all__ = (
    'ActionToMove',
    'Config',
    'DEFAULT_CONFIG',
    'get_obs_scale',
    'INITIAL_YAW_ZERO',
    'Key',
    'Obs',
    'PhysEnv',
    'VectorPhysEnv',
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
    """Configuration for a PhysEnv / VectorPhysEnv.

    Attributes:
        num_envs: Number of environments.  Must be None iff used with a `PhysEnv`.
        auto_jump: Automatically press jump when near the floor.
        initial_yaw_range: Range of yaw values on the first step.
        max_initial_speed: Maximum speed along the ground plane on the first step.
        zero_start_prob: Probability of starting with zero speed at the first step.
        time_delta: Time of each frame.
        action_range: Continuous action value range.
        time_limit: Episode length in seconds
        key_press_delay: Minimum time between consecutive presses of the same key.
        discrete_yaw_steps: How many discrete steps to use for the yaw action. Use -1 for a continuous yaw.  Does
            nothing if allow_yaw is false.
        speed_reward: Reward speed rather than velocity in y dir.
        fmove_max: Corresponds with cl_forwardspeed.
        smove_max: Corresponds with cl_sidespeed.
        hover: No gravity, fixed initial speed
        smooth_keys: Register half a key press on transitions
        allow_jump: If auto-jump is False, then permit jumping with an action
        allow_yaw: Have yaw dimension in action space.
    """
    num_envs: Optional[int]
    auto_jump: bool
    initial_yaw_range: Tuple[float, float]
    max_initial_speed: float
    zero_start_prob: float
    time_delta: float = 0.014  # Rules state 1/72, but 0.014 is the default for backwards compatibility.
    action_range: float = _MAX_YAW_SPEED * _DEFAULT_TIME_DELTA
    time_limit: float = 5
    key_press_delay: float = 0.3
    discrete_yaw_steps: int = -1
    speed_reward: bool = False
    fmove_max: float = 800.
    smove_max: float = 700.
    hover: bool = False
    smooth_keys: bool = False
    allow_jump: bool = True
    allow_yaw: bool = True

    def conforms_to_rules(self):
        """Indicate whether the config would be permitted under speed running rules."""
        return self.time_delta == 1. / 72 and not self.hover


DEFAULT_CONFIG = Config(
    num_envs=None,
    allow_jump=True,
    allow_yaw=True,
    auto_jump=False,
    discrete_yaw_steps=-1,
    fmove_max=800,
    smove_max=1060,
    hover=False,
    initial_yaw_range=(0, 360),
    key_press_delay=0.3,
    max_initial_speed=700,
    smooth_keys=True,
    speed_reward=False,
    time_delta=1. / 72,
    time_limit=10.,
    zero_start_prob=0.01,
)


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


class PhysEnv(gym.Env):
    """Quake 1 physics environment"""
    def __init__(self, config):
        if isinstance(config, dict):
            config = Config(**config)
        if config.num_envs is not None:
            assert config.num_envs is None, "num_envs must be None for PhysEnv"
        config = dataclasses.replace(config, num_envs=1)

        self._env = VectorPhysEnv(config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def step(self, action):
        (obs,), (reward,), (done,), (info,) = self._env.vector_step([action])
        return obs, reward, done, info

    def reset(self):
        (obs,) = self._env.vector_reset()
        return obs


# For direct use with rllib VectorPhysEnv needs to inherit from VectorEnv, however when being used by PhysEnv a base
# class is not required.
try:
    from ray.rllib.env import VectorEnv
except ImportError:
    VectorEnv = object


class VectorPhysEnv(VectorEnv):
    """Vectorized Quake 1 physics environment.
    
    This implements the `ray.rllib.env.VectorEnv` interface, however we don't explicitly subclass it so that `PhysEnv`
    can be used without depending on rllib.

    """
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
        if isinstance(config, dict):
            config = Config(**config)
        self._config = config
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


gym.envs.registration.register(
    id='Q1PhysEnv-v0',
    entry_point='q1physrl_env.env.PhysEnv',
    nondeterministic=False,
    kwargs={'config': DEFAULT_CONFIG},
)
