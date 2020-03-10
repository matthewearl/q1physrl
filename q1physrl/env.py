import asyncio
import dataclasses
import enum
from typing import Tuple

import gym.spaces
import numpy as np
import ray.rllib

import pyquake.client

from . import phys


__all__ = (
    'ActionToMove',
    'Config',
    'eval',
    'eval_coro',
    'Key',
    'Obs',
    'PhysEnv',
    'SinglePhysEnv',
)


_TIME_DELTA = np.float32(0.014)
_MAX_YAW_SPEED = np.float32(2 * 360)  # degrees per second
_INITIAL_STATE = {'z_pos': np.float32(32.843201),
                  'vel': np.array([0, 0, -12], dtype=np.float32),
                  'on_ground': np.bool(False),
                  'jump_released': np.bool(True)}
_INITIAL_YAW_ZERO = np.float32(90)



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
    auto_jump: bool
    initial_yaw_range: Tuple[float, float]
    max_initial_speed: float
    zero_start_prob: float
    action_range: float = _MAX_YAW_SPEED * _TIME_DELTA
    time_limit: float = 5
    key_press_delay: float = 0.3
    discrete_yaw_steps: int = -1    # -1 = continuous
    speed_reward: bool = False  # reward speed rather than velocity in y dir.
    fmove_max: float = 800.  # units per second
    smove_max: float = 700.  # units per second
    hover: bool = False     # No gravity, fixed initial speed


class ActionToMove:
    """Convert a sequence of actions into a sequence of move commands"""
    _last_key_press_time: np.ndarray
    _last_keys: np.ndarray
    _yaw: np.ndarray

    def __init__(self, config: Config):
        self._config = config
        self._num_action_keys = len(Key) - 1 if self._config.auto_jump else len(Key)

    def _fix_actions(self, actions):
        # trainer.compute_actions() and trainer.train() return data in slightly different formats.
        return np.array([[np.ravel(x)[0] for x in a] for a in actions])

    def map(self, actions, z_vel, time_remaining):
        actions = self._fix_actions(actions)
        key_actions = actions[:, :self._num_action_keys].astype(np.int)

        max_yaw_delta = _MAX_YAW_SPEED * _TIME_DELTA
        yaw_steps = self._config.discrete_yaw_steps
        if yaw_steps == -1:
            mouse_x_action = actions[:, self._num_action_keys] * max_yaw_delta / self._config.action_range
        else:
            mouse_x_action = (actions[:, self._num_action_keys] - yaw_steps) * max_yaw_delta / yaw_steps

        elapsed = (self._config.time_limit - time_remaining[:, None] >=
                    self._last_key_press_time + self._config.key_press_delay)
        keys = key_actions & (elapsed | self._last_keys)
        self._last_key_press_time = np.where(
            keys & ~self._last_keys,
            (self._config.time_limit - time_remaining[:, None]),
            self._last_key_press_time
        )
        self._last_keys = keys

        keys_int = keys.astype(np.int)
        self._yaw = self._yaw + mouse_x_action
        strafe_keys = keys_int[:, Key.STRAFE_RIGHT] - keys_int[:, Key.STRAFE_LEFT]
        smove = np.float32(self._config.smove_max) * strafe_keys
        fmove = np.float32(self._config.fmove_max) * keys_int[:, Key.FORWARD]
        if self._config.auto_jump:
            jump = z_vel <= 16
        else:
            jump = keys_int[:, Key.JUMP]

        return self._yaw, smove.astype(np.int), fmove.astype(np.int), jump.astype(np.bool)

    def vector_reset(self, yaw):
        self._last_key_press_time = np.full((self._config.num_envs, self._num_action_keys), -self._config.key_press_delay)
        self._last_keys = np.full((self._config.num_envs, self._num_action_keys), False)

        self._yaw = np.array(yaw)

    def reset_at(self, index, yaw):
        self._last_key_press_time[index] = -self._config.key_press_delay
        self._last_keys[index] = False
        self._yaw[index] = yaw


def _get_obs_scale(config): 
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
        num_keys = len(Key) - 1 if self._config.auto_jump else len(Key)
        
        if self._config.discrete_yaw_steps == -1:
            yaw_action_space = gym.spaces.Box(low=-self._config.action_range, high=self._config.action_range,
                                              shape=(1,), dtype=np.float32)
        else:
            yaw_action_space = gym.spaces.Discrete(2 * self._config.discrete_yaw_steps + 1)
        self.action_space = gym.spaces.Tuple([*(gym.spaces.Discrete(2) for _ in range(num_keys)), yaw_action_space]) 
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(6,), dtype=np.float32)
        self.reward_range = (-1000 * _TIME_DELTA, 1000 * _TIME_DELTA)
        self.metadata = {}

        self._obs_scale = _get_obs_scale(self._config)

        self._action_to_move = ActionToMove(self._config)
        self._step_num = 0
        self.vector_reset()

    def vector_reset(self):
        self.player_state = phys.PlayerState(
            **{k: np.stack([v for _ in range(self.num_envs)]) for k, v in _INITIAL_STATE.items()})

        self._zero_start = np.random.random(size=(self.num_envs,)) < self._config.zero_start_prob

        self._yaw = np.where(self._zero_start,
                             _INITIAL_YAW_ZERO,
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
        self._yaw[index] = (_INITIAL_YAW_ZERO if self._zero_start[index] else
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
        time_delta = np.full((self.num_envs,), _TIME_DELTA)
        
        inputs = phys.Inputs(yaw=self._yaw, pitch=pitch, roll=roll, fmove=fmove, smove=smove,
                             button2=button2, time_delta=time_delta)

        self.player_state = phys.apply(inputs, self.player_state)

        if self._config.speed_reward:
            reward = _TIME_DELTA * np.linalg.norm(self.player_state.vel[:, :2], axis=1)
        else:
            reward = _TIME_DELTA * self.player_state.vel[:, 1]

        self._time_remaining -= _TIME_DELTA
        done = self._time_remaining < 0
        
        self._step_num += 1

        return self._get_obs(), reward, done, [{'zero_start': self._zero_start[i]} for i in range(self.num_envs)]
        
    def get_unwrapped(self):
        return []


def _make_observation(client, time_remaining, config):
    yaw = 180 * client.angles[1] / np.pi
    vel = np.array(client.velocity)
    z_pos = client.player_origin[2]
    obs_scale = _get_obs_scale(config)
    return np.concatenate([[time_remaining], [yaw], [z_pos], vel]) / obs_scale


def _apply_action(client, action_to_move, action, time_remaining):
    (yaw,), (smove,), (fmove,), (jump,) = action_to_move.map([[a[0] for a in action]],
                                                             np.float32(client.velocity[2])[None],
                                                             np.float32(time_remaining)[None])
    yaw *= np.pi / 180

    buttons = np.where(jump, 2, 0)
    client.move(pitch=0, yaw=yaw, roll=0, forward=fmove, side=smove,
                up=0, buttons=buttons, impulse=0)


async def eval_coro(config, port, trainer, demo_file):
    client = await pyquake.client.AsyncClient.connect("localhost", port)
    config = Config({**config, 'num_envs': 1})
    action_to_move = ActionToMove(config)
    action_to_move.vector_reset(np.array([_INITIAL_YAW_ZERO]))

    obs_list = []
    action_list = []

    try:
        demo = client.record_demo()
        await client.wait_until_spawn()
        client.move(*client.angles, 0, 0, 0, 0, 0)
        await client.wait_for_movement(client.view_entity)
        start_time = client.time
        time_remaining = None
        while time_remaining is None or time_remaining >= 0:
            time_remaining = config['time_limit'] - (client.time - start_time)
            obs = _make_observation(client, time_remaining, config)
            obs_list.append(obs)
            action = trainer.compute_action(obs)
            action_list.append(action)

            _apply_action(client, action_to_move, action, time_remaining)
            await client.wait_for_movement(client.view_entity)

        demo.stop_recording()
        demo.dump(demo_file)

    finally:
        await client.disconnect()

    return obs_list, action_list

