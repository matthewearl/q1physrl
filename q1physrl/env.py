import gym.spaces
import numpy as np
import ray.rllib

import pyquake.client

from . import phys


_TIME_DELTA = np.float32(0.014)
_FMOVE_MAX = np.float32(800)  # units per second
_SMOVE_MAX = np.float32(700)  # units per second
_YAW_SPEED = np.float32(360)  # degrees per second
_INITIAL_STATE = {'z_pos': np.float32(32.82),
                  'vel': np.array([0, 0, -12], dtype=np.float32),
                  'on_ground': np.bool(False),
                  'jump_released': np.bool(True)}
_INITIAL_YAW = np.float32(90)
_TIME_LIMIT = 5.  # seconds


class PhysEnv(ray.rllib.env.VectorEnv):
    _player_state: phys.PlayerState
    _yaw: np.ndarray
    _time: np.ndarray
    _step_num: int

    def _get_obs(self):
        ps = self._player_state

        # As per the network protocol, velocity is quantized to 16 ups, and position is quantized to
        # 0.125 units.
        vel = (ps.vel / 16).astype(int) * 16  
        z_pos = (ps.z_pos * 8).astype(int) / 8

        return np.concatenate([self._yaw[:, None], z_pos[:, None], vel], axis=1)

    def _get_obs_at(self, index):
        ps = self._player_state
        return np.concatenate([self._yaw[index][None], ps.z_pos[index][None], ps.vel[index]], axis=0)

    def __init__(self, config):
        self.num_envs = config['num_envs']
        self.action_space = gym.spaces.MultiDiscrete([3, 2])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(5,), dtype=np.float32)
        self._step_num = 0
        self.vector_reset()

    def vector_reset(self):
        self._player_state = phys.PlayerState(
            **{k: np.stack([v for _ in range(self.num_envs)]) for k, v in _INITIAL_STATE.items()})

        self._yaw = np.full((self.num_envs,), _INITIAL_YAW, dtype=np.float32)
        self._time = np.zeros((self.num_envs,), np.float32)

        return self._get_obs()

    def reset_at(self, index):
        for k, v in _INITIAL_STATE.items():
            getattr(self._player_state, k)[index] = v
        self._yaw[index] = _INITIAL_YAW
        self._time[index] = 0.

        return self._get_obs_at(index)

    def vector_step(self, actions):
        actions = np.stack(actions)

        #next_yaw = self._yaw + _TIME_DELTA * _YAW_SPEED * (actions[:, 0] - 1)
        next_yaw = self._yaw
        smove = _SMOVE_MAX * (actions[:, 0] - 1)
        fmove = _FMOVE_MAX * actions[:, 1]

        pitch = np.zeros((self.num_envs,), dtype=np.float32)
        roll = np.zeros((self.num_envs,), dtype=np.float32)
        button2 = np.zeros((self.num_envs,), dtype=np.bool)
        time_delta = np.full((self.num_envs,), _TIME_DELTA)
        
        inputs = phys.Inputs(yaw=next_yaw, pitch=pitch, roll=roll, fmove=fmove, smove=smove,
                             button2=button2, time_delta=time_delta)

        vel_before = self._player_state.vel[:, 1]
        self._player_state = phys.apply(inputs, self._player_state)

        #reward = _TIME_DELTA * self._player_state.vel[:, 1]
        reward = self._player_state.vel[:, 1] - vel_before
        self._time += _TIME_DELTA
        done = self._time > _TIME_LIMIT
        
        self._step_num += 1

        return self._get_obs(), reward, done, [{} for _ in range(self.num_envs)]
        
    def get_unwrapped(self):
        return []


def _make_observation(client):
    pass


def _apply_action(client, action):
    pass


async def eval(port, trainer):
    client = await pyquake.client.AsyncClient.connect("localhost", port)

    try:
        demo = self._client.record_demo()
        await client.wait_until_spawn()
        for i in range(358):
            obs = _make_observation(client)
            action = trainer.compute_action(obs)
            _apply_action(client, action)
    finally:
        await self._client.disconnect()
