import asyncio

import gym.spaces
import numpy as np
import ray.rllib

import pyquake.client

from . import phys


__all__ = (
    'eval',
    'eval_coro',
    'PhysEnv',
)


_TIME_DELTA = np.float32(0.014)
_FMOVE_MAX = np.float32(800)  # units per second
_SMOVE_MAX = np.float32(700)  # units per second
_YAW_SPEED = np.float32(360)  # degrees per second
_INITIAL_STATE = {'z_pos': np.float32(32.843201),
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


    def _round_vel(self, v):
        # See sv_main.c : SV_WriteClientdataToMessage
        return (v / 16).astype(int) * 16

    def _round_origin(self, o):
        # See common.c : MSG_WriteCoord

        # We should not send a changed origin if the position has not changed by more than 0.1.
        # See `miss` variable in `SV_WriteEntitiesToClient`.
        return np.round(o * 8) / 8

    def _get_obs(self):
        ps = self._player_state

        vel = self._round_vel(ps.vel)
        z_pos = self._round_origin(ps.z_pos)

        return np.concatenate([self._yaw[:, None], z_pos[:, None], vel], axis=1)

    def _get_obs_at(self, index):
        ps = self._player_state
        z_pos = self._round_origin(ps.z_pos[index])
        vel = self._round_vel(ps.vel[index])
        return np.concatenate([self._yaw[index][None], z_pos[None], vel], axis=0)

    def __init__(self, config):
        self.num_envs = config['num_envs']
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2])
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

        self._yaw = self._yaw + _TIME_DELTA * _YAW_SPEED * (actions[:, 0] - 1)
        smove = _SMOVE_MAX * (actions[:, 1] - 1)
        fmove = _FMOVE_MAX * actions[:, 2]

        pitch = np.zeros((self.num_envs,), dtype=np.float32)
        roll = np.zeros((self.num_envs,), dtype=np.float32)
        button2 = np.zeros((self.num_envs,), dtype=np.bool)
        time_delta = np.full((self.num_envs,), _TIME_DELTA)
        
        inputs = phys.Inputs(yaw=self._yaw, pitch=pitch, roll=roll, fmove=fmove, smove=smove,
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
    yaw = 180 * client.angles[1] / np.pi
    vel = np.array(client.velocity)
    z_pos = client.player_origin[2]
    return np.concatenate([[yaw], [z_pos], vel])


def _apply_action(client, action):
    yaw = (180 * client.angles[1] / np.pi) + _TIME_DELTA * _YAW_SPEED * (action[0] - 1)
    yaw *= np.pi / 180
    smove = int(_SMOVE_MAX * (action[1] - 1))
    fmove = int(_FMOVE_MAX * action[2])
    client.move(pitch=0, yaw=yaw, roll=0, forward=fmove, side=smove,
                up=0, buttons=0, impulse=0)


async def eval_coro(port, trainer, demo_fname):
    client = await pyquake.client.AsyncClient.connect("localhost", port)

    obs_list = []
    action_list = []

    try:
        demo = client.record_demo()
        await client.wait_until_spawn()
        client.move(*client.angles, 0, 0, 0, 0, 0)
        await client.wait_for_movement(client.view_entity)
        for _ in range(358):
            obs = _make_observation(client)
            obs_list.append(obs)
            action = trainer.compute_action(obs)
            action_list.append(action)
            _apply_action(client, action)
            await client.wait_for_movement(client.view_entity)

        demo.stop_recording()
        with open(demo_fname, 'wb') as f:
            demo.dump(f)

    finally:
        await client.disconnect()

    return np.array(obs_list), np.array(action_list)


def eval(port: int, trainer: ray.rllib.agents.Trainer, demo_fname: str):
    """Connect to a quake server and record a demo using a trained model.
    
    The quake server must be running with +sync_movements 1.

    """
    return asyncio.run(eval_coro(port, trainer, demo_fname))

