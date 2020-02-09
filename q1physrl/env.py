import asyncio
import enum

import gym.spaces
import numpy as np
import ray.rllib

import pyquake.client

from . import phys


__all__ = (
    'ActionToMove',
    'eval',
    'eval_coro',
    'Key',
    'Obs',
    'PhysEnv',
)


_TIME_DELTA = np.float32(0.014)
_FMOVE_MAX = np.float32(800)  # units per second
_SMOVE_MAX = np.float32(700)  # units per second
_MAX_YAW_SPEED = np.float32(2 * 360)  # degrees per second
_INITIAL_STATE = {'z_pos': np.float32(32.843201),
                  'vel': np.array([0, 0, -12], dtype=np.float32),
                  'on_ground': np.bool(False),
                  'jump_released': np.bool(True)}
_INITIAL_YAW = np.float32(90)
_TIME_LIMIT = 5.  # seconds

_OBS_SCALE = [_TIME_LIMIT, 90., 100, 200, 200, 200]

_KEY_PRESS_DELAY = 0.3   # minimum time between consecutive presses of the same key


class Key(enum.IntEnum):
    STRAFE_LEFT = 0
    STRAFE_RIGHT = enum.auto()
    FORWARD = enum.auto()


class Obs(enum.IntEnum):
    TIME_LEFT = 0
    YAW = enum.auto()
    Z_POS = enum.auto()
    X_VEL = enum.auto()
    Y_VEL = enum.auto()
    Z_VEL = enum.auto()


class ActionToMove:
    """Convert a sequence of actions into a sequence of move commands"""
    _last_key_press_time: np.ndarray
    _last_keys: np.ndarray
    _yaw: np.ndarray

    def __init__(self, num_envs):
        self._num_envs = num_envs

    def _fix_actions(self, actions):
        # trainer.compute_actions() and trainer.train() return data in slightly different formats.
        return np.array([[np.ravel(x)[0] for x in a] for a in actions])

    def map(self, actions, time):
        actions = self._fix_actions(actions)
        key_actions = actions[:, :3].astype(np.int)
        mouse_x_action = actions[:, 3]

        elapsed = time[:, None] >= self._last_key_press_time + _KEY_PRESS_DELAY
        keys = key_actions & (elapsed | self._last_keys)
        self._last_key_press_time = np.where(
            keys & ~self._last_keys,
            time[:, None],
            self._last_key_press_time
        )
        self._last_keys = keys

        keys_int = keys.astype(np.int)
        self._yaw = self._yaw + mouse_x_action
        strafe_keys = keys_int[:, Key.STRAFE_RIGHT] - keys_int[:, Key.STRAFE_LEFT]
        smove = _SMOVE_MAX * strafe_keys
        fmove = _FMOVE_MAX * keys_int[:, Key.FORWARD]

        return self._yaw, smove.astype(np.int), fmove.astype(np.int)

    def vector_reset(self):
        self._last_key_press_time = np.full((self._num_envs, len(Key)), -_KEY_PRESS_DELAY)
        self._last_keys = np.full((self._num_envs, len(Key)), False)
        self._yaw = np.full((self._num_envs,), _INITIAL_YAW, dtype=np.float32)

    def reset_at(self, index):
        self._last_key_press_time[index] = -_KEY_PRESS_DELAY
        self._last_keys[index] = False
        self._yaw[index] = _INITIAL_YAW


class PhysEnv(ray.rllib.env.VectorEnv):
    player_state: phys.PlayerState
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
        ps = self.player_state

        t = _TIME_LIMIT - self._time
        vel = self._round_vel(ps.vel)
        z_pos = self._round_origin(ps.z_pos)

        obs = np.concatenate([t[:, None], self._yaw[:, None], z_pos[:, None], vel], axis=1)
        return obs / _OBS_SCALE

    def _get_obs_at(self, index):
        ps = self.player_state
        t = _TIME_LIMIT - self._time[index]
        z_pos = self._round_origin(ps.z_pos[index])
        vel = self._round_vel(ps.vel[index])
        obs = np.concatenate([t[None], self._yaw[index][None], z_pos[None], vel], axis=0)
        return obs / _OBS_SCALE

    def __init__(self, config):
        self.num_envs = config['num_envs']
        max_yaw_delta = _MAX_YAW_SPEED * _TIME_DELTA
        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(2),
            gym.spaces.Box(low=-max_yaw_delta, high=max_yaw_delta, shape=(1,), dtype=np.float32),
        ])

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(6,), dtype=np.float32)

        self._action_to_move = ActionToMove(self.num_envs)
        self._step_num = 0
        self.vector_reset()

    def vector_reset(self):
        self.player_state = phys.PlayerState(
            **{k: np.stack([v for _ in range(self.num_envs)]) for k, v in _INITIAL_STATE.items()})

        self._yaw = np.full((self.num_envs,), _INITIAL_YAW, dtype=np.float32)
        self._time = np.zeros((self.num_envs,), np.float32)

        self._action_to_move.vector_reset()

        return self._get_obs()

    def reset_at(self, index):
        for k, v in _INITIAL_STATE.items():
            getattr(self.player_state, k)[index] = v
        self._yaw[index] = _INITIAL_YAW
        self._time[index] = 0.

        self._action_to_move.reset_at(index)

        return self._get_obs_at(index)

    def vector_step(self, actions):
        self._yaw, smove, fmove = self._action_to_move.map(actions, self._time)

        pitch = np.zeros((self.num_envs,), dtype=np.float32)
        roll = np.zeros((self.num_envs,), dtype=np.float32)
        #button2 = np.zeros((self.num_envs,), dtype=np.bool)
        button2 = self.player_state.vel[:, 2] <= 16
        time_delta = np.full((self.num_envs,), _TIME_DELTA)
        
        inputs = phys.Inputs(yaw=self._yaw, pitch=pitch, roll=roll, fmove=fmove, smove=smove,
                             button2=button2, time_delta=time_delta)

        self.player_state = phys.apply(inputs, self.player_state)

        reward = _TIME_DELTA * self.player_state.vel[:, 1]
        self._time += _TIME_DELTA
        done = self._time > _TIME_LIMIT
        
        self._step_num += 1

        return self._get_obs(), reward, done, [{} for _ in range(self.num_envs)]
        
    def get_unwrapped(self):
        return []


def _make_observation(client, start_time):
    t = _TIME_LIMIT - (client.time - start_time)
    yaw = 180 * client.angles[1] / np.pi
    vel = np.array(client.velocity)
    z_pos = client.player_origin[2]
    return np.concatenate([[t], [yaw], [z_pos], vel]) / _OBS_SCALE


def _apply_action(client, action_to_move, action, time):
    (yaw,), (smove,), (fmove,) = action_to_move.map([[a[0] for a in action]], np.float32(time)[None])
    yaw *= np.pi / 180

    buttons = np.where(client.velocity[2] <= 16, 2, 0)
    client.move(pitch=0, yaw=yaw, roll=0, forward=fmove, side=smove,
                up=0, buttons=buttons, impulse=0)


async def eval_coro(port, trainer, demo_fname):
    client = await pyquake.client.AsyncClient.connect("localhost", port)
    action_to_move = ActionToMove(1)
    action_to_move.vector_reset()

    obs_list = []
    action_list = []

    try:
        demo = client.record_demo()
        await client.wait_until_spawn()
        client.move(*client.angles, 0, 0, 0, 0, 0)
        await client.wait_for_movement(client.view_entity)
        start_time = client.time
        for _ in range(358):
            obs = _make_observation(client, start_time)
            obs_list.append(obs)
            action = trainer.compute_action(obs)
            action_list.append(action)

            _apply_action(client, action_to_move, action, client.time - start_time)
            await client.wait_for_movement(client.view_entity)

        demo.stop_recording()
        with open(demo_fname, 'wb') as f:
            demo.dump(f)

    finally:
        await client.disconnect()

    return obs_list, action_list


def eval(port: int, trainer: ray.rllib.agents.Trainer, demo_fname: str):
    """Connect to a quake server and record a demo using a trained model.
    
    The quake server must be running with +sync_movements 1.

    """
    return asyncio.run(eval_coro(port, trainer, demo_fname))

