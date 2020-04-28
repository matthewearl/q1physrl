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
from typing import Optional, Tuple, Union

import gym.spaces
import numpy as np

from . import phys


__all__ = (
    'ActionDecoder',
    'Config',
    'get_obs_scale',
    'INITIAL_YAW_ZERO',
    'Key',
    'Obs',
    'PhysEnv',
    'VectorPhysEnv',
)


# These are chosen to match the initial state of the player on the 100m map.
_INITIAL_STATE = {'z_pos': np.float32(32.843201),
                  'vel': np.array([0, 0, -12], dtype=np.float32),
                  'on_ground': np.bool(False),
                  'jump_released': np.bool(True)}
INITIAL_YAW_ZERO = np.float32(90)


class Key(enum.IntEnum):
    """Enumeration of input keys.

    These correspond with action vector indices, eg. action[Key.FORWARD] indicates the agent's desire to move forward.

    Note that the mouse x action (by default continuous) is also included in the action vector, but it isn't included
    here.

    """
    STRAFE_LEFT = 0
    STRAFE_RIGHT = enum.auto()
    FORWARD = enum.auto()
    JUMP = enum.auto()   # Not used if allow_jump or auto_jump is true


class Obs(enum.IntEnum):
    """

    """

    TIME_LEFT = 0
    YAW = enum.auto()
    Z_POS = enum.auto()
    X_VEL = enum.auto()
    Y_VEL = enum.auto()
    Z_VEL = enum.auto()


# These are just used for calculation of the Config.action_range default value.
_DEFAULT_TIME_DELTA = np.float32(0.014)
_MAX_YAW_SPEED = np.float32(2 * 360)  # degrees per second


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for a PhysEnv / VectorPhysEnv.

    Attributes:
        num_envs: Number of environments.  Must be None iff used with a `PhysEnv`.
        zero_start_prob: Probability of a given episode being a zero start, ie the probability of not having randomized
            initial values.  Higher values give more accurate zero start metrics, but may hinder exploration.
        initial_yaw_range: When not doing a zero start, gives the range of yaw values that can be taken on the first
            step.
        max_initial_speed: When not doing a zero start, gives the maximum initial speed along the ground plane.
        time_delta: Time of each frame.
        time_limit: Episode length in seconds
        allow_yaw: Have a mouse dimension in the action space.
        action_range: Min and max bounds for the action space's mouse dimension.
        discrete_yaw_steps: How many discrete steps to use for the mouse action dimension. Use -1 for continuous.
            Does nothing if allow_yaw is false.
        speed_reward: Reward speed rather than y component of the velocity.
        fmove_max: Corresponds with the `cl_forwardspeed` cvar in Quake.  When forward is pressed, a forward facing
            component is added to the player's wish direction.  The magnitude of this component is given by this value.
        smove_max: Corresponds with the `cl_sidespeed` cvar in Quake.  Same as `fmove_max` but for side facing
            components.
        hover: No gravity, fixed initial velocity.  Use to learn air movement physics without being concerned about
            ground friction.
        key_press_delay: Minimum time in seconds between consecutive presses of the same key.  This means the relevant
            element of the action vector will be internally set to zero if there have been two key down events in the
            last `key_press_delay` seconds.  Here a "key down" event means a 0-1 transition of the action vector
            element.
        smooth_keys: On the first frame after a key is pressed, apply only half of `fmove_max`/`smove_max` to the
            relevant wish direction component.  This is applied after `key_press_delay`.
        auto_jump: Automatically press jump when near the floor.  The action space does not include the jump key in this
            case.
        allow_jump: If auto-jump is False, then permit jumping with an action.  Without this the player will be bound to
            the floor (unless `hover` is also set).

    """
    # The defaults assigned here are just for backwards compatibility of my own experiments.  See `get_default` for the
    # real defaults.
    num_envs: Optional[int]
    zero_start_prob: float
    initial_yaw_range: Tuple[float, float]
    max_initial_speed: float
    time_delta: float = 0.014  # Rules state 1/72, but 0.014 is the default for backwards compatibility.
    time_limit: float = 5
    allow_yaw: bool = True
    action_range: float = _MAX_YAW_SPEED * _DEFAULT_TIME_DELTA
    discrete_yaw_steps: int = -1
    speed_reward: bool = False
    fmove_max: float = 800.
    smove_max: float = 700.
    hover: bool = False
    key_press_delay: float = 0.3
    smooth_keys: bool = False
    auto_jump: bool = False
    allow_jump: bool = True

    @classmethod
    def get_default(cls):
        """These are the defaults that are used when an envirionment is made via `gym.make`."""
        return cls(
            num_envs=None,
            allow_jump=True,
            allow_yaw=True,
            auto_jump=False,
            discrete_yaw_steps=-1,
            fmove_max=800,  # These fmove_max and smove_max defaults seem to work well.
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

    def conforms_to_rules(self):
        """Indicate whether the config would be permitted under speed running rules.
        
        See http://quake.speeddemosarchive.com/quake/rules.html for list of normal rules.  Clearly we're breaking rules
        about tool assistance, but by "permitted" here I mean that applying the generated movements to a legally
        configured Quake would yield the same results.

        """
        return self.time_delta == 1. / 72 and not self.hover


class ActionDecoder:
    """Convert a sequence of actions into a sequence of move commands
    
    Like `VectorEnv`, this class is vectorized, ie. it actually maps actions for many environments into many movement
    commands, however for simplicity the docstrings refer to variables in the singular.

    The decoder performs these tasks:
        - Scales the mouse x action such that the number of degrees turned per second is between `-_MAX_YAW_SPEED` and
          `_MAX_YAW_SPEED`.
        - Rate limit key presses.  See `Config.key_press_delay` for details of how this works.
        - Smooth transitions of key presses.  See `Config.smooth_keys` for details of how this works.
        - Automatically send a jump command when near the floor if `auto_jump` is set.

    The decoder is stateful (in order to do things like rate limit key presses) as such it needs to be reset along with
    the corresponding environment.

    """
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
        """Take an action vector and map it to a move command."""
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
        """Reset the state of the action decoder.

        Called before starting a new episode.

        """
        self._last_key_press_time = np.full((self._config.num_envs, self._num_keys),
                                            -self._config.key_press_delay)
        self._last_keys = np.full((self._config.num_envs, self._num_keys), False)

        self._yaw = np.array(yaw)

    def reset_at(self, index, yaw):
        """Reset the state of a single element of the action decoder.

        Called before starting a new episode.

        """
        self._last_key_press_time[index] = -self._config.key_press_delay
        self._last_keys[index] = False
        self._yaw[index] = yaw


def get_obs_scale(config):
    """Observation values are normalized by dividing by these values before being returned."""
    return [config.time_limit, 90., 100, 200, 200, 200]


class PhysEnv(gym.Env):
    """Quake 1 physics environment.
    
    The environment models the player movement physics of Quake.  To keep things simple, the world is treated as a flat
    plane with no obstacles, with the objective of moving as far along the y-axis as possible in a ten second time
    limit.  The physics modelling is accurate enough that it can run the 100m practice map in the real game, after being
    trained on the simulated environment.

    The below describes the default behaviour of the environment but many of the parameters can be tweaked via the
    config.  See `Config` for details.

    The environment has a tuple action space consisting of:
    - Four discrete "keys", corresponding with left, right, forward, and jump.
    - A continuous dimension indicating how much the yaw should change in the next frame.  This says how far the player
      should turn left or right in this frame.

    Actions are decoded by the `ActionDecoder` class which translates action vectors into movement commands to be sent
    to the Quake simulant.  See `ActionDecoder` for details.

    The observation space is a six dimensional box and indicates:
    - Time remaining in the episode.
    - Player's current yaw angle.
    - Z (height) position.
    - X, Y, and Z velocity.

    Observations are normalized to be approximately between zero and one.

    Rewards correspond with how far the player travels along the Y-axis in the given frame (note that Z is up in the
    Quake coordinate system, which the replica engine also uses).  The positive Y direction corresponds with running
    forwards on the 100m map.

    In order to aid exploration the following aspects of the initial state are randomized:
    - Time remaining.
    - Player velocity.
    - Player yaw.
    - Player angle.

    1% of episodes start with a fixed initial state chosen to match that of a freshly spawned player in the real game.
    If this is the case, then the info dict returned by `step` indicates as such with the `zero_start` field.  The
    purpose of this is that it allows one to implement metrics which indicate performance in the real game.

    """
    def __init__(self, config: Union[Config, dict]):
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

    This is the vectorized form of `PhysEnv` --- see its docstring for details.
    
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

        self._action_decoder = ActionDecoder(self._config)
        self.action_space = self._action_decoder.action_space
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

        self._action_decoder.vector_reset(self._yaw)

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
        if self._config.hover:
            speed = 320

        move_angle = np.random.uniform(2 * np.pi)
        if self._config.hover:
            move_angle = np.pi / 2

        self.player_state.vel[index, 0] = speed * np.cos(move_angle)
        self.player_state.vel[index, 1] = speed * np.sin(move_angle)

        self._action_decoder.reset_at(index, self._yaw[index])

        return self._get_obs_at(index)

    def vector_step(self, actions):
        if self._config.hover:
            self.player_state.vel[:, 2] = 0
            self.player_state.z_pos[:] = 100

        z_vel = self.player_state.vel[:, 2]
        self._yaw, smove, fmove, jump = self._action_decoder.map(actions, z_vel, self._time_remaining)

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
    entry_point='q1physrl_env.env:PhysEnv',
    nondeterministic=False,
    kwargs={'config': Config.get_default()},
)
