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
This module applies Quake 1 player physics.

It applies inputs (`:class:.Inputs`) to the player's current state (`:class:.PlayerState`) and produces the next frame's
player state.  Inputs / player state are vectorized (represented by numpy arrays).

The inputs correspond with the inputs over the network layer in Quake 1

"""


__all__ = (
    'apply'
    'Inputs',
    'PlayerState',
)


import dataclasses

import numpy as np
import pandas as pd


_MAX_SPEED = np.float32(320)
_ACCELERATE = np.float32(10)
_FRICTION = np.float32(4)
_STOP_SPEED = np.float32(100)
_JUMP_SPEED = np.float32(270)
_GRAVITY = np.float32(800)
_FLOOR_HEIGHT = np.float32(24.03125)  # 24 + DIST_EPSILON


def _angle_vectors(yaw, pitch, roll):
    # See mathlib.c : AngleVectors().  The up axis, and third dimension are ignored since we don't actually use them.
    sy = np.sin(yaw * np.pi / 180.)
    cy = np.cos(yaw * np.pi / 180.)
    sp = np.sin(pitch * np.pi / 180.)
    cp = np.cos(pitch * np.pi / 180.)
    sr = np.sin(roll * np.pi / 180.)
    cr = np.cos(roll * np.pi / 180.)
    
    return np.stack([[cp * cy, (-1*sr*sp*cy+-1*cr*-sy)],
                     [cp * sy, (-1*sr*sp*sy+-1*cr*cy)]]).transpose((2, 0, 1))


def _accelerate(velocity, wish_speed, wish_dir, on_ground, time_delta):
    # See sv_user.c : SV_Accelerate and SV_AirAccelerate
    current_speed = np.einsum('ij,ij->i', velocity, wish_dir)
    
    clipped_wish_speed = np.where((wish_speed > 30) & ~on_ground,
                                  30,
                                  wish_speed)
    
    add_speed = np.maximum(0, clipped_wish_speed - current_speed)
    accel_speed = np.minimum(_ACCELERATE * time_delta * wish_speed, add_speed)
    
    return velocity + accel_speed[:, None] * wish_dir


def _user_friction(h_vel, time_delta):
    # See sv_user.c : SV_UserFriction
    speed = np.linalg.norm(h_vel, axis=1)
    control = np.maximum(speed, _STOP_SPEED)
    new_speed = speed - time_delta * control * _FRICTION
    new_speed = np.maximum(0, new_speed)
    
    return np.where((speed > 0)[:, None], h_vel * (new_speed / speed)[:, None], h_vel)


def _air_move(yaw, pitch, roll, fmove, smove, on_ground, time_delta, h_vel):
    # See sv_user.c : SV_AirMove
    move = np.stack([fmove, smove], axis=1)
    
    wish_vel = np.einsum('ijk,ik->ij', _angle_vectors(yaw, pitch, roll), move)
    unclipped_wish_speed = np.linalg.norm(wish_vel, axis=1)
    wish_dir = np.where(unclipped_wish_speed[:, None] > 0,
                        wish_vel / unclipped_wish_speed[:, None],
                        wish_vel)

    wish_speed = np.minimum(_MAX_SPEED, unclipped_wish_speed)
    wish_vel *= np.where(unclipped_wish_speed > 0,
                         (wish_speed / unclipped_wish_speed),
                         1)[:, None]
    
    h_vel = np.where(on_ground[:, None], _user_friction(h_vel, time_delta), h_vel)
    return _accelerate(h_vel, wish_speed, wish_dir, on_ground, time_delta)


def _do_z_physics(jump_pressed, time_delta, z_pos, z_vel, on_ground, jump_released):
    z_vel = z_vel.copy()

    # The jump logic here is ported from client.qc : PlayerJump.
    jump_released = jump_released.copy()
    jump_released |= ~jump_pressed
    do_jump = on_ground & jump_pressed & jump_released
    z_vel += do_jump * _JUMP_SPEED

    # See sv_phys.c : SV_AddGravity
    z_vel -= _GRAVITY * time_delta

    # See sv_phys.c : SV_FlyMove.  Vastly simplified as it only deals with a single ground plane.  Doesn't perfectly
    # replicate the game engine as the engine stops slightly above the ground, in a manner that I haven't reverse
    # engineered yet (around 1e-2 units away).
    z_pos = z_pos + time_delta * z_vel
    on_ground = z_pos < _FLOOR_HEIGHT
    z_pos = np.where(on_ground, _FLOOR_HEIGHT, z_pos)
    z_vel = np.where(on_ground, 0, z_vel)

    return z_pos, z_vel, on_ground, jump_released


@dataclasses.dataclass
class Inputs:
    yaw: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    fmove: np.ndarray
    smove: np.ndarray
    button2: np.ndarray
    time_delta: np.ndarray

    @classmethod
    def from_df(cls, df):
        return cls(df.yaw.to_numpy(), df.pitch.to_numpy(), df.roll.to_numpy(),
                   df.fmove.to_numpy(), df.smove.to_numpy(),
                   df.button2.to_numpy() > 0, df.host_frametime.to_numpy())
    def to_df(self):
        return pd.DataFrame({'yaw': self.yaw, 'pitch': self.pitch, 'roll': self.roll,
                             'fmove': self.fmove, 'smove': self.smove,
                             'button2': self.button2, 'host_frametime': self.time_delta})

    
@dataclasses.dataclass
class PlayerState:
    z_pos: np.ndarray
    vel: np.ndarray
    on_ground: np.ndarray
    jump_released: np.ndarray
    
    @classmethod
    def from_df(cls, df):
        return cls(df.z.to_numpy(),
                   np.stack([df.velx.to_numpy(), df.vely.to_numpy(), df.velz.to_numpy()], axis=1),
                   df.onground.to_numpy() > 0,
                   df.jumpreleased.to_numpy() > 0,)
    
    def to_df(self):
        return pd.DataFrame({'z': self.z_pos,
                             'velx': self.vel[:, 0], 'vely': self.vel[:, 1], 'velz': self.vel[:, 2],
                             'onground': self.on_ground,
                             'jumpreleased': self.jump_released,
                            })

    @classmethod
    def concatenate(cls, player_states):
        dicts = [dataclasses.asdict(ps) for ps in player_states]
        return cls(**{f.name: np.concatenate([d[f.name] for d in dicts])
                        for f in dataclasses.fields(cls)})


def apply(inputs: Inputs, player_state: PlayerState) -> PlayerState:
    vel = player_state.vel.copy()
    z_pos = player_state.z_pos.copy()
    on_ground = player_state.on_ground.copy()
    jump_released = player_state.jump_released.copy()
    
    vel[:, :2] = _air_move(inputs.yaw, inputs.pitch, inputs.roll,
                           inputs.fmove, inputs.smove, player_state.on_ground,
                           inputs.time_delta, vel[:, :2])
    
    z_pos, vel[:, 2], on_ground, jump_released = _do_z_physics(
        inputs.button2, inputs.time_delta, z_pos, vel[:, 2], on_ground, jump_released)
    
    return PlayerState(z_pos, vel, on_ground, jump_released)
