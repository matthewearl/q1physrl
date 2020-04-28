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


import cv2
import dataclasses
import sys
from pathlib import Path

import numpy as np
import ray
from q1physrl_env import env, phys

from . import train


def parse_demo(fname):
    import pyquake

    view_entity = None
    origin = None
    origins = []
    times = []
    yaws = []

    time = None
    finish_time = None

    def patch_vec(old_vec, update):
        return tuple(v if u is None else u for v, u in zip(old_vec, update))

    with open(fname, 'rb') as f:
        for angles, msg in pyquake.proto.read_demo_file(f):
            if msg.msg_type == pyquake.proto.ServerMessageType.SETVIEW:
                view_entity = msg.viewentity
            if (msg.msg_type == pyquake.proto.ServerMessageType.SPAWNBASELINE and
                    msg.entity_num == view_entity):
                origin = msg.origin
            if (msg.msg_type == pyquake.proto.ServerMessageType.UPDATE and
                    msg.entity_num == view_entity):
                origin = patch_vec(origin, msg.origin)
            if msg.msg_type == pyquake.proto.ServerMessageType.TIME:
                time = msg.time

                origins.append(origin)
                times.append(time)
                yaws.append(angles[1])
            if msg.msg_type == pyquake.proto.ServerMessageType.INTERMISSION:
                finish_time = time

    return np.array(times), np.array(origins), np.array(yaws), finish_time


@dataclasses.dataclass
class EvalSimResult:
    time_delta: float
    player_state: phys.PlayerState
    action: np.ndarray
    obs: np.ndarray
    reward: np.ndarray
    yaw: np.ndarray
    smove: np.ndarray
    fmove: np.ndarray
    jump: np.ndarray

    @property
    def move_angle(self):
        return 180. * np.arctan2(self.player_state.vel[:, 1], self.player_state.vel[:, 0]) / np.pi

    @property
    def wish_angle(self):
        return self.yaw - (180. * np.arctan2(self.smove, self.fmove) / np.pi)

    @property
    def hypothetical_delta_speeds(self):
        """Get hypthetical speed increases for this run, were a given action taken.

        The returned array has shape (360, num_frames), with the first axis corresponding with the wish angle - move
        angle difference.  Values range from -180 to 179 degrees inclusive.

        """
        delta_speeds = []
        move_angle = self.move_angle

        for rel_wish_angle in np.arange(-180, 180):
            inputs = phys.Inputs(
                yaw=move_angle + rel_wish_angle,
                pitch=np.zeros_like(move_angle),
                roll=np.zeros_like(move_angle),
                fmove=np.full_like(move_angle, 800.),
                smove=np.zeros_like(move_angle),
                button2=self.jump,
                time_delta=np.full_like(move_angle, 0.014),
            )
            speed_before = np.linalg.norm(self.player_state.vel[:, :2], axis=1)
            next_player_state = phys.apply(inputs, self.player_state)
            speed_after = np.linalg.norm(next_player_state.vel[:, :2], axis=1)
            
            delta_speeds.append(speed_after - speed_before)
            
        return np.stack(delta_speeds)

    def wish_angle_yaw_plot(self, figsize=(20, 16)):
        import matplotlib.pyplot as plt

        delta_speeds = self.hypothetical_delta_speeds
        wish_angle = self.wish_angle

        plt.figure(figsize=(20, 16))
        plt.ylim(180, -180)
        plt.ylabel('wish_angle - move_angle')
        plt.xlabel('frame')

        # Color by rank, and only show from the `100 * alpha` percentile up.
        c = np.argsort(np.argsort(delta_speeds, axis=0), axis=0)
        c = c / (delta_speeds.shape[0] - 1)
        alpha = 0.95
        c = np.maximum((c - alpha) / (1 - alpha), 0)

        # Sometimes some zero values are in the top 5th percentile, just show
        # these as zero.
        c = np.where(np.abs(delta_speeds) < 1e-3, 0, c)

        plt.imshow(c, cmap='viridis',
                   extent=(0, delta_speeds.shape[1], 180, -180))
        wrapped_angle = ((wish_angle - self.move_angle + 180) % 360) - 180
        plt.plot(wrapped_angle, color='#ff00ff', linestyle='--')

        plt.colorbar(orientation='horizontal')

        plt.show()


def _draw_arrow(im, pos: np.ndarray, vec: np.ndarray, width: float, head_size: float, color: np.ndarray,
                xform: np.ndarray):

    len_ = np.linalg.norm(vec)
    if len_ < 1e-5:
        return
    vec = vec / len_

    xform = xform @ np.array([[vec[1],  vec[0], pos[0]],
                              [-vec[0], vec[1], pos[1]],
                              [0,       0,      1]])

    points = np.array([[0.5 * width,       0,                      1],
                       [0.5 * width,       len_ * (1 - head_size), 1],
                       [len_ * head_size,  len_ * (1 - head_size), 1],
                       [0,                 len_,                   1],
                       [-len_ * head_size, len_ * (1 - head_size), 1],
                       [-0.5 * width,      len_ * (1 - head_size), 1],
                       [-0.5 * width,      0, 1]])

    points = (points @ xform.T)[:, :2]
    points = np.array([[int(x) for x in p] for p in points])

    rgb = np.ascontiguousarray(im[:, :, :3]).copy()
    a = np.ascontiguousarray(im[:, :, 3]).copy()

    cv2.fillPoly(rgb, points[None], color[:3], lineType=8)
    cv2.polylines(rgb, points[None], True, color[:3], thickness=2, lineType=8)
    cv2.fillPoly(a, points[None], color[3], lineType=cv2.LINE_AA)

    im[:, :, :3] = rgb
    im[:, :, 3] = a


def _draw_arrow_key(im, pos: np.ndarray, vec: np.ndarray, pressed: bool, xform: np.array):
    color = [0, 255, 255, 255] if pressed else [200, 200, 200, 255]
    _draw_arrow(im, pos, vec, 8.0, 0.4, color, xform)


def draw_inputs(im, keys, yaw, xform):
    _draw_arrow_key(im, np.array([40, 20]), np.array([0, -20]), keys[env.Key.FORWARD], xform)
    _draw_arrow_key(im, np.array([20, 40]), np.array([-20, 0]), keys[env.Key.STRAFE_LEFT], xform)
    _draw_arrow_key(im, np.array([40, 30]), np.array([0, 20]), False, xform)
    _draw_arrow_key(im, np.array([60, 40]), np.array([20, 0]), keys[env.Key.STRAFE_RIGHT], xform)


def eval_sim(trainer, env_config: env.Config):
    e = env.VectorPhysEnv(dataclasses.asdict(env_config))
    o, = e.vector_reset()
    action_decoder = env.ActionDecoder(env_config)
    action_decoder.vector_reset(e._yaw)

    obs = []
    reward = []
    actions = []
    done = False
    player_states = []
    yaws = []
    smoves = []
    fmoves = []
    jumps = []

    while not done:
        a = trainer.compute_action(o)
        (yaw,), (smove,), (fmove,), (jump,) = action_decoder.map(
                [a], o[None, env.Obs.Z_VEL], e._time_remaining)

        player_states.append(e.player_state)
        obs.append(o)
        actions.append(a)
        yaws.append(yaw)
        smoves.append(smove)
        fmoves.append(fmove)
        jumps.append(jump)

        (o,), (r,), (done,), _ = e.vector_step([a])

        reward.append(r)
        
    return EvalSimResult(
        time_delta=env_config.time_delta,
        player_state=phys.PlayerState.concatenate(player_states),
        action=np.stack(actions),
        obs=np.stack(obs),
        reward=np.stack(reward),
        yaw=np.stack(yaws),
        smove=np.stack(smoves),
        fmove=np.stack(fmoves),
        jump=np.stack(jumps),
    )


def plot_all_checkpoints():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')

    config = env.Config(
        num_envs=1,
        auto_jump=False,
        time_limit=10,
        key_press_delay=0.3,
        initial_yaw_range=(0, 360),
        max_initial_speed=700,
        zero_start_prob=1.0,
        action_range=1.0
    )

    checkpoint_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    ray.init()

    paths = sorted(checkpoint_dir.glob("checkpoint_*"), key=lambda x: int(x.name.split('_')[1]))
    paths = [x / x.name.replace('_', '-') for x in paths]
    for i, path in enumerate(paths):
        trainer = train.make_trainer(train.make_run_config(config))
        trainer.restore(str(path))

        r = eval_sim(trainer, config)
        r.wish_angle_yaw_plot()

        output_path = output_dir / f"{i:04d}.png"
        plt.savefig(output_path)

        print(f'Wrote {output_path}')
    
