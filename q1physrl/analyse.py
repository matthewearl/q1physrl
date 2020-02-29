import dataclasses
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pyquake
import ray

from . import env
from . import phys
from . import train


def parse_demo(fname):
    view_entity = None
    origin = None
    origins = []
    times = []
    yaws = []

    time = None

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

    return np.array(times), np.array(origins), np.array(yaws)


@dataclasses.dataclass
class EvalSimResult:
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
                button2=self.player_state.vel[:, 2] <= 16,
                time_delta=np.full_like(move_angle, 0.014),
            )
            speed_before = np.linalg.norm(self.player_state.vel[:, :2], axis=1)
            next_player_state = phys.apply(inputs, self.player_state)
            speed_after = np.linalg.norm(next_player_state.vel[:, :2], axis=1)
            
            delta_speeds.append(speed_after - speed_before)
            
        return np.stack(delta_speeds)

    def wish_angle_yaw_plot(self, figsize=(20, 16)):
        delta_speeds = self.hypothetical_delta_speeds

        wish_angle = self.yaw + (180. * np.arctan2(self.smove, self.fmove) / np.pi)

        vmin = np.min(delta_speeds)
        vmax = np.max(delta_speeds)
        norm = matplotlib.colors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)

        plt.figure(figsize=(20, 16))
        plt.ylim(180, -180)
        plt.ylabel('wish_angle - move_angle')
        plt.xlabel('frame')

        c = delta_speeds
        plt.imshow(c, cmap='seismic', norm=norm,
                   extent=(0, delta_speeds.shape[1], 180, -180)
                  )
        plt.plot(wish_angle - self.move_angle, color='#00ff00')
        plt.axhline(0)
        plt.axhline(90)
        plt.axhline(-45, alpha=0.5)
        plt.axhline(45, alpha=0.5)
        plt.axhline(-90)

        plt.colorbar()

        plt.plot(self.fmove / 20 + 100, color='#ffff00')

        plt.show()

def eval_sim(trainer, env_config):
    e = env.PhysEnv(dataclasses.asdict(env_config))
    o, = e.vector_reset()
    action_to_move = env.ActionToMove(env_config)
    action_to_move.vector_reset(e._yaw)

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
        (yaw,), (smove,), (fmove,), (jump,) = action_to_move.map(
                [a], o[None, env.Obs.Z_VEL], e._time_remaining)
        player_states.append(e.player_state)
        (o,), (r,), (done,), _ = e.vector_step([a])
        obs.append(o)
        actions.append(a)
        reward.append(r)
        yaws.append(yaw)
        smoves.append(smove)
        fmoves.append(fmove)
        jumps.append(jump)
        
    return EvalSimResult(
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
    matplotlib.use('Agg')

    config = env.Config(
        num_envs=1,
        auto_jump=False,
        time_limit=5,
        key_press_delay=0.3,
        initial_yaw_range=(0, 360),
        max_initial_speed=700,
        zero_start_prob=1.0,
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
    
