import dataclasses

import numpy as np
import pyquake

from . import env
from . import phys



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

