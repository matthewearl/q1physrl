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


def eval_sim(trainer):
    e = env.PhysEnv({'num_envs': 1})
    o, = e.vector_reset()
    action_to_move = env.ActionToMove(1)
    action_to_move.vector_reset()

    obs = []
    reward = []
    actions = []
    done = False
    player_states = []
    yaws = []
    smoves = []
    fmoves = []

    while not done:
        a = trainer.compute_action(o)
        (yaw,), (smove,), (fmove,) = action_to_move.map([a], e._time)
        player_states.append(e.player_state)
        (o,), (r,), (done,), _ = e.vector_step([a])
        obs.append(o)
        actions.append(a)
        reward.append(r)
        yaws.append(yaw)
        smoves.append(smove)
        fmoves.append(fmove)
        
    return EvalSimResult(
        player_state=phys.PlayerState.concatenate(player_states),
        action=np.stack(actions),
        obs=np.stack(obs),
        reward=np.stack(reward),
        yaw=np.stack(yaws),
        smove=np.stack(smoves),
        fmove=np.stack(fmoves),
    )

