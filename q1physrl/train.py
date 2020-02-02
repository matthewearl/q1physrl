import dataclasses
from pprint import pprint

import numpy as np
import ray
import ray.rllib

import q1physrl.env


def make_trainer():
    return ray.rllib.agents.ppo.PPOTrainer(
        # trainer = ray.rllib.agents.a3c.A2CTrainer(
        env=q1physrl.env.PhysEnv,
        config={"env_config": {"num_envs": 100}, "gamma": 0.99}
    )


_STATS_TO_TRACK = [
    'episode_reward_mean',
    'episode_reward_max'
]


@dataclasses.dataclass
class _Stat:
    val: float
    fname: str


def train():
    ray.init()

    trainer = make_trainer()

    best_stats = {}

    i = 0
    while True:
        stats = trainer.train()
        print('Iteration:', i, 'Current:', {k: stats[k] for k in _STATS_TO_TRACK})

        # Work out which (if any) stats just exceeded the previous best value.
        stats_to_save = []
        for k in _STATS_TO_TRACK:
            if not np.isnan(stats[k]) and (k not in best_stats or stats[k] > best_stats[k].val):
                stats_to_save.append(k)

        # Make a checkpoint whenever one of the stats exceeds its previous best,
        # or 100 iterations elapse.
        if i % 100 == 0 or stats_to_save:
            fname = trainer.save()
            for k in stats_to_save:
                if not np.isnan(stats[k]):
                    best_stats[k] = _Stat(stats[k], fname)
            print('Best:')
            pprint(best_stats)
        i += 1

