import sys
import dataclasses
from pprint import pprint

import numpy as np
import ray
import ray.rllib

import q1physrl.env

try:
    import wandb
    from wandb.tensorflow import WandbHook
except ImportError:
    wandb = None


def _on_episode_end(info):
    episode = info["episode"]
    if episode.last_info_for()['zero_start']:
        episode.custom_metrics['zero_start_total_reward'] = episode.total_reward


def make_trainer():
    return ray.rllib.agents.ppo.PPOTrainer(
        env=q1physrl.env.PhysEnv,
        config={"env_config": {"num_envs": 100}, "gamma": 0.99, "lr": 5e-6, "entropy_coeff": 1e-2, 
                "train_batch_size": 20_000,
                "callbacks": {
                    "on_episode_end": _on_episode_end,
                }
               }
    )


_STATS_TO_TRACK = [
    'episode_reward_mean',
    'episode_reward_max'
]

_STATS_TO_PRINT = _STATS_TO_TRACK + ['info/learner/default_policy/entropy', 'episode_len_mean']


def _get_stat(stats, k):
    parts = k.split('/')
    x = stats
    for part in parts:
        x = x[part]
    return x


@dataclasses.dataclass
class _Stat:
    val: float
    fname: str


def train():
    if wandb is not None:
        wandb.init(project="q1physrl", sync_tensorboard=True)

    ray.init()

    trainer = make_trainer()

    if len(sys.argv) >= 2:
        trainer.restore(sys.argv[1])

    best_stats = {}

    i = 0
    while True:
        stats = trainer.train()
        print('Iteration:', i, 'Current:', {k: _get_stat(stats, k) for k in _STATS_TO_PRINT})

        # Work out which (if any) stats just exceeded the previous best value.
        stats_to_save = []
        for k in _STATS_TO_TRACK:
            if not np.isnan(_get_stat(stats, k)) and (k not in best_stats or _get_stat(stats, k) > best_stats[k].val):
                stats_to_save.append(k)

        # Make a checkpoint whenever one of the stats exceeds its previous best,
        # or 100 iterations elapse.
        if i % 100 == 0 or stats_to_save:
            fname = trainer.save()
            for k in stats_to_save:
                if not np.isnan(_get_stat(stats, k)):
                    best_stats[k] = _Stat(_get_stat(stats, k), fname)
            print('Best:')
            pprint(best_stats)
        i += 1

