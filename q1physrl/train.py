import copy
import dataclasses
import sys
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


_ENV_CONFIG = q1physrl.env.Config(
    num_envs=100,
    auto_jump=False,
    time_limit=5,
    key_press_delay=0.3,
    initial_yaw_range=(0, 360),
    max_initial_speed=700.,
    zero_start_prob=1e-2,
    action_range=0.1,
)


_TRAINER_CLASSES = {
    "PPOTrainer": ray.rllib.agents.ppo.PPOTrainer
}


def make_run_config(env_config):
    return {
        "trainer_class": "PPOTrainer",
        "trainer_config": {
            "env_config": dataclasses.asdict(env_config),
            "gamma": 0.99,
            "lr": 5e-6,
            "entropy_coeff": 1e-1, 
            "num_workers": 4,
            "train_batch_size": 10_000,
            "kl_target": 3.6e-2,
            "lambda": 0.95,
            "vf_clip_param": 100,
        }
    }


def _on_episode_end(info):
    episode = info["episode"]
    if episode.last_info_for()['zero_start']:
        episode.custom_metrics['zero_start_total_reward'] = episode.total_reward


def make_trainer(run_config):
    cls = _TRAINER_CLASSES[run_config['trainer_class']]
    trainer_config = copy.deepcopy(run_config['trainer_config'])
    trainer_config['callbacks'] = {'on_episode_end': _on_episode_end}

    return cls(env=q1physrl.env.PhysEnv, config=trainer_config)


_STATS_TO_TRACK = [
    'episode_reward_mean',
    'episode_reward_max',
    'custom_metrics/zero_start_total_reward_mean',
]


_STATS_TO_PRINT = _STATS_TO_TRACK + ['info/learner/default_policy/entropy', 'episode_len_mean']


def _get_stat(stats, k):
    parts = k.split('/')
    x = stats
    for part in parts:
        try:
            x = x[part]
        except KeyError:
            return np.nan
    return x


@dataclasses.dataclass
class _Stat:
    val: float
    fname: str


def train():
    run_config = make_run_config(_ENV_CONFIG)

    if wandb is not None:
        wandb.init(project="q1physrl",
                   config=run_config,
                   sync_tensorboard=True)

    ray.init()

    trainer = make_trainer(run_config)

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

