import copy
import datetime
import dataclasses
import sys
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import ray
import ray.rllib
import yaml

import q1physrl_env.env
import q1physrl.action_dist     # Import required for the action dist within to be usable

try:
    import tensorboardX    # for some reason this is required with ray==0.8.2 for wandb to work
    import wandb
    from wandb.tensorflow import WandbHook
except ImportError:
    wandb = None


_ENV_CLASS = q1physrl_env.env.VectorPhysEnv


_TRAINER_CLASSES = {
    "PPOTrainer": ray.rllib.agents.ppo.PPOTrainer,
}


def _on_episode_end(info):
    episode = info["episode"]
    if episode.last_info_for().get('zero_start', False):
        episode.custom_metrics['zero_start_total_reward'] = episode.total_reward


def make_trainer(params):
    trainer_config = copy.deepcopy(params['trainer_config'])
    trainer_config['callbacks'] = {'on_episode_end': _on_episode_end}
    cls = _TRAINER_CLASSES[params['trainer_class']]
    return cls(env=_ENV_CLASS, config=trainer_config)


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
    # Parse the config
    with open(sys.argv[1]) as f:
        params = yaml.safe_load(f)

    # Initialize wandb, if it's installed.
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if wandb is not None:
        run = wandb.init(project="q1physrl",
                         config=params,
                         sync_tensorboard=True)
        run_id = f"{run_id}_{run.id}"

    # Initialize ray, and the trainer
    ray.init()
    trainer = make_trainer(params)
    if params['checkpoint_fname'] is not None:
        trainer.restore(params['checkpoint_fname'])

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

        # Periodically record some plots
        if wandb is not None and i % params['plot_frequency'] == 0:
            import matplotlib.pyplot as plt
            import q1physrl.analyse

            start = time.perf_counter()
            env_config = q1physrl_env.env.Config(**params['trainer_config']['env_config'])
            eval_config = dataclasses.replace(env_config, num_envs=1, zero_start_prob=1.0)
            r = q1physrl.analyse.eval_sim(trainer, eval_config)
            r.wish_angle_yaw_plot()
            wandb.log({'chart': plt})
            plt.close()
            print(f'Took {time.perf_counter() - start} seconds to record plot')

        i += 1
