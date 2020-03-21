import copy
import datetime
import dataclasses
import datetime
import sys
import time
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt

import numpy as np
import ray
import ray.rllib

import q1physrl.analyse
import q1physrl.env

try:
    import tensorboardX    # for some reason this is required with ray==0.8.2 for wandb to work
    import wandb
    from wandb.tensorflow import WandbHook
except ImportError:
    wandb = None
#wandb = None

_ENV_CLASS = q1physrl.env.PhysEnv
_ENV_CONFIG = q1physrl.env.Config(
    num_envs=100,
    auto_jump=False,
    time_limit=10,
    key_press_delay=0.3,
    initial_yaw_range=(0, 360),
    max_initial_speed=700.,
    zero_start_prob=1e-2,
    action_range=10,
    discrete_yaw_steps=-1,
    speed_reward=False,
    fmove_max=800,
    smove_max=1200,
    time_delta=1. / 72,
)


#_ENV_CLASS = q1physrl.env.SimplePhysEnv
#_ENV_CONFIG = q1physrl.env.SimpleConfig(
#    num_envs=100,
#    time_limit=10.,
#    action_range=1.0,
#)


_TRAINER_CLASSES = {
    "PPOTrainer": ray.rllib.agents.ppo.PPOTrainer,
    "SACTrainer": ray.rllib.agents.sac.SACTrainer
}


_SAC_CONFIG = {
    "gamma": 0.99,
    "num_workers": 2,
    "target_entropy": 2,
}


_PPO_CONFIG = {
    "lr": 0.000005,
    "gamma": 0.99,
    "lambda": 0.95,
    "kl_target": 0.0036,
    "num_workers": 4,
    "entropy_coeff": 0.01,
    "vf_clip_param": 100,
    "train_batch_size": 50000
}


def make_run_config(env_config):
    return {
        "trainer_class": "PPOTrainer",
        "trainer_config": {
            "env_config": dataclasses.asdict(env_config),
            **_PPO_CONFIG,
        }
    }


def _on_episode_end(info):
    episode = info["episode"]
    if episode.last_info_for().get('zero_start', False):
        episode.custom_metrics['zero_start_total_reward'] = episode.total_reward


def make_trainer(run_config):
    cls = _TRAINER_CLASSES[run_config['trainer_class']]
    trainer_config = copy.deepcopy(run_config['trainer_config'])
    trainer_config['callbacks'] = {'on_episode_end': _on_episode_end}

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
    run_config = make_run_config(_ENV_CONFIG)

    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if wandb is not None:
        run = wandb.init(project="q1physrl",
                         config=run_config,
                         sync_tensorboard=True)
        run_id = f"{run_id}_{run.id}"

    ray.init()

    trainer = make_trainer(run_config)

    if len(sys.argv) >= 2:
        trainer.restore(sys.argv[1])

    best_stats = {}

    i = 0
    while True:
        stats = trainer.train()
        print('Iteration:', i, 'Current:', {k: _get_stat(stats, k) for k in _STATS_TO_PRINT})
        #pprint(stats)

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
        if wandb is not None and i % 1 == 0:
            start = time.perf_counter()
            eval_config = dataclasses.replace(_ENV_CONFIG, num_envs=1, zero_start_prob=1.0)
            r = q1physrl.analyse.eval_sim(trainer, eval_config)
            r.wish_angle_yaw_plot()
            wandb.log({'chart': plt})

            d = Path(f"plots/{run_id}")
            d.mkdir(parents=True, exist_ok=True)
            output_path = d / f"{i:04d}.png"
            plt.savefig(str(output_path))

            plt.close()
            print(f'Took {time.perf_counter() - start} seconds to record plot')
            
        i += 1

