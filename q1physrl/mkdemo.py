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


import asyncio
import json
import logging
import pickle
import signal
import sys

import numpy as np
import ray
from q1physrl_env import env

from . import analyse, train


logger = logging.getLogger(__name__)


def _make_observation(client, time_remaining, config):
    yaw = 180 * client.angles[1] / np.pi
    vel = np.array(client.velocity)
    z_pos = client.player_origin[2]
    obs_scale = env.get_obs_scale(config)
    return np.concatenate([[time_remaining], [yaw], [z_pos], vel]) / obs_scale


def _apply_action(client, action_decoder, action, time_remaining):
    (yaw,), (smove,), (fmove,), (jump,) = action_decoder.map([[a[0] for a in action]],
                                                             np.float32(client.velocity[2])[None],
                                                             np.float32(time_remaining)[None])
    yaw *= np.pi / 180

    buttons = np.where(jump, 2, 0)
    client.move(pitch=0, yaw=yaw, roll=0, forward=fmove, side=smove,
                up=0, buttons=buttons, impulse=0)


async def _eval_coro(config, port, trainer, demo_file):
    import pyquake.client

    client = await pyquake.client.AsyncClient.connect("localhost", port)
    config = env.Config(**{**config, 'num_envs': 1})
    action_decoder = env.ActionDecoder(config)
    action_decoder.vector_reset(np.array([env.INITIAL_YAW_ZERO]))

    obs_list = []
    action_list = []

    try:
        demo = client.record_demo()
        await client.wait_until_spawn()
        client.move(*client.angles, 0, 0, 0, 0, 0)
        await client.wait_for_movement(client.view_entity)
        start_time = client.time
        time_remaining = None
        while time_remaining is None or time_remaining >= 0:
            time_remaining = config.time_limit - (client.time - start_time)
            obs = _make_observation(client, time_remaining, config)
            obs_list.append(obs)
            action = trainer.compute_action(obs)
            action_list.append(action)

            _apply_action(client, action_decoder, action, time_remaining)
            await client.wait_for_movement(client.view_entity)

        demo.stop_recording()
        demo.dump(demo_file)

    finally:
        await client.disconnect()

    return obs_list, action_list


async def make_demo(checkpoint_fname, params_fname, quakespasm_binary_fname, game_dir, demo_file_fname,
                    obs_action_fname=None):
    """Start a quakespasm server, run an agent on it, and record the results in a demo.
    
    Arguments:
        checkpoint_fname:  The checkpoint file containing the trainer weights.  Typically located in "~/ray_results/<exp
            name>/checkpoint_<n>/checkpoint-<n>".  Must be colocated with `.tune_metadata` file.
        params_fname: The params file containing the parameters used for training.  Typically located in
            "~/ray_results/<exp name>/params.json".
        quakespasm_binary_fname:  Path to the quakespasm binary.
        game_dir:  Directory containing `id1/pak0.pak`.
        demo_file_fname:  Destination demo file name.
        obs_action_fname:  Optional filename to write pickled observation and action values to.

    """
    with open(params_fname, 'r') as f:
        params = json.load(f)

    logger.info("Initializing ray")
    ray.init(memory=100 << 20, object_store_memory=100 << 20)

    logger.info("Making trainer")
    trainer = train.make_trainer({'trainer_class': 'PPOTrainer', 'trainer_config': params})
    trainer.restore(checkpoint_fname)

    logger.info("Spawning quakespasm server")
    proc = await asyncio.create_subprocess_exec(quakespasm_binary_fname,
                                                '-protocol', '15',
                                                '-dedicated', '1',
                                                '-basedir', game_dir,
                                                '+host_framerate', str(1. / 72),
                                                '+sys_ticrate', '0.0',
                                                '+sync_movements', '1',
                                                '+nomonsters', '1',
                                                '+map', '100m')
    logger.info("Created quakespasm process. Pid: %s", proc.pid)

    try:
        logger.info("Interacting with server")
        with open(demo_file_fname, 'wb') as f:
            obs, action = await _eval_coro(params['env_config'], 26000, trainer, f)
        if obs_action_fname is not None:
            with open(obs_action_fname, 'wb') as f:
                pickle.dump((obs, action), f)
    finally:
        proc.send_signal(signal.SIGINT)

    logger.info("Waiting for quakespasm to exit")
    await proc.wait()

    logger.info("Loading demo to find finish time")
    times, origins, yaws, finish_time = analyse.parse_demo(demo_file_fname)
    # Games recorded using the usual method start at ~1.356 seconds, so account for that here.
    corrected_finish_time = finish_time + (1.3 + 4. / 72) - times[0]
    logger.info("Corrected finish time: %s s", corrected_finish_time)


def make_demo_entrypoint():
    logging.basicConfig(level=logging.INFO)

    checkpoint_fname, params_fname, quakespasm_binary_fname, game_dir, deme_file_fname = sys.argv[1:6]
    if len(sys.argv) > 6:
        obs_action_fname, = sys.argv[6:]
    else:
        obs_action_fname = None

    asyncio.run(make_demo(checkpoint_fname, params_fname, quakespasm_binary_fname, game_dir, deme_file_fname,
                          obs_action_fname))
