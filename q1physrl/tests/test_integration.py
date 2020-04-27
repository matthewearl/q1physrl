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


"""Compare observations returned from the environment, with observations returned from the real game."""


import asyncio
import dataclasses
import io
from typing import Set

import numpy as np
from q1physrl_env import env

from q1physrl import analyse


@dataclasses.dataclass
class Action:
    keys: Set[env.Key]
    mouse_x: float

    def encode(self, *, auto_jump: bool):
        num_keys = len(env.Key) - 1 if auto_jump else len(env.Key)
        out = np.zeros((num_keys + 1,), np.float32)

        for k in self.keys:
            if not auto_jump or k != env.Key.JUMP:
                out[k] = 1

        out[-1] = self.mouse_x
        return [[np.float(x)] for x in out]


@dataclasses.dataclass
class DummyTrainer:
    auto_jump: bool
    _index: int = 0

    def compute_action(self, obs):
        if self._index < 100:
            action = Action({env.Key.FORWARD}, 0) 
        else:
            action = Action({env.Key.STRAFE_LEFT}, -2)

        self._index += 1
        return action.encode(auto_jump=self.auto_jump)

    def reset(self):
        self._index = 0


async def validate_trainer(time_limit, trainer):
    demo_file = io.BytesIO()
    obs_list, action_list = await env.eval_coro(time_limit, True, 26000, trainer, demo_file)

    return obs_list


async def test_actions():
    """Make a dummy trainer, and compare observations made from running it through the env, and the real game."""
    env_config = env.Config(
        num_envs=1,
        auto_jump=True,
        time_limit=5,
        key_press_delay=0.3,
        initial_yaw_range=(90, 90),
        max_initial_speed=0.,
        zero_start_prob=1.,
    )

    trainer = DummyTrainer(auto_jump=True)
    real_obs = np.stack(await validate_trainer(env_config.time_limit, trainer))

    trainer.reset()
    sim_obs = analyse.eval_sim(trainer, env_config).obs

    real_obs = real_obs[1:]

    assert abs(real_obs.shape[0] - sim_obs.shape[0]) <= 1
    num_frames = min(real_obs.shape[0], sim_obs.shape[0])

    return np.max(real_obs[:num_frames] - sim_obs[:num_frames])


if __name__ == "__main__":
    print(asyncio.run(test_actions()))
