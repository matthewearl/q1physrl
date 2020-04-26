# Quake 1 Movement Physics Reinforcement Learning

![Gif of RL agent running 100m](/data/images/wr.gif)

This is the source code for my Quake reinforcement learning project, which I described in my video
["Teaching a computer to strafe jump in Quake with reinforcement learning"](https://www.youtube.com/watch?v=hx7kvTZLHYI).

Find below instructions to use the gym environment, train your own model, and produce a demo file.


## Installing the gym env

If you just want to play with the gym environment then you need only clone this repo and install the env package:

```
git clone 'https://github.com/matthewearl/q1physrl.git'
pip install -e q1physrl/q1physrl_env
```

The environment ID is `Q1PhysEnv-v0` which is registered by importing `q1physrl_env.env`.  The environment accepts a
single argument `config` which is an instance of `q1phys_env.env.Config`.  See the docstring for this class for more
details.


## Training

Follow these instructions if you want to train the model used in the video:

1. Set up a clean virtualenv using Python 3.7 or greater (tested with Python 3.7.2).

Install this repo and its requirements:
```
git clone 'https://github.com/matthewearl/q1physrl.git'
cd q1physrl
pip install -r requirements_q1physrl.txt
pip install -e q1physrl_env
pip install -e .
```

2. (Optional.) To log to Weights and Biases (including logging angle plots with matplotlib) run:

```
pip install matplotlib
pip install wandb
wandb init
```
Alternatively, you can just view results on tensorboard with `tensorboard -logdir ~/ray_results`.

3. Finally, run:

```
q1physrl_train data/params.yml
```

...and wait for convergence.  Modify the file referenced in the argument to change hyper-parameters and environment
settings.  My current PB (the one in the video) gets about 5700 on the `zero_start_total_reward_mean` metric, after 150
million steps, which takes about a day on my i7-6700K:

![screenshot of training curve](/data/images/train.png)

I'm using RLLib's implementation of PPO --- see the [RLLib PPO
docs](https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo) to see how you can tweak the
algorithm parameters.  See [q1physrl_env.env.Config](/q1physrl_env/q1physrl_env/env.py#L81) for environment parameters.

RLLib will write checkpoint files and run parameters into `~/ray_results`, which will be needed to produce a demo file
(see *Producing a Quake demo file* below).


## Producing a Quake demo file

Demo files (extension `.dem`) are how games are recorded and shared in Quake.

To produce and play back demos you'll need a registered version of Quake.  The reason for this is that the shareware
version's license does not permit playing user developed levels with the game, of which the 100m level is one (see
`licinfo.txt` in the shareware distribution).  If you don't already have it, you can [buy it on
Steam](https://store.steampowered.com/app/2310/QUAKE/) for not much money at all.

Once you have a registered copy of the game, the easiest way to produce a demo is with the following script which makes
use of a docker image containing the rest of the required software.  It is invoked like this:

```
docker/docker-make-demo.sh \
    ~/.steam/steam/steamapps/common/Quake/Id1/ \
    data/checkpoints/wr/checkpoint data/checkpoints/wr/params.json \
    /tmp/wr.dem
```

The first argument is the `id1` directory from your Quake installation directory, which must contain `pak0.pak` and
`pak1.pak`.  The second and third arguments are the checkpoint file and run parameters.  The values shown above
reproduce the demo shown in the video, however you should substitute files found in `~/ray_results/<experiment_name>` if
you want to produce a demo using an agent that you have trained.  The final argument is the demo file that is to be
written.

The script works by first launching a modified quakespasm server, to which a custom client connects. The client
repeatedly reads state from the server, pushes it through the model, and sends movement commands into the game.  The
network traffic is recorded to produce the demo file. The quakespasm server is modified to pause at the end of each
frame until a new movement command is received, to avoid issues of synchronization.

## Playing the demo file

The demo file can be played by following these steps:
- Install a registered version of Quake. (See section above for why the registered version is necessary.)
- Download the 100m map [from here](http://quake.speeddemosarchive.com/quake/maps/100m.zip) and extract it into your
  `id1/maps` directory.
- Move the `.dem` file that you produced into the `id1` directory.
- Start Quake, bring down the console, and then enter the command `playdemo <demo name>`


## Technical notes

### Demo time offsets

Demos produced by the method described above differ from manually recorded demos in the sense that time is shifted
approximately 0.15s ahead.  For example, here I've plotted the player height at the start of the RL agent's demo
compared to the human WR demo:

![Uncorrected initial drop plot](/data/images/initial_drop_uncorrected.png)

At the start of the 100m map the player actually spawns slightly above the floor so there is a small drop. The player
has no control over the player's height (z position) during this fall and so all being fair the two trajectories should
coincide.  By adding the difference between the first time seen in each demo file to the times in the RL agent's demo we
can make this happen:

![Corrected initial drop plot](/data/images/initial_drop_corrected.png)

This is the correction applied in the comparison plot in the video.
