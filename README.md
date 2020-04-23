## Installing the gym env

If you just want to play with the gym environment then you need only clone this repo and install the env package:

```
git clone '<this repo>'
pip install -e q1physrl_env
```

The environment ID is `Q1PhysEnv-v0`.  The environment accepts a single argument `config` which is an instance of
`q1phys_env.env.Config`.  See the docstring for this class for more details.

## Training

Follow these instructions if you want to train the model used in the video:

1. Set up a clean virtualenv using Python 3.7 or greater (tested with Python 3.7.2).

Install this repo and its requirements:
```
git clone '<this repo>'
pip install -r q1physrl/requirements_train.txt
pip install -e q1physrl_env
pip install -e q1physrl
```

2. To log to Weights and Biases (including logging angle plots with matplotlib) run:

```
pip install matplotlib
pip install wandb
wandb init
```
Alternatively, you can just view results on tensorboard with `tensorboard -logdir ~/ray_results`.

3. Finally, run:

```
q1physrl_train q1physrl/params.yml
```

and wait for convergence.  Modify the file referenced in the argument to change run config.  My current PB (the one in
the video) gets about 5700 on the `zero_start_total_reward_mean` metric, after 150 million steps.

RLLib will write checkpoint files and run parameters into `~/ray_results`, which will be needed to produce a demo file
(see *Producing a Quake demo file* below).

## Producing a Quake demo file

Demo files (extension `.dem`) are how games are recorded and shared in Quake.  The easiest way to do this is with the
docker image, since there are a few bits and pieces to install.

[TODO: Add docker invocation here]

## Pretrained weights and parameters

[TODO: Add these into the repo and reference them here]

## Playing the demo file

The demo file can be played by installing Quake on your machine, and moving the `.dem` file into your `id1` directory.
Start quake and then enter the following command at the console:

```playdemo <demo name>```
