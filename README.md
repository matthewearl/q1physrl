## Just install the env

TODO


## Training

Set up a clean virtualenv using Python 3.7 or greater (tested with Python 3.7.2).

Clone the Gaussian Squashed Gaussian branch of ray and install it:
```
git clone '<me/gsg refpoint>' ray-gsg
pip install -v -e ray-gsg/python
```

Install this repo and its requirements:
```
git clone '<this repo>'
pip install -r q1physrl/requirements-train.txt
pip install -e q1physrl
```

To log to weights and biases (including logging to matplotlib) run:

```
pip install matplotlib
pip install wandb
wandb init
```

Finally, run:

```
q1physrl_train
```

and wait for convergence.


## Evaluating with Quake

TODO:  Dockerize all this stuff?

Follow these instructions to take a checkpoint file produced by the above process, run it through a (modified) Quake
server, and save the demo file.  For this step you'll need some Quake pak files, although, shareware is probably fine?

Install pyquake
```
git clone `<pyquake>`
pip install -e pyquake

```

Install modified version of Quakespasm:

```
git clone '<quakespasm me/hacks>' quakespasm-hacks
mkdir -p ~/.quakespasm/id1
cp <quake pak files> ~/.quakespasm/id1
cp 100m.bsp ~/.quakespasm/id1       # Use a search engine to find this map
cd quakespasm-hacks/quakespasm/Quake
make
```

Launch a quake server:
```
./quakespasm -protocol 15 -dedicated 1 -basedir ~/.quakespasm/ +host_framerate 0.01388888888888 +sys_ticrate
0.0 +sync_movements 1 +nomonsters 1 +map 100m
```

In another window, run the following using a checkpoint file produced from training:
```
q1physrl_evaluate  '<checkpoint file>'  '<demo file name>' 
```
TODO: Write this script

Kill the quake server in the first window.


