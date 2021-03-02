# DQN

An implementation of Double DQN with Prioritised Experience Replay.

Originally authored to develop TensorFlow familiarity. Made public to provide those investigating TensorFlow/Deep RL with an additional example implementation.

<table><tr><td><img src='https://github.com/djdnx/DQN/blob/master/cartpole.gif' alt='cartpole_demo_gif'></td><td><img src='https://github.com/djdnx/DQN/blob/master/mountain_car.gif' alt='mountain_car_demo_gif'></td></tr></table>

## Getting Started

Clone this repository and `cd` into it with

`git clone https://github.com/djdnx/DQN.git && cd DQN`.

The dependencies for this project are managed by `pipenv`. In order to maintain compatibility, the `Pipfile` specifies python 3.6 and exact package versions.

A robust approach to getting up and running is to work from a fresh python 3.6 `conda` environment. To do so, install `conda` via a miniconda/anaconda installation and run

```
$ conda create -n py36 python=3.6
$ conda activate py36
```

Next, install `pipenv` into this environment with

``
$ pip install pipenv==2018.11.26
``

Now create and activate a `pipenv` environment with the desired packages by running

```
$ pipenv install
$ pipenv shell
```

Example trained agents can then be observed by running

`$ python cartpole_runner.py`

and

`$ python mountain_car_runner.py`.

The tensorboard output generated during each agent's training can be viewed at [localhost:6006](http://localhost:6006) after running

`$ tensorboard --logdir=cartpole_tensorboard_dir --port=6006`

or

`$ tensorboard --logdir=mountain_car_tensorboard_dir --port=6006`

as desired.

To train an agent on OpenAI's `'Cartpole-v0'` or `'MountainCar-v0'` environments,
optionally modify the files [cartpole_trainer.py](cartpole_trainer.py) or [mountain_car_trainer.py](mountain_car_trainer.py)
respectively and run

`$ python cartpole_trainer.py`

or

`$ python mountain_car_trainer.py`.

## Next Steps

```bash
.
├── DQN # Main package holding the DQN implementation
│   ├── __init__.py
│   ├── dqn_importer.py # For importing trained models
│   └── dqn_trainer.py # For training models
├── cartpole_trainer.py # Train an agent on 'CartPole-v0'
├── cartpole_runner.py # Run a trained agent within 'CartPole-v0'
├── cartpole_tf_model # Tensorflow files to import trained agent into cartpole_runner.py
│   ├── q_model-30000.data-00000-of-00001
│   ├── q_model-30000.index
│   └── q_model-30000.meta
├── cartpole_tensorboard_dir # Tensorboard dir created when cartpole_trainer.py is run
│   └── ...
├── mountain_car_trainer.py # Train an agent on 'MountainCar-v0'
├── mountain_car_runner.py # Run a trained agent within 'MountainCar-v0'
├── mountain_car_tf_model # Tensorflow files to import trained agent into mountain_car_runner.py
│   ├── q_model-280000.data-00000-of-00001
│   ├── q_model-280000.index
│   └── q_model-280000.meta
├── mountain_car_tensorboard_dir # Tensorboard dir created when mountain_car_trainer.py is run
│   └── ...
```

The agents provided have been trained on the `'Cartpole-v0'` or `'MountainCar-v0'` environments,
however the `DQN` package can be used to solve further environments through the use of different
parameters and architectures.

[cartpole_trainer.py](cartpole_trainer.py) and [mountain_car_trainer.py](mountain_car_trainer.py) together provide a suggested structure for files used to train an agent, whilst [cartpole_runner.py](cartpole_runner.py) and [mountain_car_runner.py](mountain_car_runner.py) demonstrate how to observe a trained agent's behaviour within its environment. Detailed docstrings throughout the `DQN` package provide guidance on interaction with its API.

## Sources
- _Human-level control through deep reinforcement learning_, Mnih et al., 2015, http://dx.doi.org/10.1038/nature14236
- _Deep Reinforcement Learning with Double Q-learning_, van Hasselt et al., 2015, https://arxiv.org/abs/1509.06461
- _Prioritized Experience Replay_, Schaul et al., 2015, https://arxiv.org/abs/1511.05952
- _Let's Make a DQN_ series at https://jaromiru.com/
