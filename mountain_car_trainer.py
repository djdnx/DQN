"""File demonstrating how to train an agent to solve the 'MountainCar-v0' gym environment.

In the environment provided by OpenAI gym, the episode is terminated once the goal is reached
or after 200 steps, whichever occurs first. 200 steps is not enough for a random agent to encounter
the goal frequently, so env's native .step() is modified so that the episode is instead terminated
once the goal is reached or after 25000 steps.

"""

from types import MethodType

import gym

from DQN import dqn_trainer

def step(self, a):
    """Modified version of gym's env.step() which ends an episode after 25000 steps or goal."""
    s_, r, done, info = self.old_step(a)
    self.counter += 1
    done = self.counter >= 25000 or s_[0] >= 0.5
    if done:
        self.counter = 0
    return s_, r, done, info

env = gym.make("MountainCar-v0")
# Bind the modified step() to the env object
env.old_step = env.step
env.counter = 0
env.step = MethodType(step, env)

agent = dqn_trainer.DQNAgent(env, hidden_architecture=(["relu", 256], ["relu", 128]))
agent.learn("mountain_car_tf_model/q_model", "mountain_car_tensorboard_dir", 1400,
            prioritised_experience_replay=True, memory_capacity=50000,
            num_annealing_steps=100000)

env.close()
