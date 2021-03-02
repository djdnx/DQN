"""File demonstrating how to train an agent to solve the 'CartPole-v0' gym environment."""

import gym

from DQN import dqn_trainer

env = gym.make("CartPole-v0")

agent = dqn_trainer.DQNAgent(env, hidden_architecture=(["relu", 64],))
agent.learn("cartpole_tf_model/q_model", "cartpole_tensorboard_dir", 400,
            prioritised_experience_replay=True, num_annealing_steps=10000)

env.close()
