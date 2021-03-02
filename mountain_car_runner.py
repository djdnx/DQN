"""File demonstrating how to import a trained model for use with the 'MountainCar-v0' gym environment."""

import gym

from DQN import dqn_importer

env = gym.make("MountainCar-v0")
agent = dqn_importer.DQNAgent(env, "mountain_car_tf_model/q_model-280000.meta",
                              "mountain_car_tf_model/q_model-280000")

total_reward = 0
num_episodes = 100
for _ in range(num_episodes):
    s = env.reset()
    done = False
    episode_reward = 0
    while True:
        env.render()
        s, r, done, info = agent.act(s)
        episode_reward += r
        if done:
            break
    print("Episode reward: {}".format(episode_reward))
    total_reward += episode_reward

average_reward = total_reward / num_episodes
print("Average reward per episode: {}".format(average_reward))
env.close()
