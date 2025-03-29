import gymnasium as gym
import numpy as np

from ABS.Lab1.mdp import policy_iteration
from mdp import reward_and_visualize

if __name__ == "__main__":
    env = gym.make('Taxi-v3',render_mode="human")

    discount_factors = [0.5, 0.7, 0.9]

    for gamma in discount_factors:
        policy, V = policy_iteration(env, env.action_space.n, env.observation_space.n, discount_factor=gamma)
        best_actions = np.argmax(policy, axis=1)
        print(f"Најдобри акции за discount factor {gamma}: {best_actions}")

        avg_reward_50 = reward_and_visualize(env, policy, num_episodes=50)
        avg_reward_100 = reward_and_visualize(env, policy, num_episodes=100)
        print(f"Просечна награда за discount factor {gamma} за 50 итерации: {avg_reward_50[0]}, просечни чекори: {avg_reward_50[1]}")
        print(f"Просечна награда за discount factor {gamma} за 100 итерации: {avg_reward_100[0]}, просечни чекори: {avg_reward_100[1]}")

    reward_and_visualize(env, policy, num_episodes=1, visualize=True)