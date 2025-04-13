import gymnasium as gym
import numpy as np
from q_learning import random_q_table, calculate_new_q_value, get_random_action, get_action

if __name__ == '__main__':
    env_names = ['FrozenLake-v1', 'Taxi-v3']
    discount_factors = [0.5, 0.9]
    learning_rates = [0.1, 0.01]
    num_episodes = 1000
    num_steps_per_episode = 100
    epsilon = 0.1

    for env_name in env_names:
        print(f"Тестирање на околината: {env_name}")
        env = gym.make(env_name, render_mode="human")

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        for discount_factor in discount_factors:
            for learning_rate in learning_rates:
                for use_greedy in [False, True]:
                    policy_type = "ε-greedy" if use_greedy else "random"
                    print(f"Тренирање со discount_factor={discount_factor}, learning_rate={learning_rate}, policy={policy_type}")

                    q_table = random_q_table(-1, 0, (num_states, num_actions))

                    for episode in range(num_episodes):
                        state, _ = env.reset()
                        for step in range(num_steps_per_episode):
                            if use_greedy:
                                action = get_action(env, q_table, state, epsilon)
                            else:
                                action = get_random_action(env)
                            new_state, reward, terminated, truncated, _ = env.step(action)
                            new_q = calculate_new_q_value(q_table, state, new_state, action, reward,
                                                          lr=learning_rate, discount_factor=discount_factor)
                            q_table[state, action] = new_q
                            state = new_state
                            if terminated or truncated:
                                break

                    print("Final Q-table:")
                    print(q_table)

                    for num_test_episodes in [50, 100]:
                        total_rewards = 0
                        total_steps = 0
                        successful_episodes = 0
                        steps_to_goal = []

                        for episode in range(num_test_episodes):
                            state, _ = env.reset()
                            for step in range(num_steps_per_episode):
                                if use_greedy:
                                    action = get_action(env, q_table, state, epsilon)
                                else:
                                    action = get_random_action(env)
                                new_state, reward, terminated, truncated, _ = env.step(action)
                                total_rewards += reward
                                total_steps += 1
                                state = new_state
                                if terminated:
                                    if reward > 0:
                                        successful_episodes += 1
                                        steps_to_goal.append(step + 1)
                                    break
                                if truncated:
                                    break

                        avg_reward = total_rewards / num_test_episodes
                        avg_steps = total_steps / num_test_episodes
                        avg_steps_to_goal = np.mean(steps_to_goal) if steps_to_goal else 0

                        print(f"Policy: {policy_type}, Просечна награда за {num_test_episodes} епизоди: {avg_reward}, просечни чекори: {avg_steps}")
                        print(f"Policy: {policy_type}, Просечни чекори за да стигне до целта: {avg_steps_to_goal} (successful episodes: {successful_episodes})")

        env.close()