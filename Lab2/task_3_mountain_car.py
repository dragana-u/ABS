import gymnasium as gym
import numpy as np
from q_learning import random_q_table

def get_discrete_state(state, low_value, window_size):
    return tuple(((state - low_value) / window_size).astype(np.int32))

def epsilon_greedy_action(q_table, state, epsilon, action_space):
    return np.random.randint(0, action_space) if np.random.random() < epsilon else np.argmax(q_table[state])

def run_experiment(num_episodes, learning_rate=0.1, discount=0.95, epsilon=0.2, epsilon_decay=False, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    num_actions = env.action_space.n
    obs_space_size = np.array([20, 20])
    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    obs_window_size = (obs_space_high - obs_space_low) / obs_space_size
    q_table = random_q_table(-2, 0, (*obs_space_size, num_actions))
    episode_rewards, episode_steps = [], []

    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = get_discrete_state(state, obs_space_low, obs_window_size)
        episode_reward, done, truncated, steps = 0, False, False, 0
        current_epsilon = max(0.01, epsilon * (1 - episode / num_episodes)) if epsilon_decay else epsilon

        while not (done or truncated):
            action = epsilon_greedy_action(q_table, discrete_state, current_epsilon, num_actions)
            new_state, reward, done, truncated, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state, obs_space_low, obs_window_size)
            if new_state[1] > 0:
                reward += 0.1 * new_state[1]
            episode_reward += reward
            steps += 1
            if not (done or truncated):
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                q_table[discrete_state + (action,)] = current_q + learning_rate * (reward + discount * max_future_q - current_q)
            elif done:
                q_table[discrete_state + (action,)] = 0
            discrete_state = new_discrete_state

        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        if episode % 10 == 0:
            print(f"Епизода: {episode}, Награда: {episode_reward}, Чекори: {steps}, Епсилон: {current_epsilon:.4f}")

    env.close()
    return q_table, episode_rewards, episode_steps

def test_policy(q_table, num_test_episodes=100, render_final=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render_final else None)
    obs_space_size = np.array([20, 20])
    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    obs_window_size = (obs_space_high - obs_space_low) / obs_space_size
    test_rewards, test_steps = [], []

    for episode in range(num_test_episodes):
        state, _ = env.reset()
        discrete_state = get_discrete_state(state, obs_space_low, obs_window_size)
        episode_reward, done, truncated, steps = 0, False, False, 0

        while not (done or truncated):
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, truncated, _ = env.step(action)
            discrete_state = get_discrete_state(new_state, obs_space_low, obs_window_size)
            episode_reward += reward
            steps += 1

        test_rewards.append(episode_reward)
        test_steps.append(steps)

    env.close()
    return np.mean(test_rewards), np.mean(test_steps)

if __name__ == '__main__':
    q_table_50, _, _ = run_experiment(num_episodes=50, epsilon=0.2, epsilon_decay=False)
    avg_reward_50, avg_steps_50 = test_policy(q_table_50)
    print(f"50 епизоди (без намалување) - Просечна тест награда: {avg_reward_50:.2f}, Просечен број на чекори: {avg_steps_50:.2f}")

    q_table_100, _, _ = run_experiment(num_episodes=100, epsilon=0.2, epsilon_decay=False)
    avg_reward_100, avg_steps_100 = test_policy(q_table_100)
    print(f"100 епизоди (без намалување) - Просечна тест награда: {avg_reward_100:.2f}, Просечен број на чекори: {avg_steps_100:.2f}")

    q_table_50_decay, _, _ = run_experiment(num_episodes=50, epsilon=0.2, epsilon_decay=True)
    avg_reward_50_decay, avg_steps_50_decay = test_policy(q_table_50_decay)
    print(f"50 епизоди (со намалување) - Просечна тест награда: {avg_reward_50_decay:.2f}, Просечен број на чекори: {avg_steps_50_decay:.2f}")

    q_table_100_decay, _, _ = run_experiment(num_episodes=100, epsilon=0.2, epsilon_decay=True)
    avg_reward_100_decay, avg_steps_100_decay = test_policy(q_table_100_decay)
    print(f"100 епизоди (со намалување) - Просечна тест награда: {avg_reward_100_decay:.2f}, Просечен број на чекори: {avg_steps_100_decay:.2f}")

    best_q_table = q_table_100_decay
    test_policy(best_q_table, num_test_episodes=1, render_final=True)