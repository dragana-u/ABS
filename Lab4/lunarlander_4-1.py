import gymnasium as gym
import numpy as np

from Lab4.deep_q_learning import DDPG, OrnsteinUhlenbeckActionNoise


def train_agent(episodes, render=False):
    env = gym.make('LunarLanderContinuous-v3', render_mode="human" if render else None)

    agent = DDPG(state_space_shape=(8,), action_space_shape=(2,),
                 learning_rate_actor=0.001, learning_rate_critic=0.001,
                 discount_factor=0.99, batch_size=64, memory_size=10000)

    noise = OrnsteinUhlenbeckActionNoise(action_space_shape=(2,))
    max_steps = 500

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, epsilon=0, discrete=False) + noise()
            action = np.clip(action, -1.0, 1.0)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update_memory(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                break

        agent.train()

        if (episode + 1) % 5 == 0:
            agent.update_target_model()

        if (episode + 1) % 10 == 0:
            print(f"Епизода {episode+1}/{episodes} - Награда: {episode_reward:.2f}")

    env.close()
    return agent

def evaluate_agent(agent, episodes=50, render=False):
    env = gym.make('LunarLanderContinuous-v3', render_mode="human" if render else None)
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):
            action = agent.get_action(state, epsilon=0, discrete=False)
            action = np.clip(action, -1.0, 1.0)

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if render:
                env.render()

            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"Епизода {episode+1}/{episodes} - Награда {total_reward:.2f}")

    avg_reward = np.mean(rewards)
    print(f"\nПросечна награда за {episodes} епизоди - {avg_reward:.2f}")
    env.close()
    return avg_reward


if __name__ == "__main__":
    print("Тренирање агент во 50 итерации...")
    agent_50 = train_agent(50)
    print("\nТестирање агент во 50 итерации...")
    avg_reward_50 = evaluate_agent(agent_50, episodes=50, render=False)

    print("\nТренирање агент во 100 итерации...")
    agent_100 = train_agent(100)
    print("\nТестирање агент во 100 итерации...")
    avg_reward_100 = evaluate_agent(agent_100, episodes=50, render=True)

    print(f"\nРезултати:\n- Просечна награда во 50 итерации: {avg_reward_50:.2f}")
    print(f"- Просечна награда во 100 итерации: {avg_reward_100:.2f}")
