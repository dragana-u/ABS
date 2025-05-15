import gymnasium as gym
from deep_q_learning import DQNAgent, DDQNAgent

def train_agent(agent_class, env_name='MountainCar-v0'):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = agent_class(state_dim, action_dim)

    epsilon = 1.0
    eps_min = 0.01
    eps_decay = 0.995
    episodes = 200
    max_steps = 1000
    success_threshold = 0.5

    state, _ = env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        next_state, reward, _, truncated, _ = env.step(action)
        done = truncated

        position = next_state[0]
        reward = 100 * (position + 0.5) ** 2
        if position >= success_threshold:
            reward += 1000
        elif done:
            reward -= 50

        agent.update_memory(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()[0]

    best_position = -float('inf')

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        episode_max_pos = -float('inf')
        done = False

        for step in range(max_steps):
            action = agent.get_action(state, epsilon)
            next_state, _, done, truncated, _ = env.step(action)
            done = done or truncated


            position = next_state[0]
            reward = 100 * (position + 0.5) ** 2
            if position >= success_threshold:
                reward += 1000
                done = True
            elif done:
                reward -= 50

            agent.update_memory(state, action, reward, next_state, done)
            agent.train()

            total_reward += reward
            state = next_state
            episode_max_pos = max(episode_max_pos, position)

            if done:
                break

        epsilon = max(eps_min, epsilon * eps_decay)

        if episode_max_pos > best_position:
            best_position = episode_max_pos

        print(f"{agent_class.__name__:5} | Епизода {episode:3d} | "
              f"Макс. позиција: {episode_max_pos:5.2f} | "
              f"Вкупна награда: {total_reward:7.1f} | "
              f"Епсилон: {epsilon:4.2f}")

        if episode_max_pos >= success_threshold:
            print(f"Успех постигнат во епизода {episode}! Позиција: {episode_max_pos:.2f}")
            break

    env.close()
    return agent


def test_agent(agent, episodes=10):
    env = gym.make('MountainCar-v0', render_mode='human')
    success_count = 0

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        max_pos = -float('inf')

        while not done:
            action = agent.get_action(state, 0.0)
            state, _, done, truncated, _ = env.step(action)
            done = done or truncated
            max_pos = max(max_pos, state[0])

            if max_pos >= 0.5:
                success_count += 1
                break

        print(f"Тест епизода {ep + 1}: Макс. позиција {max_pos:.2f}")

    env.close()
    success_rate = success_count / episodes * 100
    print(f"Стапка на успех: {success_rate:.1f}%")
    return success_rate


if __name__ == "__main__":

    print("Тренирање на DQN агент...")
    dqn_agent = train_agent(DQNAgent)
    print("\nТренирање на DDQN агент...")
    ddqn_agent = train_agent(DDQNAgent)

    print("\nТестирање на DQN агент:")
    dqn_success = test_agent(dqn_agent)
    print("\nТестирање на DDQN агент:")
    ddqn_success = test_agent(ddqn_agent)

    print("\nКонечни резултати:")
    print(f"DQN  Стапка на успех: {dqn_success:.1f}%")
    print(f"DDQN Стапка на успех: {ddqn_success:.1f}%")
    winner = "DQN" if dqn_success > ddqn_success else "DDQN"
    print(f"\nВкупен победник: {winner}")