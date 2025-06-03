from agents import *


def display_board(observation):
    obs_reshaped = observation.reshape(3, 3, 2)
    board = np.zeros((3, 3))
    board[obs_reshaped[:, :, 0] == 1] = 1
    board[obs_reshaped[:, :, 1] == 1] = -1

    print("\n   0   1   2")
    for i in range(3):
        print(f"{i}  ", end="")
        for j in range(3):
            if board[i, j] == 1:
                print("X", end="")
            elif board[i, j] == -1:
                print("O", end="")
            else:
                print(" ", end="")
            if j < 2:
                print(" | ", end="")
        print()
    print()


def human_player_action(observation, action_mask):
    display_board(observation)
    valid_actions = np.where(action_mask)[0]

    while True:
        try:
            print(f"Valid moves: {valid_actions}")
            action = int(input("Enter your move (0-8): "))
            if action in valid_actions:
                return action
            else:
                print("Invalid move! Try again.")
        except ValueError:
            print("Enter a valid number!")


def q_learning_vs_smart_random_test(show_boards=False):
    env = tictactoe_v3.env()

    q_agent = QLearningAgent(
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.01
    )

    smart_random_agent = SmartRandomAgent()

    print("Training Q-Learning agent...")

    for episode in range(1000):
        env.reset()
        agent_states = {}
        agent_actions = {}
        game_over = False

        for agent_name in env.agent_iter():
            if game_over:
                break

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                if agent_name == 'player_1' and agent_name in agent_states:
                    q_agent.update_q_table(
                        agent_states[agent_name],
                        agent_actions[agent_name],
                        reward,
                        None,
                        None,
                        done=True
                    )
                game_over = True
                break

            current_state = observation['observation']
            action_mask = observation['action_mask']

            if agent_name == 'player_1' and agent_name in agent_states:
                q_agent.update_q_table(
                    agent_states[agent_name],
                    agent_actions[agent_name],
                    reward,
                    current_state,
                    action_mask,
                    done=False
                )

            if agent_name == 'player_1':
                action = q_agent.choose_action(current_state, action_mask)
                agent_states[agent_name] = current_state.copy()
                agent_actions[agent_name] = action
            else:
                action = smart_random_agent.choose_action(current_state, action_mask)

            env.step(action)

        q_agent.decay_epsilon()

    print("Testing trained Q-Learning agent vs Smart Random agent over 50 iterations...")

    results = {'q_wins': 0, 'smart_random_wins': 0, 'draws': 0}
    q_rewards = []
    smart_random_rewards = []

    for episode in range(50):
        env.reset()
        total_rewards = {'player_1': 0, 'player_2': 0}
        game_over = False
        step_count = 0

        if show_boards and episode % 10 == 0:
            print(f"\nEpisode {episode + 1}")
            print("-" * 30)

        for agent_name in env.agent_iter():
            if game_over:
                break

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                total_rewards[agent_name] += reward
                game_over = True
                break

            current_state = observation['observation']
            action_mask = observation['action_mask']
            total_rewards[agent_name] += reward

            if show_boards and episode % 10 == 0:
                print(f"\nMove {step_count + 1} - {agent_name}")
                display_board(current_state)

            if agent_name == 'player_1':
                action = q_agent.choose_action(current_state, action_mask, training=False)
            else:
                action = smart_random_agent.choose_action(current_state, action_mask)

            if show_boards and episode % 10 == 0:
                row, col = action // 3, action % 3
                symbol = 'X' if agent_name == 'player_1' else 'O'
                print(f"{agent_name} plays {symbol} at position ({row}, {col})")

            env.step(action)
            step_count += 1

            if show_boards and episode % 10 == 0:
                time.sleep(0.5)

        if total_rewards['player_1'] > total_rewards['player_2']:
            results['q_wins'] += 1
        elif total_rewards['player_2'] > total_rewards['player_1']:
            results['smart_random_wins'] += 1
        else:
            results['draws'] += 1

        q_rewards.append(total_rewards['player_1'])
        smart_random_rewards.append(total_rewards['player_2'])

        if show_boards and episode % 10 == 0:
            print("\nFinal state:")
            display_board(observation['observation'])

            if total_rewards['player_1'] > total_rewards['player_2']:
                print("Winner: Player 1 (X)")
            elif total_rewards['player_2'] > total_rewards['player_1']:
                print("Winner: Player 2 (O)")
            else:
                print("Draw!")

            print(f"Rewards: Player 1: {total_rewards['player_1']}, Player 2: {total_rewards['player_2']}")

    avg_q_reward = np.mean(q_rewards)
    avg_smart_random_reward = np.mean(smart_random_rewards)

    print("\n" + "=" * 60)
    print("Q-LEARNING vs SMART RANDOM AGENT - RESULTS")
    print("=" * 60)
    print(f"Total test episodes: 50")
    print(f"Q-Learning agent wins: {results['q_wins']} ({results['q_wins'] / 50 * 100:.1f}%)")
    print(f"Smart Random agent wins: {results['smart_random_wins']} ({results['smart_random_wins'] / 50 * 100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws'] / 50 * 100:.1f}%)")
    print(f"Average reward Q-Learning agent: {avg_q_reward:.3f}")
    print(f"Average reward Smart Random agent: {avg_smart_random_reward:.3f}")
    print(f"Q-table size: {len(q_agent.q_table)} states")
    print(f"Final epsilon: {q_agent.epsilon:.4f}")

    if avg_q_reward > avg_smart_random_reward:
        print("\nQ-Learning agent has higher average reward")
    elif avg_smart_random_reward > avg_q_reward:
        print("\nSmart Random agent has higher average reward")
    else:
        print("\nBoth agents have equal average reward")

    if results['q_wins'] > results['smart_random_wins']:
        print("Q-Learning agent has more wins")
    elif results['smart_random_wins'] > results['q_wins']:
        print("Smart Random agent has more wins")
    else:
        print("Both agents have equal number of wins")


def run_visual_game(mode='training', num_episodes=50, show_every=10, human_play=False):
    env = tictactoe_v3.env()

    q_agent = QLearningAgent(
        learning_rate=0.15,
        discount_factor=0.98,
        epsilon=0.95,
        epsilon_decay=0.995,
        min_epsilon=0.02
    )

    smart_random_agent = SmartRandomAgent()

    results = {'q_wins': 0, 'smart_random_wins': 0, 'draws': 0}
    episode_rewards = []

    print(f"Starting {mode} mode...")
    print("=" * 60)

    for episode in range(num_episodes):
        env.reset()

        agent_states = {}
        agent_actions = {}
        total_rewards = {'player_1': 0, 'player_2': 0}

        show_this_episode = (episode % show_every == 0) or human_play
        game_over = False
        step_count = 0

        if show_this_episode:
            print(f"\nEpisode {episode + 1}")
            print("-" * 30)

        for agent_name in env.agent_iter():
            if game_over:
                break

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                if agent_name == 'player_1' and agent_name in agent_states:
                    q_agent.update_q_table(
                        agent_states[agent_name],
                        agent_actions[agent_name],
                        reward,
                        None,
                        None,
                        done=True
                    )

                total_rewards[agent_name] += reward
                game_over = True
                break

            current_state = observation['observation']
            action_mask = observation['action_mask']

            if agent_name == 'player_1' and agent_name in agent_states:
                q_agent.update_q_table(
                    agent_states[agent_name],
                    agent_actions[agent_name],
                    reward,
                    current_state,
                    action_mask,
                    done=False
                )

            total_rewards[agent_name] += reward

            if show_this_episode and not human_play:
                print(f"\nMove {step_count + 1} - {agent_name}")
                display_board(current_state)

            if agent_name == 'player_1':
                if human_play and agent_name == 'player_1':
                    action = human_player_action(current_state, action_mask)
                else:
                    action = q_agent.choose_action(current_state, action_mask)
                agent_states[agent_name] = current_state.copy()
                agent_actions[agent_name] = action
            else:
                action = smart_random_agent.choose_action(current_state, action_mask)

            if show_this_episode:
                row, col = action // 3, action % 3
                symbol = 'X' if agent_name == 'player_1' else 'O'
                print(f"{agent_name} plays {symbol} at position ({row}, {col})")

            env.step(action)
            step_count += 1

            if show_this_episode and not human_play:
                time.sleep(0.5)

        if total_rewards['player_1'] > total_rewards['player_2']:
            results['q_wins'] += 1
            q_agent.update_stats('win')
        elif total_rewards['player_2'] > total_rewards['player_1']:
            results['smart_random_wins'] += 1
            q_agent.update_stats('loss')
        else:
            results['draws'] += 1
            q_agent.update_stats('draw')

        episode_rewards.append(total_rewards['player_1'])

        if show_this_episode:
            print("\nFinal state:")
            display_board(observation['observation'])

            if total_rewards['player_1'] > total_rewards['player_2']:
                print("Winner: Player 1 (X)")
            elif total_rewards['player_2'] > total_rewards['player_1']:
                print("Winner: Player 2 (O)")
            else:
                print("Draw!")

            print(f"Rewards: Player 1: {total_rewards['player_1']}, Player 2: {total_rewards['player_2']}")

        q_agent.decay_epsilon()

        if (episode + 1) % show_every == 0 and not human_play:
            win_rate = results['q_wins'] / (episode + 1) * 100
            print(f"\nEpisode {episode + 1}: Q-Agent win rate: {win_rate:.1f}%, Epsilon: {q_agent.epsilon:.3f}")

    if not human_play:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Total episodes: {num_episodes}")
        print(f"Q-Learning agent wins: {results['q_wins']} ({results['q_wins'] / num_episodes * 100:.1f}%)")
        print(
            f"Smart Random agent wins: {results['smart_random_wins']} ({results['smart_random_wins'] / num_episodes * 100:.1f}%)")
        print(f"Draws: {results['draws']} ({results['draws'] / num_episodes * 100:.1f}%)")
        print(f"Average reward Q-agent: {np.mean(episode_rewards):.3f}")
        print(f"Number of learned states: {len(q_agent.q_table)}")
        print(f"Final epsilon: {q_agent.epsilon:.4f}")

    return q_agent, results


def evaluate_agent(agent, num_episodes=50):
    env = tictactoe_v3.env()
    smart_random_agent = SmartRandomAgent()

    results = {'q_wins': 0, 'smart_random_wins': 0, 'draws': 0}
    q_rewards = []
    random_rewards = []

    print(f"Evaluating agent over {num_episodes} episodes...")

    for episode in range(num_episodes):
        env.reset()
        agent_states = {}
        agent_actions = {}
        total_rewards = {'player_1': 0, 'player_2': 0}
        game_over = False

        for agent_name in env.agent_iter():
            if game_over:
                break

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                total_rewards[agent_name] += reward
                game_over = True
                break

            current_state = observation['observation']
            action_mask = observation['action_mask']

            if agent_name == 'player_1':
                action = agent.choose_action(current_state, action_mask, training=False)
            else:
                action = smart_random_agent.choose_action(current_state, action_mask)

            env.step(action)
            total_rewards[agent_name] += reward

        if total_rewards['player_1'] > total_rewards['player_2']:
            results['q_wins'] += 1
        elif total_rewards['player_2'] > total_rewards['player_1']:
            results['smart_random_wins'] += 1
        else:
            results['draws'] += 1

        q_rewards.append(total_rewards['player_1'])
        random_rewards.append(total_rewards['player_2'])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Q-Learning agent wins: {results['q_wins']} ({results['q_wins'] / num_episodes * 100:.1f}%)")
    print(
        f"Smart Random agent wins: {results['smart_random_wins']} ({results['smart_random_wins'] / num_episodes * 100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws'] / num_episodes * 100:.1f}%)")
    print(f"Average reward Q-agent: {np.mean(q_rewards):.3f}")
    print(f"Average reward Random agent: {np.mean(random_rewards):.3f}")

    if np.mean(q_rewards) > np.mean(random_rewards):
        print("Q-Learning agent has higher average reward")
    else:
        print("Smart Random agent has higher average reward")

    if results['q_wins'] > results['smart_random_wins']:
        print("Q-Learning agent has more wins")
    else:
        print("Smart Random agent has more wins")


def dual_q_learning():
    env = tictactoe_v3.env()

    q_agent_1 = QLearningAgent(
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.01
    )

    q_agent_2 = QLearningAgent(
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.01
    )

    smart_random_agent = SmartRandomAgent()

    print("Training first Q-Learning agent...")
    for episode in range(1000):
        env.reset()
        agent_states = {}
        agent_actions = {}
        game_over = False

        for agent_name in env.agent_iter():
            if game_over:
                break

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                if agent_name == 'player_1' and agent_name in agent_states:
                    q_agent_1.update_q_table(
                        agent_states[agent_name],
                        agent_actions[agent_name],
                        reward,
                        None,
                        None,
                        done=True
                    )
                game_over = True
                break

            current_state = observation['observation']
            action_mask = observation['action_mask']

            if agent_name == 'player_1' and agent_name in agent_states:
                q_agent_1.update_q_table(
                    agent_states[agent_name],
                    agent_actions[agent_name],
                    reward,
                    current_state,
                    action_mask,
                    done=False
                )

            if agent_name == 'player_1':
                action = q_agent_1.choose_action(current_state, action_mask)
                agent_states[agent_name] = current_state.copy()
                agent_actions[agent_name] = action
            else:
                action = smart_random_agent.choose_action(current_state, action_mask)

            env.step(action)

        q_agent_1.decay_epsilon()

    print("Training second Q-Learning agent...")
    for episode in range(1000):
        env.reset()
        agent_states = {}
        agent_actions = {}
        game_over = False

        for agent_name in env.agent_iter():
            if game_over:
                break

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                if agent_name == 'player_2' and agent_name in agent_states:
                    q_agent_2.update_q_table(
                        agent_states[agent_name],
                        agent_actions[agent_name],
                        reward,
                        None,
                        None,
                        done=True
                    )
                game_over = True
                break

            current_state = observation['observation']
            action_mask = observation['action_mask']

            if agent_name == 'player_2' and agent_name in agent_states:
                q_agent_2.update_q_table(
                    agent_states[agent_name],
                    agent_actions[agent_name],
                    reward,
                    current_state,
                    action_mask,
                    done=False
                )

            if agent_name == 'player_2':
                action = q_agent_2.choose_action(current_state, action_mask)
                agent_states[agent_name] = current_state.copy()
                agent_actions[agent_name] = action
            else:
                action = smart_random_agent.choose_action(current_state, action_mask)

            env.step(action)

        q_agent_2.decay_epsilon()

    print("Testing trained Q-Learning agents against each other over 50 iterations...")

    results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}
    agent1_rewards = []
    agent2_rewards = []

    for episode in range(50):
        env.reset()
        total_rewards = {'player_1': 0, 'player_2': 0}
        game_over = False

        for agent_name in env.agent_iter():
            if game_over:
                break

            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                total_rewards[agent_name] += reward
                game_over = True
                break

            current_state = observation['observation']
            action_mask = observation['action_mask']
            total_rewards[agent_name] += reward

            if agent_name == 'player_1':
                action = q_agent_1.choose_action(current_state, action_mask, training=False)
            else:
                action = q_agent_2.choose_action(current_state, action_mask, training=False)

            env.step(action)

        if total_rewards['player_1'] > total_rewards['player_2']:
            results['agent1_wins'] += 1
        elif total_rewards['player_2'] > total_rewards['player_1']:
            results['agent2_wins'] += 1
        else:
            results['draws'] += 1

        agent1_rewards.append(total_rewards['player_1'])
        agent2_rewards.append(total_rewards['player_2'])

    avg_agent1_reward = np.mean(agent1_rewards)
    avg_agent2_reward = np.mean(agent2_rewards)

    print("\n" + "=" * 60)
    print("DUAL Q-LEARNING AGENTS - RESULTS")
    print("=" * 60)
    print(f"Total test episodes: 50")
    print(f"Q-Learning agent 1 wins: {results['agent1_wins']} ({results['agent1_wins'] / 50 * 100:.1f}%)")
    print(f"Q-Learning agent 2 wins: {results['agent2_wins']} ({results['agent2_wins'] / 50 * 100:.1f}%)")
    print(f"Draws: {results['draws']} ({results['draws'] / 50 * 100:.1f}%)")
    print(f"Average reward Q-Learning agent 1: {avg_agent1_reward:.3f}")
    print(f"Average reward Q-Learning agent 2: {avg_agent2_reward:.3f}")
    print(f"Q-table size agent 1: {len(q_agent_1.q_table)} states")
    print(f"Q-table size agent 2: {len(q_agent_2.q_table)} states")

    if avg_agent1_reward > avg_agent2_reward:
        print("\nQ-Learning agent 1 has higher average reward")
    elif avg_agent2_reward > avg_agent1_reward:
        print("\nQ-Learning agent 2 has higher average reward")
    else:
        print("\nBoth agents have equal average reward")

    if results['agent1_wins'] > results['agent2_wins']:
        print("Q-Learning agent 1 has more wins")
    elif results['agent2_wins'] > results['agent1_wins']:
        print("Q-Learning agent 2 has more wins")
    else:
        print("Both agents have equal number of wins")


def main():
    while True:
        print("\n" + "=" * 50)
        print("=" * 50)
        print("1. Train agent with board visualization (50 iterations)")
        print("2. AI vs human")
        print("3. Exit")
        print("4. Q-Learning vs Smart Random (50 test games, just prints)")
        print("5. Dual Q-Learning (Two independent Q-learning agents)")

        choice = input("\nSelect option (1-5): ")

        if choice == '1':
            print("\nTraining in progress...")
            q_learning_vs_smart_random_test(show_boards=True)

        elif choice == '2':
            print("\nTraining the agent first...")
            agent, _ = run_visual_game(mode='quick_training', num_episodes=50, show_every=999)
            print("\nNow play against the agent!")
            input("Press Enter to start...")
            run_visual_game(mode='human_vs_ai', num_episodes=1, show_every=1, human_play=True)

        elif choice == '3':
            break

        elif choice == '4':
            q_learning_vs_smart_random_test(show_boards=False)

        elif choice == '5':
            dual_q_learning()

        else:
            print("Invalid option!")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    main()
