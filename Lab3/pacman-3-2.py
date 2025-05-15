import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from ale_py import ALEInterface
from Lab3.deep_q_learning_2 import DQN,DuelingDQN


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )

    def __len__(self):
        return len(self.buffer)


class ConvModel(nn.Module):
    def __init__(self, num_actions):
        super(ConvModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=6, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        return self.net(x)


class FrameStack:
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = None

    def reset(self, initial_frame):
        self.frames = np.stack([initial_frame] * self.num_frames, axis=0)

    def add_frame(self, frame):
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = frame

    def get_state(self):
        return self.frames


def preprocess_frame(frame):
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    gray = frame.convert('L')
    resized = gray.resize((84, 84), Image.LANCZOS)
    normalized = np.array(resized) / 255.0
    return normalized


def preprocess_reward(reward):
    return np.clip(reward, -1, 1)


class ParallelEnv:
    def __init__(self, env_name, num_envs=4):
        self.envs = [gym.make(env_name) for _ in range(num_envs)]
        self.num_envs = num_envs
        self.executor = ThreadPoolExecutor(max_workers=num_envs)

    def step(self, actions):
        futures = [self.executor.submit(env.step, action)
                   for env, action in zip(self.envs, actions)]
        results = [future.result() for future in futures]
        next_states, rewards, dones, truncated, infos = zip(*results)
        return next_states, rewards, dones, truncated, infos

    def reset(self):
        futures = [self.executor.submit(env.reset) for env in self.envs]
        results = [future.result() for future in futures]
        states, infos = zip(*results)
        return states, infos

    def close(self):
        for env in self.envs:
            env.close()
        self.executor.shutdown()


def test_agent(agent, env, episodes=10, render=False):
    frame_stack = FrameStack(4)
    total_rewards = []

    for _ in range(episodes):
        state, _ = env.reset()
        preprocessed_state = preprocess_frame(state)
        frame_stack.reset(preprocessed_state)

        done = False
        episode_reward = 0

        while not done:
            stacked_state = frame_stack.get_state()
            action = agent.get_action(stacked_state, epsilon=0.05)

            next_state, reward, done, _, _ = env.step(action)
            if render:
                env.render()

            preprocessed_next_state = preprocess_frame(next_state)
            frame_stack.add_frame(preprocessed_next_state)

            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Награда = {episode_reward}")
    return np.mean(total_rewards)


def train_agent(agent, num_episodes):
    frame_stack = FrameStack(4)
    episode_rewards = []
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.9995
    epsilon = epsilon_start
    update_target_freq = 1000
    learning_starts = 1000
    total_steps = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        preprocessed_state = preprocess_frame(state)
        frame_stack.reset(preprocessed_state)

        episode_reward = 0
        done = False

        while not done:
            stacked_state = frame_stack.get_state()
            action = agent.get_action(stacked_state, epsilon)

            next_state, reward, done, _, _ = env.step(action)
            preprocessed_next_state = preprocess_frame(next_state)
            frame_stack.add_frame(preprocessed_next_state)

            processed_reward = preprocess_reward(reward)

            agent.update_memory(
                stacked_state,
                action,
                processed_reward,
                frame_stack.get_state(),
                done
            )

            episode_reward += reward
            total_steps += 1

            if total_steps > learning_starts and total_steps % 4 == 0:
                agent.train()

            if total_steps % update_target_freq == 0:
                agent.update_target_model()

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(episode_reward)

        print(f"Епизода {episode + 1}: Награда = {episode_reward}, Епсилон = {epsilon:.4f}")

    return np.mean(episode_rewards), episode_rewards


if __name__ == '__main__':
    env_name = 'ALE/MsPacman-v5'
    env = gym.make(env_name)

    num_actions = env.action_space.n
    state_dim = (1, env.observation_space.shape[0], env.observation_space.shape[1])

    main_model = ConvModel(num_actions)
    target_model = ConvModel(num_actions)

    agent_DQN = DQN(
        state_space_shape=state_dim,
        num_actions=num_actions,
        model=main_model,
        target_model=target_model,
        learning_rate=0.00025,
        discount_factor=0.99,
        batch_size=32,
        memory_size=100000
    )

    print("Тренирање DQN...")
    avg_reward_dqn, rewards_dqn = train_agent(agent_DQN, 50)
    print(f"Просечна награда (DQN): {avg_reward_dqn}")
    best_dqn_idx = np.argmax(rewards_dqn)
    print(f"Најдобра: {best_dqn_idx + 1} со награда {rewards_dqn[best_dqn_idx]}")

    agent_dueling = DuelingDQN(
        state_space_shape=(4,84,84),
        num_actions=num_actions,
        learning_rate=0.00025,
        discount_factor=0.99,
        batch_size=32,
        memory_size=100000
    )

    print("\nТренирање Dueling DQN...")
    avg_reward_duel, rewards_duel = train_agent(agent_dueling, 50)
    print(f"Просечна награда (Dueling DQN): {avg_reward_duel}")
    best_duel_idx = np.argmax(rewards_duel)
    print(f"Најдобар Dueling DQN {best_duel_idx + 1} со награда {rewards_duel[best_duel_idx]}")

    env = gym.make(env_name, render_mode="human")
    print("\nТестирање DQN...")
    test_agent(agent_DQN, env, episodes=1, render=True)
    print("Тестирање Dueling DQN...")
    test_agent(agent_dueling, env, episodes=1, render=True)
    env.close()
