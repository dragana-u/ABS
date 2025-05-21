import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN:
    def __init__(self, state_space_shape, num_actions, model, target_model, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = model
        self.target_model = target_model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def load(self, model_name, episode):
        self.model.load_state_dict(torch.load(f'dqn_{model_name}_{episode}.pt'))

    def save(self, model_name, episode):
        torch.save(self.model.state_dict(), f'dqn_{model_name}_{episode}.pt')

    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor).detach().clone().squeeze()
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    max_future_q = torch.max(self.target_model(next_state_tensor)).item()
                target[action] = reward + self.discount_factor * max_future_q

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()


class DDQN(DQN):
    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor).detach().clone().squeeze()
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                    best_action = torch.argmax(next_q_values)
                    max_q = self.target_model(next_state_tensor)[0][best_action].item()
                target[action] = reward + self.discount_factor * max_q

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()


class DuelingDQNModel(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DuelingDQNModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, num_actions)

    def forward(self, x):
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DuelingDQN:
    def __init__(self, state_space_shape, num_actions, learning_rate=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        self.state_space_shape = state_space_shape
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = DuelingDQNModel(state_space_shape, num_actions)
        self.target_model = DuelingDQNModel(state_space_shape, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def load(self, model_name, episode):
        self.model.load_state_dict(torch.load(f'duelingdqn_{model_name}_{episode}.pt'))

    def save(self, model_name, episode):
        torch.save(self.model.state_dict(), f'duelingdqn_{model_name}_{episode}.pt')

    def train(self):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        states, targets = [], []

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            target = self.model(state_tensor).detach().clone().squeeze()
            if done:
                target[action] = reward
            else:
                with torch.no_grad():
                    max_future_q = torch.max(self.target_model(next_state_tensor)).item()
                target[action] = reward + self.discount_factor * max_future_q

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        targets_tensor = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.input_state = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.input_action = nn.Sequential(
            nn.Linear(action_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )

    def forward(self, state, action):
        state_features = self.input_state(state)
        action_features = self.input_action(action)
        concat = torch.cat([state_features, action_features], dim=1)
        return self.output(concat)


class DDPG:
    def __init__(self, state_space_shape, action_space_shape, learning_rate_actor=0.1, learning_rate_critic=0.1,
                 discount_factor=0.95, batch_size=16, memory_size=100):
        self.state_dim = state_space_shape if isinstance(state_space_shape, int) else state_space_shape[0]
        self.action_dim = action_space_shape if isinstance(action_space_shape, int) else action_space_shape[0]

        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        self.actor = Actor(self.state_dim, self.action_dim)
        self.target_actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)
        self.loss_fn = nn.MSELoss()

        self.update_target_model()

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self, tau=1.0):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def _get_discrete_action(self, state, epsilon=0):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.actor(state_tensor)).item()

    def _get_continuous_action(self, state, epsilon=0):
        if np.random.random() < epsilon:
            return np.random.uniform(low=0.0, high=1.0, size=self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.actor(state_tensor).squeeze(0).numpy()

    def get_action(self, state, epsilon=0, discrete=True):
        if discrete:
            return self._get_discrete_action(state, epsilon)
        else:
            return self._get_continuous_action(state, epsilon)

    def save(self, model_name, episode):
        torch.save(self.actor.state_dict(), f'actor_{model_name}_{episode}.pt')
        torch.save(self.critic.state_dict(), f'critic_{model_name}_{episode}.pt')

    def load(self, model_name, episode):
        self.actor.load_state_dict(torch.load(f'actor_{model_name}_{episode}.pt'))
        self.critic.load_state_dict(torch.load(f'critic_{model_name}_{episode}.pt'))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.tensor(np.array([m[0] for m in minibatch]), dtype=torch.float32)
        actions = torch.tensor(np.array([m[1] for m in minibatch]), dtype=torch.float32)
        rewards = torch.tensor(np.array([m[2] for m in minibatch]), dtype=torch.float32)
        next_states = torch.tensor(np.array([m[3] for m in minibatch]), dtype=torch.float32)
        dones = torch.tensor(np.array([m[4] for m in minibatch]), dtype=torch.float32)

        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions).squeeze()
            y = rewards + self.discount_factor * target_q * (1 - dones)

        current_q = self.critic(states, actions).squeeze()
        critic_loss = self.loss_fn(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions = self.actor(states)
        actor_loss = -self.critic(states, new_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_space_shape, theta=.2, sigma=0.15, dt=1e-2, x0=None):
        """
        Initializes Ornstein Uhlenbeck action noise process.
        :param action_space_shape: shape of the action space
        :param theta: the rate of mean reversion
        :param sigma: scale of the noise
        :param dt: the timestep for the noise
        :param x0: the initial value for noise
        """
        self.mu = np.zeros(action_space_shape)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Returns action noise for one timestep.
        """
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        """
        Resets the Ornstein Uhlenbeck action noise to the initial position.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)