import random
import numpy as np
from pettingzoo.classic import tictactoe_v3
from collections import defaultdict
import time


class QLearningAgent:
    def __init__(self, learning_rate=0.15, discount_factor=0.98, epsilon=0.95, epsilon_decay=0.995, min_epsilon=0.02):
        self.q_table = defaultdict(lambda: np.zeros(9))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.episode_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.draw_count = 0
        self.visit_count = defaultdict(lambda: np.zeros(9))

    def state_to_key(self, observation):
        return tuple(observation.flatten())

    def observation_to_board(self, observation):
        obs_reshaped = observation.reshape(3, 3, 2)
        board = np.zeros((3, 3))
        board[obs_reshaped[:, :, 0] == 1] = 1
        board[obs_reshaped[:, :, 1] == 1] = -1
        return board

    def get_strategic_value(self, observation, action):
        board = self.observation_to_board(observation)
        row, col = action // 3, action % 3

        strategic_value = 0

        temp_board = board.copy()
        temp_board[row, col] = 1

        if self._check_win(temp_board, 1):
            strategic_value += 200

        temp_board[row, col] = -1
        if self._check_win(temp_board, -1):
            strategic_value += 150

        strategic_value += self._evaluate_position(board, action, 1)
        strategic_value -= self._evaluate_position(board, action, -1)

        if action == 4:
            strategic_value += 15

        if action in [0, 2, 6, 8]:
            strategic_value += 8

        return strategic_value

    def _evaluate_position(self, board, action, player):
        row, col = action // 3, action % 3
        temp_board = board.copy()
        temp_board[row, col] = player

        score = 0

        lines = [
            temp_board[row, :],
            temp_board[:, col],
            np.diag(temp_board) if row == col else None,
            np.diag(np.fliplr(temp_board)) if row + col == 2 else None
        ]

        for line in lines:
            if line is not None:
                player_count = np.sum(line == player)
                empty_count = np.sum(line == 0)
                if player_count == 2 and empty_count == 1:
                    score += 50
                elif player_count == 1 and empty_count == 2:
                    score += 10

        return score

    def _check_win(self, board, player):
        for i in range(3):
            if np.all(board[i, :] == player):
                return True
        for j in range(3):
            if np.all(board[:, j] == player):
                return True
        if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False

    def choose_action(self, observation, action_mask, training=True):
        state_key = self.state_to_key(observation)
        valid_actions = np.where(action_mask)[0]

        if training and random.random() < self.epsilon:
            action_values = []
            for action in valid_actions:
                strategic_bonus = self.get_strategic_value(observation, action)
                exploration_bonus = 20 / (1 + self.visit_count[state_key][action])
                action_values.append(strategic_bonus + exploration_bonus + random.random() * 5)

            best_idx = np.argmax(action_values)
            return valid_actions[best_idx]
        else:
            q_values = self.q_table[state_key].copy()

            for action in valid_actions:
                strategic_bonus = self.get_strategic_value(observation, action)
                q_values[action] += strategic_bonus * 0.02

            valid_q_values = q_values[valid_actions]
            best_actions = valid_actions[valid_q_values == np.max(valid_q_values)]
            return random.choice(best_actions)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def update_q_table(self, state, action, reward, next_state, next_action_mask, done=False):
        state_key = self.state_to_key(state)
        self.visit_count[state_key][action] += 1

        if done:
            target = reward
        else:
            next_state_key = self.state_to_key(next_state)
            valid_next_actions = np.where(next_action_mask)[0]
            if len(valid_next_actions) > 0:
                max_next_q = np.max(self.q_table[next_state_key][valid_next_actions])
            else:
                max_next_q = 0
            target = reward + self.discount_factor * max_next_q

        current_q = self.q_table[state_key][action]
        td_error = target - current_q

        adaptive_lr = self.learning_rate / (1 + self.visit_count[state_key][action] * 0.01)
        adaptive_lr = max(adaptive_lr, 0.01)

        self.q_table[state_key][action] = current_q + adaptive_lr * td_error

    def update_stats(self, result):
        if result == 'win':
            self.win_count += 1
        elif result == 'loss':
            self.loss_count += 1
        else:
            self.draw_count += 1


class SmartRandomAgent:
    def observation_to_board(self, observation):
        obs_reshaped = observation.reshape(3, 3, 2)
        board = np.zeros((3, 3))
        board[obs_reshaped[:, :, 0] == 1] = 1
        board[obs_reshaped[:, :, 1] == 1] = -1
        return board

    def choose_action(self, observation, action_mask):
        valid_actions = np.where(action_mask)[0]
        board = self.observation_to_board(observation)

        for action in valid_actions:
            row, col = action // 3, action % 3
            temp_board = board.copy()
            temp_board[row, col] = -1
            if self._check_win(temp_board, -1):
                return action

        for action in valid_actions:
            row, col = action // 3, action % 3
            temp_board = board.copy()
            temp_board[row, col] = 1
            if self._check_win(temp_board, 1):
                return action

        if 4 in valid_actions:
            if random.random() < 0.7:
                return 4

        corners = [action for action in valid_actions if action in [0, 2, 6, 8]]
        if corners and random.random() < 0.5:
            return random.choice(corners)

        return random.choice(valid_actions)

    def _check_win(self, board, player):
        for i in range(3):
            if np.all(board[i, :] == player):
                return True
        for j in range(3):
            if np.all(board[:, j] == player):
                return True
        if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False


class RandomAgent:
    def choose_action(self, observation, action_mask):
        valid_actions = np.where(action_mask)[0]
        return random.choice(valid_actions)
