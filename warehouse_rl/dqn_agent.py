"""Deep Q-Network (DQN) and Double DQN agent with Experience Replay.

Implements the fundamental Bellman optimality equation in deep RL:
Q*(s, a) = E[ R(s, a, s') + gamma * max_a' Q*(s', a') | s, a ]
"""
from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Deep Q-Network approximating action-value function Q(s, a; theta)."""

    def __init__(self, state_dim: int = 5, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Cyclic experience replay memory buffer to decorrelate consecutive transitions."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class DQNHyperparameters:
    """Hyperparameters for DQN training."""
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 25000
    min_replay_size: int = 200
    target_update_freq: int = 200  # steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    double_dqn: bool = True
    hidden_dim: int = 128


class DQNAgent:
    """DQN Agent with target network, experience replay, and epsilon-greedy exploration."""

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 4,
        params: Optional[DQNHyperparameters] = None,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.params = params or DQNHyperparameters()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_network = QNetwork(state_dim, action_dim, self.params.hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, self.params.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.params.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stable gradient descent

        self.replay_buffer = ReplayBuffer(self.params.buffer_capacity)
        self.epsilon = self.params.epsilon_start
        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy policy or greedy policy during evaluation."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return raw Q-values for all actions given a state observation."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_network(state_tensor).squeeze(0).cpu().numpy()
        return q_vals

    def step_learning(self) -> Optional[float]:
        """Perform one step of gradient descent using Bellman temporal-difference target."""
        if len(self.replay_buffer) < self.params.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.params.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q-values: Q(s, a; theta)
        current_q = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.params.double_dqn:
                # Double DQN: action selected by online network, evaluated by target network
                # y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
                next_online_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states_t).gather(1, next_online_actions)
            else:
                # Standard DQN:
                # y = r + gamma * max_a' Q_target(s', a')
                next_q = self.target_network(next_states_t).max(dim=1, keepdim=True)[0]

            # Terminal states yield target = reward (no future discounted return)
            target_q = rewards_t + (1.0 - dones_t) * self.params.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for numerical stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1

        # Periodic update of target network
        if self.total_steps % self.params.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Decay exploration rate epsilon after each episode."""
        self.epsilon = max(self.params.epsilon_end, self.epsilon * self.params.epsilon_decay)

    def save_checkpoint(self, filepath: str) -> None:
        """Save model weights and hyperparameters."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "params": self.params.__dict__,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model weights and hyperparameters from checkpoint."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file {filepath} does not exist.")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
