"""Faulty Baseline Agent Implementation for Controlled RL Debugging Experiments.

Implements a controlled, isolated baseline exhibiting confirmed RL defects:
1. Standard DQN maximization bias (single Q-network target calculation)
2. Premature exponential epsilon decay without guided exploration (safe-loop local optimum)
3. Unthrottled per-step gradient updates (high correlation and compute overhead)
4. Configurable terminal bootstrap leakage and non-Markovian state masking for controlled ablation.
"""
from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from warehouse_rl.dqn_agent import QNetwork, ReplayBuffer


@dataclass
class FaultyBaselineHyperparameters:
    """Hyperparameters exhibiting the confirmed baseline faults."""
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 10000
    min_replay_size: int = 200
    target_update_freq: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.970  # Rapid premature collapse
    double_dqn: bool = False      # FAULT: Standard DQN maximization bias
    hidden_dim: int = 128
    leak_terminal_mask: bool = False  # Controlled ablation for terminal bootstrap bug
    omit_package_bit: bool = False    # Controlled ablation for non-Markovian bug


class FaultyBaselineDQNAgent:
    """Baseline agent isolating the confirmed algorithmic and implementation defects."""

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 4,
        params: Optional[FaultyBaselineHyperparameters] = None,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.params = params or FaultyBaselineHyperparameters()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_network = QNetwork(state_dim, action_dim, self.params.hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, self.params.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.params.lr)
        self.criterion = nn.MSELoss()  # FAULT: Standard MSE instead of Huber loss (sensitive to outliers)

        self.replay_buffer = ReplayBuffer(self.params.buffer_capacity)
        self.epsilon = self.params.epsilon_start
        self.total_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action with premature epsilon decay."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def step_learning(self) -> Optional[float]:
        """Perform one step of gradient descent using the faulty baseline target rule."""
        if len(self.replay_buffer) < self.params.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.params.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # FAULT: Standard DQN maximization target
            # y = r + gamma * max_a' Q_target(s', a')
            next_q = self.target_network(next_states_t).max(dim=1, keepdim=True)[0]

            if self.params.leak_terminal_mask:
                # FAULT: Terminal leakage bug (no (1 - done) masking)
                target_q = rewards_t + self.params.gamma * next_q
            else:
                target_q = rewards_t + (1.0 - dones_t) * self.params.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # FAULT: No gradient clipping
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.params.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Rapid exponential decay leading to premature exploration freeze."""
        self.epsilon = max(self.params.epsilon_end, self.epsilon * self.params.epsilon_decay)
