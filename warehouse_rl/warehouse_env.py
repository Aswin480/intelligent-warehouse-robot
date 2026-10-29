"""Warehouse Markov Decision Process (MDP) Environment.

Models a finite sequential decision-making problem for an autonomous warehouse robot:
M = (S, A, P, R, gamma)

- State space S: (row, col, has_package) for all non-obstacle cells and package status in {0, 1}.
- Action space A: {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}.
- Transitions P(s' | s, a): Deterministic transitions with obstacle and boundary bounce-back.
- Rewards R(s, a, s'): Movement penalty, obstacle collision penalty, boundary penalty,
  package pickup reward, delivery success reward, and timeout penalty.
- Terminal conditions: Package delivered successfully at Delivery destination, or max steps reached.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Tuple, List, Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class WarehouseRewardConfig:
    """Configurable reward parameters for the Warehouse MDP."""
    step_penalty: float = -1.0
    obstacle_penalty: float = -10.0
    boundary_penalty: float = -5.0
    pickup_reward: float = 25.0
    delivery_reward: float = 100.0
    timeout_penalty: float = -20.0


# Actions definition
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_NAMES = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
}

ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


# Default 6x6 warehouse map designed with clear navigable paths, obstacles, pickup, and delivery
DEFAULT_WAREHOUSE_GRID = [
    "S...#.",
    ".##.#.",
    ".#..P.",
    ".#.#..",
    "....#.",
    ".##..D",
]


class WarehouseEnv:
    """Finite Markov Decision Process environment for warehouse robot navigation with package pickup and delivery."""

    ACTIONS: ClassVar[Tuple[int, ...]] = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)

    def __init__(
        self,
        grid: Optional[List[str]] = None,
        gamma: float = 0.95,
        max_steps: int = 100,
        rewards: Optional[WarehouseRewardConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.grid = tuple(grid if grid is not None else DEFAULT_WAREHOUSE_GRID)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.gamma = float(gamma)
        self.max_steps = int(max_steps)
        self.rewards = rewards or WarehouseRewardConfig()
        self.rng = np.random.default_rng(seed)

        self._validate_grid()

        # Locate key positions
        self.start_pos = self._find_cell("S")
        self.pickup_pos = self._find_cell("P")
        self.delivery_pos = self._find_cell("D")

        # Set of obstacle coordinates
        self.obstacles = {
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == "#"
        }

        # Build finite state space
        # A state is (row, col, has_package)
        self.free_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.obstacles
        ]
        self.states = [
            (r, c, p)
            for (r, c) in self.free_cells
            for p in (0, 1)
        ]
        self.state_to_idx = {state: i for i, state in enumerate(self.states)}
        self.idx_to_state = {i: state for i, state in enumerate(self.states)}

        # Episode state variables
        self.robot_pos = self.start_pos
        self.has_package = 0
        self.current_step = 0
        self.episode_return = 0.0
        self.terminated = False
        self.truncated = False
        self.collision_count = 0

    def _validate_grid(self) -> None:
        if not self.grid or not self.grid[0]:
            raise ValueError("Grid cannot be empty")
        for row in self.grid:
            if len(row) != self.cols:
                raise ValueError("All rows in grid must have equal length")
        text = "".join(self.grid)
        if text.count("S") != 1:
            raise ValueError("Grid must contain exactly one Start ('S')")
        if text.count("P") != 1:
            raise ValueError("Grid must contain exactly one Pickup ('P')")
        if text.count("D") != 1:
            raise ValueError("Grid must contain exactly one Delivery ('D')")
        allowed = {"S", "P", "D", ".", "#"}
        if any(ch not in allowed for ch in text):
            raise ValueError(f"Grid contains invalid characters. Allowed: {allowed}")

    def _find_cell(self, char: str) -> Tuple[int, int]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == char:
                    return (r, c)
        raise ValueError(f"Character '{char}' not found in grid")

    @property
    def state(self) -> Tuple[int, int, int]:
        """Current Markov state (row, col, has_package)."""
        return (self.robot_pos[0], self.robot_pos[1], self.has_package)

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_actions(self) -> int:
        return len(self.ACTIONS)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.robot_pos = self.start_pos
        self.has_package = 0
        self.current_step = 0
        self.episode_return = 0.0
        self.terminated = False
        self.truncated = False
        self.collision_count = 0

        obs = self.get_observation()
        info = {
            "robot_pos": self.robot_pos,
            "has_package": self.has_package,
            "step": self.current_step,
            "collision_count": self.collision_count,
        }
        return obs, info

    def get_observation(self, state: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Encode state into normalized vector for neural network."""
        if state is None:
            r, c, p = self.state
        else:
            r, c, p = state

        # Normalized coordinates and package status:
        # [norm_row, norm_col, has_package, dist_to_pickup_norm, dist_to_delivery_norm]
        pickup_dist = (abs(r - self.pickup_pos[0]) + abs(c - self.pickup_pos[1])) / (self.rows + self.cols)
        delivery_dist = (abs(r - self.delivery_pos[0]) + abs(c - self.delivery_pos[1])) / (self.rows + self.cols)

        obs = np.array([
            r / (self.rows - 1),
            c / (self.cols - 1),
            float(p),
            pickup_dist,
            delivery_dist,
        ], dtype=np.float32)
        return obs

    def transition_dynamics(
        self, state: Tuple[int, int, int], action: int
    ) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        """Core Markov transition function: P(s' | s, a) and R(s, a, s').

        Returns:
            next_state: (r', c', p')
            reward: scalar immediate reward
            is_terminal: True if package delivered successfully
            event_info: details about collision, pickup, delivery
        """
        r, c, p = state
        delta = ACTION_DELTAS[action]
        nr, nc = r + delta[0], c + delta[1]

        # 1. Boundary collision
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return (r, c, p), self.rewards.boundary_penalty, False, {"event": "boundary_collision"}

        # 2. Obstacle collision
        if (nr, nc) in self.obstacles:
            return (r, c, p), self.rewards.obstacle_penalty, False, {"event": "obstacle_collision"}

        # 3. Valid move to non-obstacle cell
        next_p = p
        event = "move"
        reward = self.rewards.step_penalty

        # Check pickup
        if (nr, nc) == self.pickup_pos and p == 0:
            next_p = 1
            reward += self.rewards.pickup_reward
            event = "pickup"

        # Check delivery
        if (nr, nc) == self.delivery_pos and next_p == 1:
            reward += self.rewards.delivery_reward
            event = "delivery"
            return (nr, nc, next_p), reward, True, {"event": event}

        return (nr, nc, next_p), reward, False, {"event": event}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one sequential action step in the environment."""
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is already done. Call reset() before stepping.")

        next_state, reward, is_terminal, event_info = self.transition_dynamics(self.state, action)

        self.robot_pos = (next_state[0], next_state[1])
        self.has_package = next_state[2]
        self.current_step += 1
        self.episode_return += reward

        if event_info["event"] in ("boundary_collision", "obstacle_collision"):
            self.collision_count += 1

        self.terminated = is_terminal

        if not self.terminated and self.current_step >= self.max_steps:
            self.truncated = True
            reward += self.rewards.timeout_penalty
            self.episode_return += self.rewards.timeout_penalty
            event_info["event"] = "timeout"

        obs = self.get_observation()
        info = {
            "next_state": next_state,
            "event": event_info["event"],
            "robot_pos": self.robot_pos,
            "has_package": self.has_package,
            "step": self.current_step,
            "episode_return": self.episode_return,
            "collision_count": self.collision_count,
            "success": self.terminated,
        }

        return obs, float(reward), self.terminated, self.truncated, info

    def render_ascii(self) -> str:
        """Render ASCII warehouse representation showing current robot and package status."""
        lines = []
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if (r, c) == self.robot_pos:
                    row_str.append("🤖*" if self.has_package else "🤖")
                elif (r, c) == self.start_pos:
                    row_str.append("S ")
                elif (r, c) == self.pickup_pos:
                    row_str.append("📦" if not self.has_package else "· ")
                elif (r, c) == self.delivery_pos:
                    row_str.append("🏁")
                elif (r, c) in self.obstacles:
                    row_str.append("██")
                else:
                    row_str.append(". ")
            lines.append("".join(row_str))
        return "\n".join(lines)
