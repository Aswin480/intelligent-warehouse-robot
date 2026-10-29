"""Warehouse grid MDP used by the browser simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

State = tuple[int, int]
Action = str


@dataclass(frozen=True)
class RewardConfig:
    goal: float = 100.0
    movement: float = -1.0
    obstacle: float = -10.0
    boundary: float = -5.0


class WarehouseMDP:
    """Finite deterministic MDP for a 2D warehouse.

    A state is a ``(row, column)`` coordinate. Walls are excluded from the
    state space. The transition is deterministic: invalid moves stay in the
    current state and receive the corresponding collision penalty.
    """

    ACTIONS: ClassVar[tuple[Action, ...]] = ("UP", "DOWN", "LEFT", "RIGHT")
    DELTAS: ClassVar[dict[Action, State]] = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
    }

    def __init__(
        self,
        grid: list[str] | tuple[str, ...],
        *,
        gamma: float = 0.9,
        rewards: RewardConfig | None = None,
    ) -> None:
        normalized_grid = tuple(str(row) for row in grid)
        self._validate_grid(normalized_grid, gamma)
        self.grid = normalized_grid
        self.rows = len(normalized_grid)
        self.cols = len(normalized_grid[0])
        self.gamma = float(gamma)
        self.rewards = rewards or RewardConfig()
        self.start = self._find_single("S")
        self.goal = self._find_single("G")
        self.states = tuple(
            (row, column)
            for row in range(self.rows)
            for column in range(self.cols)
            if self.grid[row][column] != "#"
        )
        self.index = {state: number for number, state in enumerate(self.states)}

    @staticmethod
    def _validate_grid(grid: tuple[str, ...], gamma: float) -> None:
        if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("grid must contain equally sized, non-empty rows")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must be between 0 and 1")
        symbols = "".join(grid)
        if symbols.count("S") != 1 or symbols.count("G") != 1:
            raise ValueError("grid must contain exactly one S and one G")
        if any(cell not in {"S", "G", ".", "#"} for cell in symbols):
            raise ValueError("grid cells must be S, G, ., or #")

    def _find_single(self, symbol: str) -> State:
        return next(
            (state for state in self._all_coordinates() if self.grid[state[0]][state[1]] == symbol),
            None,
        ) or self._raise_missing(symbol)

    def _raise_missing(self, symbol: str) -> State:
        raise ValueError(f"grid must contain exactly one {symbol}")

    def _all_coordinates(self):
        return (
            (row, column)
            for row in range(self.rows)
            for column in range(self.cols)
        )

    @property
    def state_count(self) -> int:
        return len(self.states)

    def is_terminal(self, state: State) -> bool:
        return state == self.goal

    def transition(self, state: State, action: Action) -> tuple[State, float]:
        """Return ``(next_state, reward)`` for one deterministic action."""
        if action not in self.ACTIONS:
            raise ValueError(f"unknown action: {action}")
        if self.is_terminal(state):
            return state, 0.0
        delta = self.DELTAS[action]
        candidate = (state[0] + delta[0], state[1] + delta[1])
        if not (0 <= candidate[0] < self.rows and 0 <= candidate[1] < self.cols):
            return state, self.rewards.boundary
        if self.grid[candidate[0]][candidate[1]] == "#":
            return state, self.rewards.obstacle
        if candidate == self.goal:
            return candidate, self.rewards.goal
        return candidate, self.rewards.movement

    def state_reward(self, state: State, action: Action) -> float:
        return self.transition(state, action)[1]