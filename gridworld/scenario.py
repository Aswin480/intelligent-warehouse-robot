"""Load a navigation scenario (grid + rewards + coords) from YAML."""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from gridworld.grid_coords import GridConfig
from gridworld.mdp import GridMDP


@dataclass
class Scenario:
    grid: list
    step_reward: float
    terminal_rewards: dict
    slip_prob: float
    gamma: float
    coords: GridConfig
    start: tuple

    def build_mdp(self):
        return GridMDP(
            self.grid,
            step_reward=self.step_reward,
            terminal_rewards=self.terminal_rewards,
            slip_prob=self.slip_prob,
            gamma=self.gamma,
        )


def load_scenario(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    grid = list(data["grid"])
    coords = GridConfig(
        cell_size=float(data["cell_size"]),
        origin_x=float(data["origin_x"]),
        origin_y=float(data["origin_y"]),
        n_rows=len(grid),
        n_cols=len(grid[0]),
    )
    return Scenario(
        grid=grid,
        step_reward=float(data["step_reward"]),
        terminal_rewards={k: float(v) for k, v in data["terminal_rewards"].items()},
        slip_prob=float(data["slip_prob"]),
        gamma=float(data["gamma"]),
        coords=coords,
        start=tuple(data["start"]),
    )
