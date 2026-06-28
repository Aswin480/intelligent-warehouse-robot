"""Build a tabular finite MDP from a text grid."""
from __future__ import annotations

import numpy as np

ACTIONS = ("N", "S", "E", "W")
_DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
_PERP = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}


class GridMDP:
    def __init__(self, grid, *, step_reward=-1.0, terminal_rewards=None,
                 slip_prob=0.1, gamma=0.95):
        if terminal_rewards is None:
            terminal_rewards = {"G": 1.0}
        self.grid = list(grid)
        self.shape = (len(self.grid), len(self.grid[0]))
        if any(len(r) != self.shape[1] for r in self.grid):
            raise ValueError(
                f"grid rows must all have equal length; "
                f"got {[len(r) for r in self.grid]}"
            )
        self.gamma = float(gamma)
        self.n_actions = len(ACTIONS)
        self._terminal_rewards = dict(terminal_rewards)
        self._step_reward = float(step_reward)
        self._slip_prob = float(slip_prob)

        rows, cols = self.shape
        # States = every non-wall cell.
        self.states = [(r, c) for r in range(rows) for c in range(cols)
                       if self.grid[r][c] != "#"]
        self.index = {rc: i for i, rc in enumerate(self.states)}
        self.n_states = len(self.states)
        self.terminal = np.array(
            [self.grid[r][c] in terminal_rewards for (r, c) in self.states],
            dtype=bool,
        )

        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))
        self._build()

    def _reward_of(self, rc):
        # Reward-on-entering convention: r(s,a)=E[reward(s')]. Matches Sutton's constant-reward gridworlds exactly;
        # for Russell-Norvig it yields the same optimal policy as the reward-on-state convention but different V* magnitudes.
        r, c = rc
        ch = self.grid[r][c]
        return self._terminal_rewards.get(ch, self._step_reward)

    def _resolve(self, rc, direction):
        """Where you land moving `direction` from `rc` (stay if wall/off-grid)."""
        rows, cols = self.shape
        dr, dc = _DELTA[direction]
        nr, nc = rc[0] + dr, rc[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols and self.grid[nr][nc] != "#":
            return (nr, nc)
        return rc

    def _build(self):
        slip = self._slip_prob
        for s_idx, rc in enumerate(self.states):
            if self.terminal[s_idx]:
                self.T[s_idx, :, s_idx] = 1.0  # absorbing, zero reward
                continue
            for a_idx, action in enumerate(ACTIONS):
                outcomes = [(1.0 - 2 * slip, action)]
                outcomes += [(slip, p) for p in _PERP[action]]
                for prob, direction in outcomes:
                    if prob <= 0.0:
                        continue
                    nxt = self._resolve(rc, direction)
                    j = self.index[nxt]
                    self.T[s_idx, a_idx, j] += prob
                    self.R[s_idx, a_idx] += prob * self._reward_of(nxt)

    def grid_values(self, V):
        rows, cols = self.shape
        out = np.full((rows, cols), np.nan)
        for i, (r, c) in enumerate(self.states):
            out[r, c] = V[i]
        return out
