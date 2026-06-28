import numpy as np
import pytest
from gridworld.mdp import GridMDP, ACTIONS


def test_states_exclude_walls():
    mdp = GridMDP([".#", ".G"], terminal_rewards={"G": 1.0})
    # 4 cells, one wall -> 3 states
    assert mdp.n_states == 3
    assert (0, 1) not in mdp.index  # the wall is not a state


def test_transition_rows_are_distributions():
    mdp = GridMDP(["...", ".#.", "..G"], terminal_rewards={"G": 1.0}, slip_prob=0.1)
    sums = mdp.T.sum(axis=2)  # (n_states, n_actions)
    assert np.allclose(sums, 1.0)


def test_bump_wall_stays_in_place():
    # Deterministic: from (0,0) moving N (off-grid) must stay at (0,0).
    mdp = GridMDP(["..", "..", ".G"], terminal_rewards={"G": 1.0}, slip_prob=0.0)
    s = mdp.index[(0, 0)]
    n = ACTIONS.index("N")
    assert mdp.T[s, n, s] == 1.0


def test_terminals_are_absorbing_with_zero_reward():
    mdp = GridMDP([".G"], terminal_rewards={"G": 1.0})
    g = mdp.index[(0, 1)]
    assert mdp.terminal[g]
    assert np.allclose(mdp.T[g, :, g], 1.0)
    assert np.allclose(mdp.R[g, :], 0.0)


def test_expected_reward_on_entering_goal():
    # Deterministic move E from (0,0) into goal (0,1) yields the goal reward.
    mdp = GridMDP([".G"], terminal_rewards={"G": 5.0}, step_reward=-1.0, slip_prob=0.0)
    s = mdp.index[(0, 0)]
    e = ACTIONS.index("E")
    assert np.isclose(mdp.R[s, e], 5.0)


def test_grid_values_places_values_and_nans_walls():
    mdp = GridMDP([".#", ".G"], terminal_rewards={"G": 1.0})
    V = np.arange(mdp.n_states, dtype=float)
    grid = mdp.grid_values(V)
    assert np.isnan(grid[0, 1])  # the wall is NaN
    assert grid[0, 0] == V[mdp.index[(0, 0)]]  # non-wall cell holds its value


def test_jagged_grid_rejected():
    with pytest.raises(ValueError):
        GridMDP(["...", ".."])
