import numpy as np
from gridworld.mdp import GridMDP, ACTIONS
from gridworld.solvers import policy_evaluation, value_iteration, greedy_policy, policy_actions, policy_iteration


def _sutton_4x4():
    # Terminals at top-left (0,0) and bottom-right (3,3); reward -1 on every
    # transition; deterministic; undiscounted. (Sutton & Barto Example 4.1.)
    grid = ["T...", "....", "....", "...T"]
    return GridMDP(grid, step_reward=-1.0, terminal_rewards={"T": -1.0},
                   slip_prob=0.0, gamma=1.0)


def test_policy_evaluation_matches_sutton_fig_4_1():
    mdp = _sutton_4x4()
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions  # uniform random
    V = policy_evaluation(mdp, policy, theta=1e-8)
    grid_V = mdp.grid_values(V)
    expected = np.array([
        [0.0, -14.0, -20.0, -22.0],
        [-14.0, -18.0, -20.0, -20.0],
        [-20.0, -20.0, -18.0, -14.0],
        [-22.0, -20.0, -14.0, 0.0],
    ])
    assert np.allclose(grid_V, expected, atol=1e-1)


def _manhattan_to_nearest_terminal(rc, terminals):
    return min(abs(rc[0] - t[0]) + abs(rc[1] - t[1]) for t in terminals)


def test_value_iteration_deterministic_is_negative_manhattan():
    mdp = _sutton_4x4()
    V, policy, history = value_iteration(mdp, theta=1e-9)
    terminals = [(0, 0), (3, 3)]
    for i, rc in enumerate(mdp.states):
        expected = -_manhattan_to_nearest_terminal(rc, terminals)
        assert np.isclose(V[i], expected, atol=1e-3), (rc, V[i], expected)
    assert len(history["V"]) >= 1
    assert np.allclose(history["V"][-1], V)


def _russell_norvig():
    # row 0 at top. G=+1 at (0,3), P=-1 at (1,3), wall at (1,1). slip 0.1 each.
    grid = ["...G", ".#.P", "...."]
    return GridMDP(grid, step_reward=-0.04,
                   terminal_rewards={"G": 1.0, "P": -1.0},
                   slip_prob=0.1, gamma=1.0)


def test_value_iteration_drives_into_goal_from_left_of_it():
    mdp = _russell_norvig()
    V, policy, _ = value_iteration(mdp, theta=1e-9)
    actions = policy_actions(policy)
    # (0,2) sits directly left of the +1 terminal: moving E is strictly optimal.
    assert actions[mdp.index[(0, 2)]] == ACTIONS.index("E")


def test_pi_matches_vi_deterministic():
    mdp = _sutton_4x4()
    V_vi, pol_vi, _ = value_iteration(mdp, theta=1e-9)
    V_pi, pol_pi, history = policy_iteration(mdp)
    assert np.allclose(V_vi, V_pi, atol=1e-3)
    assert np.array_equal(policy_actions(pol_vi), policy_actions(pol_pi))
    assert history["stable_at"] >= 1


def test_pi_matches_vi_stochastic():
    mdp = _russell_norvig()
    V_vi, _, _ = value_iteration(mdp, theta=1e-9)
    V_pi, pol_pi, _ = policy_iteration(mdp)
    assert np.allclose(V_vi, V_pi, atol=1e-3)
    # Greedy policy derived from each converged V must agree.
    a_vi = policy_actions(greedy_policy(mdp, V_vi))
    a_pi = policy_actions(pol_pi)
    assert np.array_equal(a_vi[~mdp.terminal], a_pi[~mdp.terminal])
