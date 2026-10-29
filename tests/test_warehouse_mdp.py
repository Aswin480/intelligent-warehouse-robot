import pytest

from mdp import WarehouseMDP


def test_state_count_excludes_obstacles():
    mdp = WarehouseMDP(["S#.", "..G"])

    assert mdp.state_count == 5
    assert (0, 1) not in mdp.index


def test_transitions_distinguish_valid_goal_obstacle_and_boundary_moves():
    mdp = WarehouseMDP(["S#.", "..G"])

    assert mdp.transition((0, 0), "RIGHT") == ((0, 0), -10.0)
    assert mdp.transition((0, 0), "UP") == ((0, 0), -5.0)
    assert mdp.transition((0, 0), "DOWN") == ((1, 0), -1.0)
    assert mdp.transition((1, 1), "RIGHT") == ((1, 2), 100.0)


def test_goal_is_terminal_and_absorbing():
    mdp = WarehouseMDP(["S.", ".G"])

    assert mdp.is_terminal(mdp.goal)
    assert mdp.transition(mdp.goal, "UP") == (mdp.goal, 0.0)


def test_invalid_grid_is_rejected():
    with pytest.raises(ValueError):
        WarehouseMDP(["S..", "..#"])