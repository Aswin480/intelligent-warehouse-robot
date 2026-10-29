import math

import pytest

from async_dp import asynchronous_value_iteration, generate_path, synchronous_value_iteration
from mdp import WarehouseMDP


def test_async_value_iteration_converges_with_finite_values():
    result = asynchronous_value_iteration(WarehouseMDP(["S..", ".#.", "..G"]))

    assert result.converged
    assert result.state_updates > 0
    assert all(math.isfinite(value) for value in result.values.values())


def test_policy_and_path_reach_goal():
    mdp = WarehouseMDP(["S..", ".#.", "..G"])
    result = asynchronous_value_iteration(mdp)
    path = generate_path(mdp, result.policy)

    assert result.policy[mdp.start] in mdp.ACTIONS
    assert path["goal_reached"]
    assert path["path_length"] > 0


def test_sync_and_async_solve_same_environment():
    mdp = WarehouseMDP(["S..", ".#.", "..G"])
    sync = synchronous_value_iteration(mdp)
    async_result = asynchronous_value_iteration(mdp)

    assert sync.converged and async_result.converged
    assert generate_path(mdp, sync.policy)["goal_reached"]
    assert generate_path(mdp, async_result.policy)["goal_reached"]


def test_solver_rejects_invalid_limits():
    mdp = WarehouseMDP(["S.G"])

    with pytest.raises(ValueError):
        asynchronous_value_iteration(mdp, max_iterations=0)
    with pytest.raises(ValueError):
        synchronous_value_iteration(mdp, theta=0)