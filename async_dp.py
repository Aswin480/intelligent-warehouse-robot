"""Synchronous and in-place asynchronous dynamic programming."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

from mdp import Action, State, WarehouseMDP


@dataclass(frozen=True)
class SolverResult:
    algorithm: str
    values: dict[State, float]
    policy: dict[State, Action | None]
    iterations: int
    state_updates: int
    max_delta: float
    converged: bool
    execution_time_ms: float

    def as_dict(self) -> dict:
        result = asdict(self)
        result["values"] = {f"{row},{column}": value for (row, column), value in self.values.items()}
        result["policy"] = {
            f"{row},{column}": action
            for (row, column), action in self.policy.items()
        }
        return result


def _action_value(mdp: WarehouseMDP, values: dict[State, float], state: State, action: Action) -> float:
    next_state, reward = mdp.transition(state, action)
    return reward + mdp.gamma * values[next_state]


def _greedy_policy(mdp: WarehouseMDP, values: dict[State, float]) -> dict[State, Action | None]:
    policy: dict[State, Action | None] = {}
    for state in mdp.states:
        if mdp.is_terminal(state):
            policy[state] = None
            continue
        policy[state] = max(
            mdp.ACTIONS,
            key=lambda action: _action_value(mdp, values, state, action),
        )
    return policy


def _initial_values(mdp: WarehouseMDP) -> dict[State, float]:
    return {state: 0.0 for state in mdp.states}


def synchronous_value_iteration(
    mdp: WarehouseMDP,
    *,
    theta: float = 0.001,
    max_iterations: int = 1000,
) -> SolverResult:
    """Use a copied value map for each full Bellman sweep as a baseline."""
    if theta <= 0 or max_iterations < 1:
        raise ValueError("theta must be positive and max_iterations must be at least 1")
    values = _initial_values(mdp)
    state_updates = 0
    max_delta = float("inf")
    converged = False
    started = perf_counter()
    for iteration in range(1, max_iterations + 1):
        new_values = dict(values)
        max_delta = 0.0
        for state in mdp.states:
            if mdp.is_terminal(state):
                new_values[state] = 0.0
                continue
            best_value = max(_action_value(mdp, values, state, action) for action in mdp.ACTIONS)
            new_values[state] = best_value
            max_delta = max(max_delta, abs(best_value - values[state]))
            state_updates += 1
        values = new_values
        if max_delta < theta:
            converged = True
            break
    elapsed = (perf_counter() - started) * 1000
    return SolverResult(
        "Synchronous Value Iteration", values, _greedy_policy(mdp, values),
        iteration, state_updates, max_delta, converged, elapsed,
    )


def asynchronous_value_iteration(
    mdp: WarehouseMDP,
    *,
    theta: float = 0.001,
    max_iterations: int = 1000,
) -> SolverResult:
    """Perform in-place row-major Bellman updates.

    ``values`` is updated immediately after every state, so later states in
    the same sweep can use newer information. This is asynchronous in the
    dynamic-programming sense, not Python async/await or concurrent execution.
    """
    if theta <= 0 or max_iterations < 1:
        raise ValueError("theta must be positive and max_iterations must be at least 1")
    values = _initial_values(mdp)
    state_updates = 0
    max_delta = float("inf")
    converged = False
    started = perf_counter()
    for iteration in range(1, max_iterations + 1):
        max_delta = 0.0
        for state in mdp.states:
            if mdp.is_terminal(state):
                values[state] = 0.0
                continue
            old_value = values[state]
            new_value = max(_action_value(mdp, values, state, action) for action in mdp.ACTIONS)
            values[state] = new_value
            max_delta = max(max_delta, abs(new_value - old_value))
            state_updates += 1
        if max_delta < theta:
            converged = True
            break
    elapsed = (perf_counter() - started) * 1000
    return SolverResult(
        "Asynchronous Value Iteration", values, _greedy_policy(mdp, values),
        iteration, state_updates, max_delta, converged, elapsed,
    )


def generate_path(
    mdp: WarehouseMDP,
    policy: dict[State, Action | None],
    *,
    max_steps: int = 200,
) -> dict:
    """Follow a greedy policy, stopping at the goal or a detected cycle."""
    state = mdp.start
    path = [state]
    visited = {state}
    total_reward = 0.0
    cycle_detected = False
    for _ in range(max_steps):
        if mdp.is_terminal(state):
            break
        action = policy.get(state)
        if action is None:
            break
        next_state, reward = mdp.transition(state, action)
        total_reward += reward
        path.append(next_state)
        if next_state in visited:
            cycle_detected = True
            break
        visited.add(next_state)
        state = next_state
    return {
        "path": [[row, column] for row, column in path],
        "path_length": max(0, len(path) - 1),
        "total_reward": total_reward,
        "goal_reached": mdp.is_terminal(path[-1]),
        "cycle_detected": cycle_detected,
    }


def compare_algorithms(mdp: WarehouseMDP) -> dict[str, SolverResult]:
    return {
        "synchronous": synchronous_value_iteration(mdp),
        "asynchronous": asynchronous_value_iteration(mdp),
    }