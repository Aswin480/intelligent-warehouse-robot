"""Bellman Equation and State-Value Analysis Module.

Provides exact tabular dynamic programming solutions (exact V* and Q*),
theoretical Bellman optimality and expectation analyses, and worked
numerical step-by-step comparisons between different actions.
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Any, Optional
import numpy as np

from warehouse_rl.warehouse_env import (
    WarehouseEnv,
    ACTION_NAMES,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
)


class BellmanAnalyzer:
    """Analytical solver and Bellman equation breakdown for Warehouse MDP."""

    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.gamma = env.gamma
        self.states = env.states
        self.num_states = env.num_states
        self.state_to_idx = env.state_to_idx
        self.idx_to_state = env.idx_to_state
        self.actions = env.ACTIONS

        # Tabular storage
        self.V_star: Dict[Tuple[int, int, int], float] = {}
        self.Q_star: Dict[Tuple[Tuple[int, int, int], int], float] = {}
        self.optimal_policy: Dict[Tuple[int, int, int], int] = {}
        self.is_solved = False

    def solve_exact_value_iteration(
        self, theta: float = 1e-5, max_iterations: int = 1000
    ) -> Dict[str, Any]:
        """Compute exact V*(s) and Q*(s, a) using Bellman Optimality Value Iteration:

        V_{k+1}(s) = max_a [ R(s, a) + gamma * sum_{s'} P(s'|s, a) * V_k(s') ]
        """
        V = {s: 0.0 for s in self.states}

        iterations = 0
        max_delta = float("inf")

        for iteration in range(1, max_iterations + 1):
            delta = 0.0
            new_V = dict(V)

            for s in self.states:
                # Terminal condition: package delivered
                if s[0] == self.env.delivery_pos[0] and s[1] == self.env.delivery_pos[1] and s[2] == 1:
                    new_V[s] = 0.0
                    continue

                best_val = -float("inf")
                for a in self.actions:
                    next_s, reward, is_term, _ = self.env.transition_dynamics(s, a)
                    next_v = 0.0 if is_term else V[next_s]
                    q_val = reward + self.gamma * next_v
                    if q_val > best_val:
                        best_val = q_val

                new_V[s] = best_val
                delta = max(delta, abs(new_V[s] - V[s]))

            V = new_V
            iterations = iteration
            if delta < theta:
                max_delta = delta
                break
            max_delta = delta

        self.V_star = V

        # Derive exact Q*(s, a) and optimal policy pi*(s)
        self.Q_star = {}
        self.optimal_policy = {}

        for s in self.states:
            if s[0] == self.env.delivery_pos[0] and s[1] == self.env.delivery_pos[1] and s[2] == 1:
                self.optimal_policy[s] = ACTION_UP
                for a in self.actions:
                    self.Q_star[(s, a)] = 0.0
                continue

            best_a = self.actions[0]
            best_q = -float("inf")
            for a in self.actions:
                next_s, reward, is_term, _ = self.env.transition_dynamics(s, a)
                next_v = 0.0 if is_term else self.V_star[next_s]
                q_val = reward + self.gamma * next_v
                self.Q_star[(s, a)] = q_val
                if q_val > best_q:
                    best_q = q_val
                    best_a = a

            self.optimal_policy[s] = best_a

        self.is_solved = True
        return {
            "converged": max_delta < theta,
            "iterations": iterations,
            "max_delta": max_delta,
            "num_states": self.num_states,
        }

    def compute_action_breakdown(
        self, state: Tuple[int, int, int], action: int
    ) -> Dict[str, Any]:
        """Compute the detailed Bellman components for taking action `a` in state `s`:

        Q*(s, a) = R(s, a) + gamma * V*(s')
        """
        if not self.is_solved:
            self.solve_exact_value_iteration()

        next_state, reward, is_terminal, event_info = self.env.transition_dynamics(state, action)
        next_v = 0.0 if is_terminal else self.V_star[next_state]
        discounted_future = self.gamma * next_v
        total_q = reward + discounted_future

        return {
            "state": state,
            "action_id": action,
            "action_name": ACTION_NAMES[action],
            "immediate_reward": float(reward),
            "next_state": next_state,
            "is_terminal": is_terminal,
            "event": event_info["event"],
            "gamma": float(self.gamma),
            "next_state_value_V": float(next_v),
            "discounted_future": float(discounted_future),
            "action_value_Q": float(total_q),
        }

    def compare_actions_numerical_example(
        self,
        state: Tuple[int, int, int],
        action_a: int,
        action_b: int,
        nn_agent_q_values: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Provide a side-by-side numerical Bellman calculation comparing two candidate actions."""
        res_a = self.compute_action_breakdown(state, action_a)
        res_b = self.compute_action_breakdown(state, action_b)

        comparison = {
            "state": state,
            "action_A": res_a,
            "action_B": res_b,
            "advantage_A_over_B": res_a["action_value_Q"] - res_b["action_value_Q"],
            "best_action_between_two": res_a["action_name"] if res_a["action_value_Q"] >= res_b["action_value_Q"] else res_b["action_name"],
        }

        if nn_agent_q_values is not None:
            comparison["nn_predictions"] = {
                "Q_pred_A": float(nn_agent_q_values[action_a]),
                "Q_pred_B": float(nn_agent_q_values[action_b]),
                "nn_error_A": abs(float(nn_agent_q_values[action_a]) - res_a["action_value_Q"]),
                "nn_error_B": abs(float(nn_agent_q_values[action_b]) - res_b["action_value_Q"]),
            }

        return comparison
