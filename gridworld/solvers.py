"""Dynamic Programming solvers (Sutton & Barto Ch 4)."""
from __future__ import annotations

import numpy as np


def q_from_v(mdp, V):
    # Q[s,a] = R[s,a] + gamma * sum_s' T[s,a,s'] V[s']
    return mdp.R + mdp.gamma * (mdp.T @ V)


def policy_evaluation(mdp, policy, *, theta=1e-6, max_sweeps=100000):
    V = np.zeros(mdp.n_states)
    for _ in range(max_sweeps):
        Q = q_from_v(mdp, V)
        new_V = np.einsum("sa,sa->s", policy, Q)
        new_V[mdp.terminal] = 0.0
        delta = np.max(np.abs(new_V - V))
        V = new_V
        if delta < theta:
            break
    return V


def greedy_policy(mdp, V):
    Q = q_from_v(mdp, V)
    best = np.argmax(Q, axis=1)
    policy = np.zeros_like(Q)
    policy[np.arange(mdp.n_states), best] = 1.0
    return policy


def policy_actions(policy):
    return np.argmax(policy, axis=1)


def value_iteration(mdp, *, theta=1e-6, max_sweeps=100000):
    V = np.zeros(mdp.n_states)
    history = {"V": [], "delta": []}
    for _ in range(max_sweeps):
        Q = q_from_v(mdp, V)
        new_V = np.max(Q, axis=1)
        new_V[mdp.terminal] = 0.0
        delta = float(np.max(np.abs(new_V - V)))
        V = new_V
        history["V"].append(V.copy())
        history["delta"].append(delta)
        if delta < theta:
            break
    return V, greedy_policy(mdp, V), history


def policy_iteration(mdp, *, eval_theta=1e-8, max_iters=1000):
    """Policy iteration: alternate full evaluation and greedy improvement until stable (Sutton & Barto §4.3)."""
    policy = np.ones((mdp.n_states, mdp.n_actions)) / mdp.n_actions
    history = {"V": [], "stable_at": 0}
    for i in range(1, max_iters + 1):
        V = policy_evaluation(mdp, policy, theta=eval_theta)
        history["V"].append(V.copy())
        new_policy = greedy_policy(mdp, V)
        if np.array_equal(policy_actions(new_policy), policy_actions(policy)):
            history["stable_at"] = i
            return V, new_policy, history
        policy = new_policy
    history["stable_at"] = max_iters
    return V, policy, history
