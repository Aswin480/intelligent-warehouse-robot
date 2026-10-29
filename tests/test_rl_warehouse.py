"""Unit tests for Warehouse RL Environment, Bellman Analyzer, and DQN Agent."""
import os
import tempfile
import numpy as np
import pytest
import torch

from warehouse_rl.warehouse_env import (
    WarehouseEnv,
    WarehouseRewardConfig,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_NAMES,
)
from warehouse_rl.dqn_agent import DQNAgent, DQNHyperparameters, ReplayBuffer
from warehouse_rl.bellman_analyzer import BellmanAnalyzer


def test_warehouse_env_initialization_and_states():
    env = WarehouseEnv()
    assert env.rows == 6
    assert env.cols == 6
    assert env.start_pos == (0, 0)
    assert env.pickup_pos == (2, 4)
    assert env.delivery_pos == (5, 5)
    # State space must consist of free cells * 2 (p=0 and p=1)
    assert env.num_states == len(env.free_cells) * 2
    assert env.num_actions == 4
    assert env.state == (0, 0, 0)


def test_boundary_and_obstacle_collisions():
    env = WarehouseEnv()
    env.reset()

    # Move UP from (0, 0) -> boundary collision
    obs, reward, terminated, truncated, info = env.step(ACTION_UP)
    assert env.robot_pos == (0, 0)
    assert reward == env.rewards.boundary_penalty
    assert info["event"] == "boundary_collision"
    assert env.collision_count == 1
    assert not terminated

    # Move LEFT from (0, 0) -> boundary collision
    obs, reward, terminated, truncated, info = env.step(ACTION_LEFT)
    assert env.robot_pos == (0, 0)
    assert reward == env.rewards.boundary_penalty
    assert env.collision_count == 2

    # Move RIGHT into obstacle cell (0, 4) is # in default grid:
    # Let's test transition_dynamics directly on an obstacle boundary
    next_s, r, term, event_info = env.transition_dynamics((1, 0, 0), ACTION_RIGHT)
    # In default grid: (1, 1) is #
    assert next_s == (1, 0, 0)
    assert r == env.rewards.obstacle_penalty
    assert event_info["event"] == "obstacle_collision"


def test_package_pickup_and_delivery_transitions():
    env = WarehouseEnv()
    env.reset()

    # Move adjacent to pickup pos (2, 4): e.g. from (1, 4) downwards into (2, 4)
    next_s, r, term, event_info = env.transition_dynamics((1, 4, 0), ACTION_DOWN)
    assert next_s == (2, 4, 1)
    assert r == env.rewards.step_penalty + env.rewards.pickup_reward
    assert event_info["event"] == "pickup"
    assert not term

    # Visiting delivery pos without package (p=0) should NOT deliver
    next_s, r, term, event_info = env.transition_dynamics((4, 5, 0), ACTION_DOWN)
    assert next_s == (5, 5, 0)
    assert r == env.rewards.step_penalty
    assert not term
    assert event_info["event"] == "move"

    # Visiting delivery pos with package (p=1) MUST deliver successfully
    next_s, r, term, event_info = env.transition_dynamics((4, 5, 1), ACTION_DOWN)
    assert next_s == (5, 5, 1)
    assert r == env.rewards.step_penalty + env.rewards.delivery_reward
    assert term is True
    assert event_info["event"] == "delivery"


def test_timeout_and_max_steps():
    env = WarehouseEnv(max_steps=5)
    env.reset()
    for _ in range(4):
        obs, r, term, trunc, info = env.step(ACTION_RIGHT)
        assert not trunc

    # 5th step should trigger truncation
    obs, r, term, trunc, info = env.step(ACTION_RIGHT)
    assert trunc is True
    assert r < 0  # Includes timeout penalty
    assert info["event"] == "timeout"


def test_observation_shape_and_normalization():
    env = WarehouseEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    assert all(0.0 <= val <= 1.0 for val in obs)


def test_bellman_analyzer_exact_solution():
    env = WarehouseEnv()
    analyzer = BellmanAnalyzer(env)
    result = analyzer.solve_exact_value_iteration(theta=1e-4)

    assert result["converged"] is True
    assert analyzer.is_solved is True
    assert len(analyzer.V_star) == env.num_states
    assert len(analyzer.Q_star) == env.num_states * 4

    # Terminal delivery state has value 0
    delivery_state = (env.delivery_pos[0], env.delivery_pos[1], 1)
    assert analyzer.V_star[delivery_state] == 0.0

    # Start state should have positive expected optimal return
    assert analyzer.V_star[(0, 0, 0)] > 0.0


def test_bellman_numerical_comparison():
    env = WarehouseEnv()
    analyzer = BellmanAnalyzer(env)
    analyzer.solve_exact_value_iteration()

    # Compare optimal move RIGHT vs invalid boundary move UP at (0, 0, 0)
    # Action RIGHT moves along the hallway towards pickup; Action UP hits boundary
    breakdown = analyzer.compare_actions_numerical_example(
        state=(0, 0, 0),
        action_a=ACTION_RIGHT,
        action_b=ACTION_UP,
    )

    res_a = breakdown["action_A"]
    res_b = breakdown["action_B"]

    # Verify Bellman equation equality: Q = R + gamma * V
    assert np.isclose(res_a["action_value_Q"], res_a["immediate_reward"] + res_a["gamma"] * res_a["next_state_value_V"])
    assert np.isclose(res_b["action_value_Q"], res_b["immediate_reward"] + res_b["gamma"] * res_b["next_state_value_V"])
    assert breakdown["advantage_A_over_B"] > 0
    assert breakdown["best_action_between_two"] == "RIGHT"


def test_dqn_replay_buffer_and_learning_step():
    buffer = ReplayBuffer(capacity=100)
    for i in range(50):
        buffer.push(np.zeros(5), 0, 1.0, np.ones(5), False)
    assert len(buffer) == 50

    s, a, r, ns, d = buffer.sample(16)
    assert s.shape == (16, 5)
    assert a.shape == (16,)
    assert r.shape == (16,)
    assert ns.shape == (16, 5)
    assert d.shape == (16,)


def test_dqn_agent_step_and_checkpoint():
    params = DQNHyperparameters(min_replay_size=20, batch_size=16)
    agent = DQNAgent(state_dim=5, action_dim=4, params=params)

    obs = np.random.randn(5).astype(np.float32)
    action = agent.select_action(obs, evaluate=True)
    assert action in (0, 1, 2, 3)

    q_vals = agent.get_q_values(obs)
    assert q_vals.shape == (4,)

    # Fill buffer to enable learning step
    for _ in range(30):
        agent.replay_buffer.push(obs, 1, 1.0, obs, False)

    loss = agent.step_learning()
    assert loss is not None
    assert loss >= 0.0

    # Test saving and loading checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        agent.save_checkpoint(tmp_path)
        assert os.path.exists(tmp_path)

        new_agent = DQNAgent(state_dim=5, action_dim=4, params=params)
        new_agent.load_checkpoint(tmp_path)

        # Check weights equivalence
        for p1, p2 in zip(agent.q_network.parameters(), new_agent.q_network.parameters()):
            assert torch.allclose(p1, p2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
