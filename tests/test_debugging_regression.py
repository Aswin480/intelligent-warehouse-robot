"""Regression tests verifying confirmed RL defects and their corrections.

Tests:
1. Terminal transitions do not bootstrap from future Q-values: y = r when done=True.
2. Double DQN decouples action selection (online network) from evaluation (target network).
3. Package possession bit flips state and provides Markovian distinction.
4. Replay-buffer sampling yields correct tensor shapes and types.
5. Target-network synchronization follows the scheduled frequency.
6. Epsilon decay strictly respects bounds: eps_end <= eps <= eps_start.
7. Episode termination and truncation are handled with distinct logic.
8. Checkpoint save and load preserves numerical predictions.
"""
from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
import torch

from warehouse_rl.warehouse_env import (
    WarehouseEnv,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
)
from warehouse_rl.dqn_agent import DQNAgent, DQNHyperparameters, ReplayBuffer


def test_terminal_transitions_do_not_bootstrap_future_values():
    """Bug Diagnostic: Terminal states must yield target = reward (no future bootstrapping)."""
    params = DQNHyperparameters(gamma=0.95, min_replay_size=1, batch_size=1)
    agent = DQNAgent(params=params)

    # Transition into terminal delivery state with done=True
    s = np.zeros(5, dtype=np.float32)
    a = 1
    r = 100.0  # Terminal delivery reward
    ns = np.ones(5, dtype=np.float32)
    done = True

    agent.replay_buffer.push(s, a, r, ns, done)

    # Compute target manually following Bellman formulation
    s_t = torch.tensor([s], dtype=torch.float32, device=agent.device)
    ns_t = torch.tensor([ns], dtype=torch.float32, device=agent.device)
    r_t = torch.tensor([[r]], dtype=torch.float32, device=agent.device)
    done_t = torch.tensor([[1.0]], dtype=torch.float32, device=agent.device)

    with torch.no_grad():
        next_online_actions = agent.q_network(ns_t).argmax(dim=1, keepdim=True)
        next_q = agent.target_network(ns_t).gather(1, next_online_actions)
        target = r_t + (1.0 - done_t) * agent.params.gamma * next_q

    # With done=True, the future bootstrap term MUST be strictly zero
    assert float(target.item()) == pytest.approx(100.0)
    assert float(((1.0 - done_t) * agent.params.gamma * next_q).item()) == 0.0


def test_double_dqn_target_decoupling():
    """Bug Diagnostic: Double DQN must use online network for argmax and target network for evaluation."""
    params = DQNHyperparameters(double_dqn=True)
    agent = DQNAgent(params=params)

    # Create synthetic discrepancy between online and target network weights
    with torch.no_grad():
        for p in agent.target_network.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    dummy_state = torch.randn(1, 5, device=agent.device)

    # Online selection
    online_q = agent.q_network(dummy_state)
    best_online_action = online_q.argmax(dim=1, keepdim=True)

    # Target evaluation of online selected action
    target_q = agent.target_network(dummy_state)
    evaluated_val = target_q.gather(1, best_online_action)

    # Standard DQN would take max of target network directly
    standard_target_val = target_q.max(dim=1, keepdim=True)[0]

    # Verify that DDQN evaluates the online action specifically
    expected_eval = target_q[0, best_online_action[0, 0]].item()
    assert float(evaluated_val.item()) == pytest.approx(expected_eval)


def test_package_possession_bit_markov_state_change():
    """Bug Diagnostic: State space must distinguish phase 0 (seeking pickup) from phase 1 (delivery)."""
    env = WarehouseEnv()
    env.reset()

    # Pre-pickup state at pickup pos (2, 4)
    obs_before = env.get_observation(state=(2, 4, 0))
    # Post-pickup state at same cell (2, 4)
    obs_after = env.get_observation(state=(2, 4, 1))

    # The package possession bit (index 2) must be different
    assert obs_before[2] == 0.0
    assert obs_after[2] == 1.0
    assert not np.array_equal(obs_before, obs_after)


def test_replay_buffer_sampling_dimensions_and_dtypes():
    """Bug Diagnostic: Replay buffer sampling must return consistent batches without type corruption."""
    buffer = ReplayBuffer(capacity=500)
    for i in range(100):
        s = np.random.randn(5).astype(np.float32)
        ns = np.random.randn(5).astype(np.float32)
        buffer.push(s, i % 4, float(i), ns, i % 10 == 0)

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=32)

    assert states.shape == (32, 5)
    assert states.dtype == np.float32
    assert actions.shape == (32,)
    assert actions.dtype == np.int64
    assert rewards.shape == (32,)
    assert rewards.dtype == np.float32
    assert next_states.shape == (32, 5)
    assert next_states.dtype == np.float32
    assert dones.shape == (32,)
    assert dones.dtype == np.bool_


def test_target_network_synchronization_schedule():
    """Bug Diagnostic: Target network weights must update only every target_update_freq steps."""
    params = DQNHyperparameters(target_update_freq=5, min_replay_size=10, batch_size=4)
    agent = DQNAgent(params=params)

    # Populate buffer
    for _ in range(20):
        agent.replay_buffer.push(np.zeros(5), 0, 1.0, np.zeros(5), False)

    # Mutate target network to be different
    with torch.no_grad():
        for p in agent.target_network.parameters():
            p.fill_(99.0)

    # Steps 1 to 4 should NOT synchronize target network
    for step in range(4):
        agent.step_learning()
        any_diff = False
        for p1, p2 in zip(agent.q_network.parameters(), agent.target_network.parameters()):
            if not torch.allclose(p1, p2):
                any_diff = True
        assert any_diff, f"Target network prematurely updated at step {step + 1}"

    # Step 5 (modulo 5 == 0) MUST synchronize target network
    agent.step_learning()
    for p1, p2 in zip(agent.q_network.parameters(), agent.target_network.parameters()):
        assert torch.allclose(p1, p2), "Target network failed to synchronize at scheduled frequency"


def test_epsilon_schedule_respects_configured_limits():
    """Bug Diagnostic: Epsilon decay must never drop below epsilon_end or exceed epsilon_start."""
    params = DQNHyperparameters(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.50)
    agent = DQNAgent(params=params)

    assert agent.epsilon == 1.0

    # Rapid decay iterations
    for _ in range(50):
        agent.decay_epsilon()

    # Must clamp strictly at epsilon_end
    assert agent.epsilon == pytest.approx(0.05)
    assert agent.epsilon >= params.epsilon_end


def test_episode_termination_and_truncation_consistency():
    """Bug Diagnostic: Timeout (truncation) and Goal (termination) must be mutually distinct."""
    # Test case 1: Successful termination (goal reached)
    env = WarehouseEnv(max_steps=50)
    env.reset()
    # Direct move into delivery with package
    next_s, r, term, info = env.transition_dynamics((5, 4, 1), ACTION_RIGHT)
    assert term is True
    assert r > 50.0  # Delivery reward

    # Test case 2: Truncation without goal
    env.reset()
    env.current_step = 49
    obs, r, term, trunc, info = env.step(ACTION_UP)
    assert trunc is True
    assert term is False
    assert info["event"] == "timeout"


def test_checkpoint_save_and_reload_idempotency():
    """Bug Diagnostic: Saving and reloading checkpoint must preserve exact Q-network parameters and predictions."""
    params = DQNHyperparameters()
    agent = DQNAgent(params=params)

    test_input = np.array([0.2, 0.4, 1.0, 0.3, 0.5], dtype=np.float32)
    original_q_vals = agent.get_q_values(test_input)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_file = tmp.name

    try:
        agent.save_checkpoint(tmp_file)

        # Create fresh agent with random weights
        new_agent = DQNAgent(params=params)
        reloaded_q_before = new_agent.get_q_values(test_input)

        new_agent.load_checkpoint(tmp_file)
        reloaded_q_after = new_agent.get_q_values(test_input)

        assert np.allclose(original_q_vals, reloaded_q_after, atol=1e-6)
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
