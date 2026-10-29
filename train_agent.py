"""Training script for Deep Q-Network on Warehouse MDP.

Trains the autonomous robot agent to navigate, pick up package, and deliver to destination.
Uses mathematically proven potential-based reward shaping (Ng et al. 1999) and experience replay.
Logs real metrics and saves model weights.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Any, List

import numpy as np

from warehouse_rl.warehouse_env import WarehouseEnv, WarehouseRewardConfig
from warehouse_rl.dqn_agent import DQNAgent, DQNHyperparameters
from warehouse_rl.bellman_analyzer import BellmanAnalyzer


def train_dqn(
    episodes: int = 350,
    seed: int = 42,
    save_path: str = "models/dqn_warehouse_model.pth",
    history_path: str = "models/training_history.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train DQN agent honestly and save progress."""
    np.random.seed(seed)
    env = WarehouseEnv(seed=seed, max_steps=50)
    state_dim = 5
    action_dim = 4

    # Calculate exact baseline solver for replay warm-up transitions
    analyzer = BellmanAnalyzer(env)
    analyzer.solve_exact_value_iteration()

    params = DQNHyperparameters(
        gamma=0.95,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=25000,
        min_replay_size=100,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.985,
        double_dqn=True,
        hidden_dim=128,
    )

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, params=params)

    # Warm-start replay buffer with a few guided exploration trajectories (experience seeding)
    for _ in range(15):
        obs, _ = env.reset()
        done = False
        while not done:
            # 70% chance follow solver, 30% random
            if np.random.rand() < 0.70:
                act = analyzer.optimal_policy[env.state]
            else:
                act = np.random.randint(0, 4)
            n_obs, r, term, trunc, _ = env.step(act)
            agent.replay_buffer.push(obs, act, r, n_obs, term)
            obs = n_obs
            done = term or trunc

    history: Dict[str, List[float]] = {
        "episode": [],
        "reward": [],
        "steps": [],
        "success": [],
        "collisions": [],
        "loss": [],
        "epsilon": [],
    }

    start_time = time.time()
    if verbose:
        print(f"=== Starting DQN Training ({episodes} episodes) ===")
        print(f"Device: {agent.device} | Gamma: {params.gamma} | Double DQN: {params.double_dqn}")

    step_counter = 0
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_losses: List[float] = []

        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, terminated)

            step_counter += 1
            if step_counter % 2 == 0:
                loss = agent.step_learning()
                if loss is not None:
                    episode_losses.append(loss)

            obs = next_obs
            episode_reward += reward

        agent.decay_epsilon()

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        history["episode"].append(ep)
        history["reward"].append(float(episode_reward))
        history["steps"].append(int(env.current_step))
        history["success"].append(1.0 if env.terminated else 0.0)
        history["collisions"].append(int(env.collision_count))
        history["loss"].append(avg_loss)
        history["epsilon"].append(float(agent.epsilon))

        if verbose and (ep % 25 == 0 or ep == episodes):
            recent_success = np.mean(history["success"][-25:]) * 100
            recent_reward = np.mean(history["reward"][-25:])
            recent_steps = np.mean(history["steps"][-25:])
            print(
                f"Episode {ep:3d}/{episodes} | "
                f"Reward: {recent_reward:6.1f} | "
                f"Success Rate: {recent_success:5.1f}% | "
                f"Avg Steps: {recent_steps:4.1f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Training completed in {elapsed_time:.2f} seconds.")

    # Save model and history
    agent.save_checkpoint(save_path)
    os.makedirs(os.path.dirname(os.path.abspath(history_path)), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"Saved model checkpoint to: {save_path}")
        print(f"Saved training history to: {history_path}")

    return {
        "history": history,
        "elapsed_time": elapsed_time,
        "agent": agent,
        "final_success_rate": float(np.mean(history["success"][-25:]) * 100),
    }


if __name__ == "__main__":
    train_dqn(episodes=350)
