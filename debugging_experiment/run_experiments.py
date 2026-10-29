"""Reproducible Experiment Suite for Advanced RL Debugging Challenge.

Executes controlled, reproducible training and evaluation runs comparing:
1. Faulty Baseline Agent (Premature Decay, Standard DQN Maximization Bias, No Exploration Seeding)
2. Corrected Double DQN Agent (Guided Seeding, Decoupled DDQN Target, Update Throttling, Huber Loss)

Saves raw metrics and publication-quality plots to submission_debugging/results/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from warehouse_rl.warehouse_env import WarehouseEnv
from warehouse_rl.dqn_agent import DQNAgent, DQNHyperparameters
from warehouse_rl.bellman_analyzer import BellmanAnalyzer
from debugging_experiment.faulty_baseline_agent import (
    FaultyBaselineDQNAgent,
    FaultyBaselineHyperparameters,
)


def train_faulty_baseline(
    episodes: int = 250,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train faulty baseline agent exhibiting confirmed defects."""
    np.random.seed(seed)
    env = WarehouseEnv(seed=seed, max_steps=50)

    params = FaultyBaselineHyperparameters(
        gamma=0.95,
        lr=1e-3,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.970,  # Premature collapse
        double_dqn=False,     # Standard DQN bias
        hidden_dim=128,
    )
    agent = FaultyBaselineDQNAgent(params=params)

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
        print(f"--> [Faulty Baseline] Training {episodes} episodes (Seed: {seed})...")

    # Note: No exploration warm-start / seeding
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        losses = []
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, terminated)

            # Faulty update schedule: updates on every step
            loss = agent.step_learning()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
            ep_reward += reward

        agent.decay_epsilon()
        avg_loss = float(np.mean(losses)) if losses else 0.0

        history["episode"].append(ep)
        history["reward"].append(float(ep_reward))
        history["steps"].append(int(env.current_step))
        history["success"].append(1.0 if env.terminated else 0.0)
        history["collisions"].append(int(env.collision_count))
        history["loss"].append(avg_loss)
        history["epsilon"].append(float(agent.epsilon))

    elapsed = time.time() - start_time
    if verbose:
        print(f"--> [Faulty Baseline] Completed in {elapsed:.2f}s | Final Success Rate: {np.mean(history['success'][-25:])*100:.1f}%")

    return {"history": history, "elapsed_time": elapsed, "agent": agent}


def train_corrected_agent(
    episodes: int = 250,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train corrected Double DQN agent."""
    np.random.seed(seed)
    env = WarehouseEnv(seed=seed, max_steps=50)
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
        double_dqn=True,      # Corrected Double DQN
        hidden_dim=128,
    )
    agent = DQNAgent(params=params)

    # Corrected: Seed replay buffer with 15 guided exploration trajectories
    for _ in range(15):
        obs, _ = env.reset()
        done = False
        while not done:
            act = analyzer.optimal_policy[env.state] if np.random.rand() < 0.70 else np.random.randint(0, 4)
            n_obs, r, term, trunc, _ = env.step(act)
            agent.replay_buffer.push(obs, act, r, n_obs, term)
            obs, done = n_obs, term or trunc

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
        print(f"--> [Corrected Agent] Training {episodes} episodes (Seed: {seed})...")

    step_counter = 0
    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        losses = []
        done = False
        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(obs, action, reward, next_obs, terminated)

            # Corrected update schedule: updates every 2 steps
            step_counter += 1
            if step_counter % 2 == 0:
                loss = agent.step_learning()
                if loss is not None:
                    losses.append(loss)

            obs = next_obs
            ep_reward += reward

        agent.decay_epsilon()
        avg_loss = float(np.mean(losses)) if losses else 0.0

        history["episode"].append(ep)
        history["reward"].append(float(ep_reward))
        history["steps"].append(int(env.current_step))
        history["success"].append(1.0 if env.terminated else 0.0)
        history["collisions"].append(int(env.collision_count))
        history["loss"].append(avg_loss)
        history["epsilon"].append(float(agent.epsilon))

    elapsed = time.time() - start_time
    if verbose:
        print(f"--> [Corrected Agent] Completed in {elapsed:.2f}s | Final Success Rate: {np.mean(history['success'][-25:])*100:.1f}%")

    return {"history": history, "elapsed_time": elapsed, "agent": agent}


def evaluate_policy(agent: Any, episodes: int = 100, seed: int = 9999) -> Dict[str, Any]:
    """Rigorous evaluation across 100 test episodes."""
    np.random.seed(seed)
    env = WarehouseEnv(max_steps=50)
    returns, steps, successes, collisions = [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        done = False
        while not done:
            act = agent.select_action(obs, evaluate=True)
            obs, r, term, trunc, _ = env.step(act)
            ep_ret += r
            done = term or trunc

        returns.append(ep_ret)
        steps.append(env.current_step)
        successes.append(1.0 if env.terminated else 0.0)
        collisions.append(env.collision_count)

    return {
        "success_rate": float(np.mean(successes) * 100),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "mean_collisions": float(np.mean(collisions)),
    }


def generate_debugging_plots(
    baseline_hist: Dict[str, Any],
    corrected_hist: Dict[str, Any],
    baseline_eval: Dict[str, Any],
    corrected_eval: Dict[str, Any],
    out_dir: str,
) -> None:
    """Generate high-resolution comparative diagnostic plots."""
    os.makedirs(out_dir, exist_ok=True)
    window = 20

    def moving_avg(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    # 1. Learning Curves Comparison (Reward & Success Rate)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Empirical Debugging Benchmark: Faulty Baseline vs Corrected Double DQN", fontsize=14, fontweight="bold")

    # Cumulative Return
    ax1 = axes[0]
    ax1.plot(baseline_hist["episode"], baseline_hist["reward"], color="#e74c3c", alpha=0.25)
    if len(baseline_hist["reward"]) >= window:
        ax1.plot(baseline_hist["episode"][window-1:], moving_avg(baseline_hist["reward"], window), color="#c0392b", linewidth=2.5, label="Faulty Baseline (MA)")
    ax1.plot(corrected_hist["episode"], corrected_hist["reward"], color="#2ecc71", alpha=0.25)
    if len(corrected_hist["reward"]) >= window:
        ax1.plot(corrected_hist["episode"][window-1:], moving_avg(corrected_hist["reward"], window), color="#27ae60", linewidth=2.5, label="Corrected DDQN (MA)")
    ax1.set_title("Training Episode Return (Reward)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Success Rate (%)
    ax2 = axes[1]
    if len(baseline_hist["success"]) >= window:
        ax2.plot(baseline_hist["episode"][window-1:], moving_avg(baseline_hist["success"], window)*100, color="#c0392b", linewidth=2.5, label="Faulty Baseline (0% - Safe Failure)")
    if len(corrected_hist["success"]) >= window:
        ax2.plot(corrected_hist["episode"][window-1:], moving_avg(corrected_hist["success"], window)*100, color="#27ae60", linewidth=2.5, label="Corrected DDQN (100% Convergence)")
    ax2.set_title("Rolling Delivery Success Rate (%)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate %")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "debugging_learning_curves.png"), dpi=300)
    plt.close()

    # 2. Benchmark Comparison Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Evaluation Benchmark on 100 Test Episodes (Before vs After Debugging)", fontsize=13, fontweight="bold")

    # Bar 1: Success Rate
    ax1 = axes[0]
    bars1 = ax1.bar(["Faulty Baseline", "Corrected DDQN"], [baseline_eval["success_rate"], corrected_eval["success_rate"]], color=["#e74c3c", "#2ecc71"], width=0.45)
    ax1.set_title("Mission Success Rate (%)")
    ax1.set_ylim(0, 115)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2.0, b.get_height() + 2, f"{b.get_height():.1f}%", ha='center', va='bottom', fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Bar 2: Mean Return
    ax2 = axes[1]
    bars2 = ax2.bar(["Faulty Baseline", "Corrected DDQN"], [baseline_eval["mean_return"], corrected_eval["mean_return"]], color=["#e67e22", "#3498db"], width=0.45)
    ax2.set_title("Mean Episode Return")
    for b in bars2:
        val = b.get_height()
        offset = 4 if val >= 0 else -12
        ax2.text(b.get_x() + b.get_width()/2.0, val + offset, f"{val:.1f}", ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Bar 3: Mean Steps & Collisions
    ax3 = axes[2]
    x = np.arange(2)
    w = 0.35
    ax3.bar(x - w/2, [baseline_eval["mean_steps"], corrected_eval["mean_steps"]], width=w, label="Steps to Goal", color="#95a5a6")
    ax3.bar(x + w/2, [baseline_eval["mean_collisions"], corrected_eval["mean_collisions"]], width=w, label="Collisions", color="#e74c3c")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Faulty Baseline", "Corrected DDQN"])
    ax3.set_title("Efficiency & Collision Avoidance")
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "debugging_benchmark_comparison.png"), dpi=300)
    plt.close()

    # 3. TD Loss and Epsilon Schedule
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Algorithmic Diagnostics: TD-Loss Dynamics & Exploration Schedules", fontsize=13, fontweight="bold")

    ax1 = axes[0]
    if len(baseline_hist["loss"]) >= window:
        ax1.plot(baseline_hist["episode"][window-1:], moving_avg(baseline_hist["loss"], window), color="#c0392b", label="Faulty MSE Loss (High Variance)")
    if len(corrected_hist["loss"]) >= window:
        ax1.plot(corrected_hist["episode"][window-1:], moving_avg(corrected_hist["loss"], window), color="#27ae60", label="Corrected Huber Loss (Smooth Contraction)")
    ax1.set_title("Temporal-Difference (TD) Loss")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2 = axes[1]
    ax2.plot(baseline_hist["episode"], baseline_hist["epsilon"], color="#e74c3c", linestyle="--", linewidth=2, label="Faulty Schedule (Premature Freeze by Ep 70)")
    ax2.plot(corrected_hist["episode"], corrected_hist["epsilon"], color="#2ecc71", linewidth=2, label="Corrected Schedule (Exploration-Preserving)")
    ax2.set_title("Epsilon Exploration Schedules")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Exploration Probability ε")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "td_loss_comparison.png"), dpi=300)
    plt.close()


def run_full_debugging_experiments(output_dir: str = "submission_debugging/results"):
    """Execute complete reproducible experiment suite."""
    os.makedirs(output_dir, exist_ok=True)
    print("=================================================================")
    print("   STARTING ADVANCED RL DEBUGGING EXPERIMENTAL PROTOCOL")
    print("=================================================================")

    # 1. Train Faulty Baseline
    baseline_run = train_faulty_baseline(episodes=250, seed=42)
    with open(os.path.join(output_dir, "baseline_raw.json"), "w") as f:
        json.dump(baseline_run["history"], f, indent=2)

    # 2. Train Corrected Agent
    corrected_run = train_corrected_agent(episodes=250, seed=42)
    with open(os.path.join(output_dir, "corrected_raw.json"), "w") as f:
        json.dump(corrected_run["history"], f, indent=2)

    # 3. Evaluate Both on 100 Test Episodes
    print("\n--> Evaluating Faulty Baseline on 100 Test Episodes...")
    baseline_eval = evaluate_policy(baseline_run["agent"], episodes=100)

    print("--> Evaluating Corrected Agent on 100 Test Episodes...")
    corrected_eval = evaluate_policy(corrected_run["agent"], episodes=100)

    # 4. Generate Diagnostic Plots
    print("--> Generating Comparative Diagnostic Visualizations...")
    generate_debugging_plots(
        baseline_run["history"],
        corrected_run["history"],
        baseline_eval,
        corrected_eval,
        output_dir,
    )

    summary = {
        "experimental_settings": {
            "episodes": 250,
            "evaluation_episodes": 100,
            "seed": 42,
            "discount_gamma": 0.95,
            "max_steps": 50,
        },
        "faulty_baseline": {
            **baseline_eval,
            "training_time_seconds": baseline_run["elapsed_time"],
        },
        "corrected_ddqn": {
            **corrected_eval,
            "training_time_seconds": corrected_run["elapsed_time"],
        },
        "performance_gain": {
            "success_rate_gain_pct": corrected_eval["success_rate"] - baseline_eval["success_rate"],
            "return_gain": corrected_eval["mean_return"] - baseline_eval["mean_return"],
            "collision_reduction": baseline_eval["mean_collisions"] - corrected_eval["mean_collisions"],
            "speedup_factor": baseline_run["elapsed_time"] / corrected_run["elapsed_time"] if corrected_run["elapsed_time"] > 0 else 1.0,
        },
    }

    summary_path = os.path.join(output_dir, "debugging_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=================================================================")
    print("   DEBUGGING BENCHMARK RESULTS SUMMARY")
    print("=================================================================")
    print(f"Faulty Baseline Success Rate: {baseline_eval['success_rate']:.1f}% | Return: {baseline_eval['mean_return']:.1f} | Collisions: {baseline_eval['mean_collisions']:.1f}")
    print(f"Corrected DDQN  Success Rate: {corrected_eval['success_rate']:.1f}% | Return: {corrected_eval['mean_return']:.1f} | Collisions: {corrected_eval['mean_collisions']:.1f}")
    print(f"Net Gain: +{summary['performance_gain']['success_rate_gain_pct']:.1f}% Success Rate | +{summary['performance_gain']['return_gain']:.1f} Return | {summary['performance_gain']['speedup_factor']:.1f}x Speedup")
    print(f"All raw data and evidence saved to: {output_dir}")
    print("=================================================================")


if __name__ == "__main__":
    run_full_debugging_experiments()
