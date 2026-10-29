"""Evaluation and benchmark script comparing Untrained vs Trained Agent.

Generates actual metric comparisons and publication-quality plots in submission/evidence/.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from warehouse_rl.warehouse_env import WarehouseEnv
from warehouse_rl.dqn_agent import DQNAgent, DQNHyperparameters
from warehouse_rl.bellman_analyzer import BellmanAnalyzer


def run_evaluation_episodes(
    agent: DQNAgent,
    env: WarehouseEnv,
    episodes: int = 100,
    is_untrained: bool = False,
    seed: int = 12345,
) -> Dict[str, Any]:
    """Run honest evaluation over fixed seed episodes."""
    np.random.seed(seed)
    returns = []
    step_counts = []
    successes = []
    collisions = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_return = 0.0
        done = False
        while not done:
            if is_untrained:
                # Random exploration / untrained policy
                action = np.random.randint(0, env.num_actions)
            else:
                action = agent.select_action(obs, evaluate=True)

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_return += reward
            done = terminated or truncated

        returns.append(ep_return)
        step_counts.append(env.current_step)
        successes.append(1.0 if env.terminated else 0.0)
        collisions.append(env.collision_count)

    return {
        "success_rate": float(np.mean(successes) * 100),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(step_counts)),
        "std_steps": float(np.std(step_counts)),
        "mean_collisions": float(np.mean(collisions)),
        "returns": [float(x) for x in returns],
        "steps": [int(x) for x in step_counts],
        "successes": [float(x) for x in successes],
        "collisions": [int(x) for x in collisions],
    }


def plot_training_history(history_path: str, output_path: str) -> None:
    """Plot actual training loss, return, success rate, and epsilon."""
    with open(history_path, "r") as f:
        data = json.load(f)

    episodes = data["episode"]
    rewards = data["reward"]
    successes = data["success"]
    losses = data["loss"]
    epsilons = data["epsilon"]

    # Moving averages
    window = 25
    def moving_avg(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DQN Training Progress & Convergence Metrics", fontsize=15, fontweight="bold")

    # 1. Cumulative Reward
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.25, color="steelblue", label="Episode Return")
    if len(rewards) >= window:
        ma_rew = moving_avg(rewards, window)
        ax1.plot(episodes[window-1:], ma_rew, color="darkblue", linewidth=2, label=f"MA ({window})")
    ax1.set_title("Episode Return (Cumulative Reward)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Return")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Success Rate
    ax2 = axes[0, 1]
    if len(successes) >= window:
        ma_succ = moving_avg(successes, window) * 100
        ax2.plot(episodes[window-1:], ma_succ, color="forestgreen", linewidth=2, label="Success Rate (%)")
    ax2.set_title("Delivery Success Rate (%)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate %")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 3. TD Loss
    ax3 = axes[1, 0]
    ax3.plot(episodes, losses, color="coral", alpha=0.6, label="TD Loss")
    if len(losses) >= window:
        ma_loss = moving_avg(losses, window)
        ax3.plot(episodes[window-1:], ma_loss, color="darkred", linewidth=2, label=f"MA ({window})")
    ax3.set_title("Temporal-Difference (TD) Huber Loss")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # 4. Epsilon Exploration Decay
    ax4 = axes[1, 1]
    ax4.plot(episodes, epsilons, color="purple", linewidth=2, label="Epsilon (ε)")
    ax4.set_title("Epsilon Exploration Schedule")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Exploration Probability ε")
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training plot to: {output_path}")


def plot_comparison(untrained_res: Dict[str, Any], trained_res: Dict[str, Any], output_path: str) -> None:
    """Plot comparative metrics between untrained baseline and trained agent."""
    labels = ["Success Rate (%)", "Mean Steps / Episode", "Mean Collisions"]
    untrained_vals = [
        untrained_res["success_rate"],
        untrained_res["mean_steps"],
        untrained_res["mean_collisions"],
    ]
    trained_vals = [
        trained_res["success_rate"],
        trained_res["mean_steps"],
        trained_res["mean_collisions"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Agent Performance Benchmark: Untrained Baseline vs Trained DQN Agent", fontsize=14, fontweight="bold")

    # Bar 1: Success Rate
    ax1 = axes[0]
    bars1 = ax1.bar(["Untrained", "Trained DQN"], [untrained_vals[0], trained_vals[0]], color=["#e74c3c", "#2ecc71"], width=0.5)
    ax1.set_title("Delivery Success Rate")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_ylim(0, 110)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Bar 2: Mean Return
    ax2 = axes[1]
    bars2 = ax2.bar(["Untrained", "Trained DQN"], [untrained_res["mean_return"], trained_res["mean_return"]], color=["#e67e22", "#3498db"], width=0.5)
    ax2.set_title("Mean Episode Return")
    ax2.set_ylabel("Cumulative Discounted Return")
    for bar in bars2:
        yval = bar.get_height()
        offset = 5 if yval >= 0 else -15
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + offset, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Bar 3: Mean Steps & Collisions
    ax3 = axes[2]
    x = np.arange(2)
    w = 0.35
    ax3.bar(x - w/2, [untrained_vals[1], trained_vals[1]], width=w, label="Steps to Goal", color="#95a5a6")
    ax3.bar(x + w/2, [untrained_vals[2], trained_vals[2]], width=w, label="Collisions", color="#e74c3c")
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Untrained", "Trained DQN"])
    ax3.set_title("Efficiency & Collision Avoidance")
    ax3.set_ylabel("Count per Episode")
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def plot_bellman_heatmap(analyzer: BellmanAnalyzer, output_path: str) -> None:
    """Plot exact Bellman state value V*(s) heatmaps for has_package=0 and has_package=1."""
    if not analyzer.is_solved:
        analyzer.solve_exact_value_iteration()

    rows = analyzer.env.rows
    cols = analyzer.env.cols

    grid_p0 = np.full((rows, cols), np.nan)
    grid_p1 = np.full((rows, cols), np.nan)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in analyzer.env.obstacles:
                grid_p0[r, c] = analyzer.V_star.get((r, c, 0), np.nan)
                grid_p1[r, c] = analyzer.V_star.get((r, c, 1), np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Exact Bellman State-Value Function V*(s) Across Warehouse Grid", fontsize=14, fontweight="bold")

    # Heatmap Phase 1: Seeking Pickup (has_package = 0)
    ax1 = axes[0]
    im1 = ax1.imshow(grid_p0, cmap="viridis")
    ax1.set_title("Phase 1: Seeking Package Pickup (p=0)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Annotate key locations
    ax1.text(analyzer.env.start_pos[1], analyzer.env.start_pos[0], "S", color="white", ha="center", va="center", fontweight="bold", fontsize=14)
    ax1.text(analyzer.env.pickup_pos[1], analyzer.env.pickup_pos[0], "P", color="yellow", ha="center", va="center", fontweight="bold", fontsize=14)
    ax1.text(analyzer.env.delivery_pos[1], analyzer.env.delivery_pos[0], "D", color="red", ha="center", va="center", fontweight="bold", fontsize=14)

    # Heatmap Phase 2: Carrying to Delivery (has_package = 1)
    ax2 = axes[1]
    im2 = ax2.imshow(grid_p1, cmap="magma")
    ax2.set_title("Phase 2: Carrying Package to Delivery (p=1)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax2.text(analyzer.env.start_pos[1], analyzer.env.start_pos[0], "S", color="white", ha="center", va="center", fontweight="bold", fontsize=14)
    ax2.text(analyzer.env.pickup_pos[1], analyzer.env.pickup_pos[0], "P", color="cyan", ha="center", va="center", fontweight="bold", fontsize=14)
    ax2.text(analyzer.env.delivery_pos[1], analyzer.env.delivery_pos[0], "D", color="lightgreen", ha="center", va="center", fontweight="bold", fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Bellman heatmap to: {output_path}")


def evaluate_and_generate_evidence(
    model_path: str = "models/dqn_warehouse_model.pth",
    history_path: str = "models/training_history.json",
    evidence_dir: str = "submission/evidence",
    episodes: int = 100,
) -> Dict[str, Any]:
    """Execute complete evaluation suite and save all evidence assets."""
    os.makedirs(evidence_dir, exist_ok=True)
    env = WarehouseEnv(max_steps=60)

    # Load trained agent
    agent = DQNAgent()
    agent.load_checkpoint(model_path)

    print(f"Running evaluation on {episodes} episodes for Untrained Baseline...")
    untrained_results = run_evaluation_episodes(agent, env, episodes=episodes, is_untrained=True)

    print(f"Running evaluation on {episodes} episodes for Trained DQN Agent...")
    trained_results = run_evaluation_episodes(agent, env, episodes=episodes, is_untrained=False)

    print("\n--- Evaluation Summary ---")
    print(f"Untrained Baseline Success Rate: {untrained_results['success_rate']:.1f}% | Mean Return: {untrained_results['mean_return']:.1f} | Collisions: {untrained_results['mean_collisions']:.1f}")
    print(f"Trained DQN Agent  Success Rate: {trained_results['success_rate']:.1f}% | Mean Return: {trained_results['mean_return']:.1f} | Collisions: {trained_results['mean_collisions']:.1f}")

    # Generate charts
    training_curve_path = os.path.join(evidence_dir, "training_curves.png")
    plot_training_history(history_path, training_curve_path)

    comparison_path = os.path.join(evidence_dir, "evaluation_comparison.png")
    plot_comparison(untrained_results, trained_results, comparison_path)

    heatmap_path = os.path.join(evidence_dir, "bellman_value_heatmap.png")
    analyzer = BellmanAnalyzer(env)
    plot_bellman_heatmap(analyzer, heatmap_path)

    # Save metrics JSON
    metrics_data = {
        "evaluation_episodes": episodes,
        "untrained_baseline": {
            "success_rate": untrained_results["success_rate"],
            "mean_return": untrained_results["mean_return"],
            "std_return": untrained_results["std_return"],
            "mean_steps": untrained_results["mean_steps"],
            "std_steps": untrained_results["std_steps"],
            "mean_collisions": untrained_results["mean_collisions"],
        },
        "trained_dqn": {
            "success_rate": trained_results["success_rate"],
            "mean_return": trained_results["mean_return"],
            "std_return": trained_results["std_return"],
            "mean_steps": trained_results["mean_steps"],
            "std_steps": trained_results["std_steps"],
            "mean_collisions": trained_results["mean_collisions"],
        },
    }

    metrics_path = os.path.join(evidence_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    return metrics_data


if __name__ == "__main__":
    evaluate_and_generate_evidence()
