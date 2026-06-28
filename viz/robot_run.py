"""Animate the robot's real Gazebo run over the value function + policy.

Top-down view: the value-iteration heatmap as background, walls, the optimal
policy arrows, the goal, and the robot tracing the path it actually drove in
Gazebo (from a captured /odom trajectory). This is the LinkedIn hero shot —
it ties the RL (heatmap + policy) directly to the robot's motion, and renders
far cleaner than Gazebo's GUI (which can't paint over XQuartz's old OpenGL).

Usage: python viz/robot_run.py scenarios/maze.yaml <odom_stream.txt> out.gif

The trajectory file is whatever `ros2 topic echo /odom --field
pose.pose.position` prints: repeating `x: <v>` / `y: <v>` / `z: <v>` blocks.
"""
from __future__ import annotations

import os
import sys

# make `gridworld` (repo root) importable when run as `python viz/robot_run.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from gridworld.scenario import load_scenario
from gridworld.solvers import value_iteration, policy_actions
from gridworld.mdp import ACTIONS, _DELTA


def load_trajectory(path):
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("x:"):
                xs.append(float(line.split(":", 1)[1]))
            elif line.startswith("y:"):
                ys.append(float(line.split(":", 1)[1]))
    n = min(len(xs), len(ys))
    return list(zip(xs[:n], ys[:n]))


def render(scenario_path, traj_path, out_path, max_frames=160, fps=15):
    sc = load_scenario(scenario_path)
    mdp = sc.build_mdp()
    cfg = sc.coords
    V, policy, _ = value_iteration(mdp, theta=1e-9)
    acts = policy_actions(policy)
    grid_V = mdp.grid_values(V)

    traj = load_trajectory(traj_path)
    if len(traj) > max_frames:
        traj = traj[:: len(traj) // max_frames]

    def to_px(x, y):
        # world (m) -> imshow (col, row) where cell (r,c) center is at (c, r)
        return ((x - cfg.origin_x) / cfg.cell_size - 0.5,
                (cfg.origin_y - y) / cfg.cell_size - 0.5)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(grid_V, cmap="viridis")
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            if sc.grid[r][c] == "#":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="0.15"))
    for i, (r, c) in enumerate(mdp.states):
        if mdp.terminal[i]:
            ax.plot(c, r, marker="*", color="gold", markersize=26, markeredgecolor="k", zorder=5)
            continue
        dr, dc = _DELTA[ACTIONS[acts[i]]]
        ax.arrow(c, r, 0.25 * dc, 0.25 * dr, head_width=0.12,
                 color="white", alpha=0.55, length_includes_head=True)

    trail, = ax.plot([], [], "-", color="red", lw=2.5, alpha=0.8, zorder=6)
    dot, = ax.plot([], [], "o", color="red", markersize=13, markeredgecolor="k", zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("TurtleBot3 driving the Dynamic-Programming policy")

    px, py = [], []

    def update(k):
        cx, cy = to_px(*traj[k])
        px.append(cx); py.append(cy)
        trail.set_data(px, py)
        dot.set_data([cx], [cy])
        return trail, dot

    anim = animation.FuncAnimation(fig, update, frames=len(traj), interval=1000 / fps, blit=False)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    scenario, traj, out = sys.argv[1], sys.argv[2], sys.argv[3]
    print("wrote", render(scenario, traj, out))
