"""Animate the value function converging under value iteration."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: required for rendering in a container/CI
import matplotlib.pyplot as plt
from matplotlib import animation

from gridworld.mdp import ACTIONS, _DELTA
from gridworld.solvers import value_iteration, policy_actions


def value_iteration_frames(mdp):
    final_V, final_policy, history = value_iteration(mdp, theta=1e-9)
    frames = [mdp.grid_values(V) for V in history["V"]]
    return frames, final_V, final_policy


def render_value_iteration(scenario, out_path, *, fps=4):
    mdp = scenario.build_mdp()
    frames, final_V, final_policy = value_iteration_frames(mdp)
    actions = policy_actions(final_policy)

    fig, ax = plt.subplots(figsize=(6, 5))
    vmin = np.nanmin(frames[-1])
    vmax = np.nanmax(frames[-1])
    im = ax.imshow(frames[0], cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title("")
    fig.colorbar(im, ax=ax, label="V(s)")

    def update(k):
        im.set_data(frames[k])
        title.set_text(f"Value iteration — sweep {k + 1}/{len(frames)}")
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )

    # Also save a still with the optimal-policy arrows for the post.
    if out_path.endswith(".gif"):
        anim.save(out_path, writer="pillow", fps=fps)
    else:
        anim.save(out_path, writer="ffmpeg", fps=fps)

    # Save a companion still with policy arrows for the post.
    _save_policy_still(mdp, final_V, actions, out_path)
    plt.close(fig)
    return out_path


def _save_policy_still(mdp, V, actions, anim_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    grid_V = mdp.grid_values(V)
    ax.imshow(grid_V, cmap="viridis")
    for i, (r, c) in enumerate(mdp.states):
        if mdp.terminal[i]:
            continue
        dr, dc = _DELTA[ACTIONS[actions[i]]]
        ax.arrow(c, r, 0.3 * dc, 0.3 * dr, head_width=0.18,
                 length_includes_head=True, color="white")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Optimal policy π*")
    still = anim_path.rsplit(".", 1)[0] + "_policy.png"
    fig.savefig(still, dpi=150, bbox_inches="tight")
    plt.close(fig)
