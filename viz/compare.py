"""Compare Value Iteration and Policy Iteration on the same MDP."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridworld.solvers import value_iteration, policy_iteration, policy_actions, greedy_policy


def compare_vi_pi(scenario):
    mdp = scenario.build_mdp()
    V_vi, _, hist_vi = value_iteration(mdp, theta=1e-9)
    V_pi, pol_pi, hist_pi = policy_iteration(mdp)
    a_vi = policy_actions(greedy_policy(mdp, V_vi))
    a_pi = policy_actions(pol_pi)
    agree = bool(np.array_equal(a_vi[~mdp.terminal], a_pi[~mdp.terminal]))
    assert agree, "VI and PI converged to different policies"
    return {
        "vi_sweeps": len(hist_vi["delta"]),
        "pi_iters": hist_pi["stable_at"],
        "vi_delta": hist_vi["delta"],
        "agree": agree,
        "_V_vi": V_vi,
        "_V_pi": V_pi,
        "_mdp": mdp,
    }


def render_comparison(scenario, out_path):
    r = compare_vi_pi(scenario)
    mdp = r["_mdp"]
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4))

    ax0.semilogy(range(1, len(r["vi_delta"]) + 1), r["vi_delta"])
    ax0.set_xlabel("VI sweep"); ax0.set_ylabel("max Bellman residual")
    ax0.set_title(f"VI converged in {r['vi_sweeps']} sweeps\n"
                  f"(PI: {r['pi_iters']} policy iterations)")
    ax0.grid(alpha=0.3)

    im1 = ax1.imshow(mdp.grid_values(r["_V_vi"]), cmap="viridis")
    ax1.set_title("Value Iteration V*"); ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(mdp.grid_values(r["_V_pi"]), cmap="viridis")
    ax2.set_title("Policy Iteration V*"); ax2.set_xticks([]); ax2.set_yticks([])
    fig.colorbar(im2, ax=ax2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
