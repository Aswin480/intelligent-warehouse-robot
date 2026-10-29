# inference_app.py  -  Model Inference Interface
# Intelligent Warehouse Robot Navigation  --  Deep RL Decision Engine
# Student  : Aswin S
# Reg. No  : RA2311026050067
# Subject  : 21CSE417TA / Fundamentals of Reinforcement Learning
# Institute: SRM Institute of Science and Technology, Tiruchirappalli - 621 105
# School   : School of Computing | B.Tech AI & ML
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch

from warehouse_rl.warehouse_env import (
    WarehouseEnv,
    WarehouseRewardConfig,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_NAMES,
    DEFAULT_WAREHOUSE_GRID,
)
from warehouse_rl.dqn_agent import DQNAgent, DQNHyperparameters
from warehouse_rl.bellman_analyzer import BellmanAnalyzer
from evaluate_agent import run_evaluation_episodes

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Warehouse Robot | RL Decision Engine | Aswin S",
    page_icon="robot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS  --  dark-gradient theme matching report aesthetics
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.cover-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #1d4ed8 100%);
    border-radius: 16px; padding: 36px 40px; margin-bottom: 28px;
    border: 1px solid rgba(59,130,246,0.3);
    box-shadow: 0 8px 40px rgba(30,58,138,0.4);
}
.cover-institute {
    font-size: 0.85rem; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; color: #93c5fd; margin-bottom: 4px;
}
.cover-title {
    font-size: 2.0rem; font-weight: 800; color: #ffffff;
    line-height: 1.25; margin-bottom: 10px;
}
.cover-title span { color: #60a5fa; }
.cover-subtitle { font-size: 1.0rem; color: #bfdbfe; margin-bottom: 20px; }
.cover-meta { display: flex; gap: 20px; flex-wrap: wrap; }
.cover-chip {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px; padding: 8px 16px;
    font-size: 0.82rem; color: #e2e8f0;
}
.cover-chip strong {
    color: #93c5fd; display: block; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px;
}
.section-title {
    font-size: 1.25rem; font-weight: 700; color: #1e3a8a;
    border-left: 4px solid #3b82f6; padding-left: 12px; margin: 24px 0 12px;
}
.theory-box {
    background: #f0f9ff; border: 1px solid #bae6fd;
    border-radius: 10px; padding: 20px 24px; margin: 12px 0;
}
.footer-strip {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    border-radius: 12px; padding: 16px 24px;
    text-align: center; color: #93c5fd;
    font-size: 0.82rem; margin-top: 40px;
}
div[data-testid="stTabs"] button { font-weight: 600; font-size: 0.9rem; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#f8fafc,#eff6ff);
    border-radius: 10px; border: 1px solid #dbeafe; padding: 12px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "env" not in st.session_state:
    st.session_state.env = WarehouseEnv(max_steps=60)

if "agent" not in st.session_state:
    agent = DQNAgent()
    model_path = "models/dqn_warehouse_model.pth"
    if os.path.exists(model_path):
        agent.load_checkpoint(model_path)
    st.session_state.agent = agent

if "sim_history" not in st.session_state:
    st.session_state.sim_history = []

if "step_event" not in st.session_state:
    st.session_state.step_event = "Environment Ready"

if "last_action" not in st.session_state:
    st.session_state.last_action = "None"

if "last_reward" not in st.session_state:
    st.session_state.last_reward = 0.0

if "bellman_analyzer" not in st.session_state:
    _az = BellmanAnalyzer(st.session_state.env)
    _az.solve_exact_value_iteration()
    st.session_state.bellman_analyzer = _az


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def reset_simulation():
    obs, info = st.session_state.env.reset()
    st.session_state.sim_history = [{
        "step": 0, "state": st.session_state.env.state,
        "action": "-", "reward": 0.0, "return": 0.0, "event": "Reset",
    }]
    st.session_state.step_event = "Environment Reset"
    st.session_state.last_action = "None"
    st.session_state.last_reward = 0.0


def step_simulation(action: int):
    env = st.session_state.env
    if env.terminated or env.truncated:
        st.session_state.step_event = "Episode finished - please reset."
        return
    obs, reward, terminated, truncated, info = env.step(action)
    st.session_state.last_action = ACTION_NAMES[action]
    st.session_state.last_reward = reward
    st.session_state.step_event = info["event"]
    st.session_state.sim_history.append({
        "step": env.current_step, "state": env.state,
        "action": ACTION_NAMES[action], "reward": reward,
        "return": env.episode_return, "event": info["event"],
    })


def render_grid_html(env):
    parts = [
        "<div style='display:inline-block;border:3px solid #1e293b;"
        "border-radius:12px;padding:8px;background:#f1f5f9;"
        "box-shadow:0 4px 20px rgba(0,0,0,0.1);'>"
    ]
    for r in range(env.rows):
        parts.append("<div style='display:flex;'>")
        for c in range(env.cols):
            is_robot    = (r, c) == env.robot_pos
            is_start    = (r, c) == env.start_pos
            is_pickup   = (r, c) == env.pickup_pos
            is_delivery = (r, c) == env.delivery_pos
            is_obstacle = (r, c) in env.obstacles

            bg, border, content, extra = "#FFFFFF", "1px solid #CBD5E1", "", ""

            if is_obstacle:
                bg = "#1e293b"
                content = "<span style='font-size:18px;'>&#x1F9F1;</span>"
            elif is_robot:
                bg, border = "#bfdbfe", "2px solid #3b82f6"
                extra = "box-shadow:0 0 0 3px rgba(59,130,246,0.25);"
                content = (
                    "<span style='font-size:20px;'>&#x1F916;&#x1F4E6;</span>"
                    if env.has_package
                    else "<span style='font-size:20px;'>&#x1F916;</span>"
                )
            elif is_pickup and not env.has_package:
                bg, border = "#fef9c3", "2px solid #ca8a04"
                content = "<span style='font-size:18px;'>&#x1F4E6;</span>"
            elif is_pickup and env.has_package:
                bg = "#f0fdf4"
                content = "<span style='font-size:14px;color:#16a34a;'>&#x2713;</span>"
            elif is_delivery:
                bg, border = "#bbf7d0", "2px solid #16a34a"
                content = "<span style='font-size:18px;'>&#x1F3C1;</span>"
            elif is_start:
                bg = "#ede9fe"
                content = "<span style='font-size:13px;font-weight:800;color:#6d28d9;'>S</span>"

            parts.append(
                f"<div style='width:56px;height:56px;background:{bg};border:{border};"
                f"display:flex;align-items:center;justify-content:center;"
                f"margin:2px;border-radius:8px;{extra}'>{content}</div>"
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# COVER BANNER
# ─────────────────────────────────────────────────────────────────────────────
BANNER = """
<div class="cover-banner">
  <div class="cover-institute">
    SRM Institute of Science and Technology, Tiruchirappalli &mdash; 621 105
    &nbsp;|&nbsp; School of Computing
  </div>
  <div class="cover-title">
    Intelligent Warehouse Robot Navigation<br>
    <span>RL Decision Engine &amp; Model Inference Interface</span>
  </div>
  <div class="cover-subtitle">
    21CSE417TA &nbsp;/&nbsp; Fundamentals of Reinforcement Learning
    &nbsp;|&nbsp; B.Tech Artificial Intelligence &amp; Machine Learning
  </div>
  <div class="cover-meta">
    <div class="cover-chip"><strong>Student</strong>Aswin S</div>
    <div class="cover-chip"><strong>Registration No.</strong>RA2311026050067</div>
    <div class="cover-chip"><strong>Algorithm</strong>Double DQN + Bellman VI</div>
    <div class="cover-chip"><strong>State Space</strong>Finite MDP &nbsp;|&nbsp; |S| = 2 x free-cells</div>
    <div class="cover-chip"><strong>Action Space</strong>|A| = 4 (UP, DOWN, LEFT, RIGHT)</div>
  </div>
</div>
"""
st.markdown(BANNER, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="text-align:center;padding:12px 0 8px;">
        <div style="font-size:48px;">&#x1F916;</div>
        <div style="font-weight:800;font-size:1.1rem;color:#1e3a8a;">Robot Control Panel</div>
        <div style="font-size:0.78rem;color:#64748b;">Aswin S &nbsp;|&nbsp; RA2311026050067</div>
        </div>
        <hr style="border:1px solid #e2e8f0;margin:8px 0 16px;"/>""",
        unsafe_allow_html=True,
    )

    st.subheader("MDP Hyperparameters")
    gamma_val     = st.slider("Discount Factor (gamma)", 0.50, 0.99, 0.95, 0.01)
    max_steps_val = st.slider("Max Episode Steps", 20, 120, 60, 5)

    if st.button("Apply MDP Settings", use_container_width=True):
        st.session_state.env = WarehouseEnv(gamma=gamma_val, max_steps=max_steps_val)
        _az2 = BellmanAnalyzer(st.session_state.env)
        _az2.solve_exact_value_iteration()
        st.session_state.bellman_analyzer = _az2
        reset_simulation()
        st.success("MDP parameters updated and Value Iteration re-solved!")

    st.markdown("---")
    st.subheader("Model Status")
    model_exists = os.path.exists("models/dqn_warehouse_model.pth")
    if model_exists:
        _ckpt = torch.load("models/dqn_warehouse_model.pth", map_location="cpu", weights_only=False)
        _steps = _ckpt.get("total_steps", "N/A")
        _eps   = _ckpt.get("epsilon", "N/A")
        st.success("Pretrained Checkpoint Loaded")
        st.caption(f"Total gradient steps: `{_steps}`")
        if isinstance(_eps, float):
            st.caption(f"Final epsilon: `{_eps:.4f}`")
    else:
        st.warning("No checkpoint found. Train the agent first.")

    if st.button("Reload Checkpoint", use_container_width=True):
        if model_exists:
            st.session_state.agent.load_checkpoint("models/dqn_warehouse_model.pth")
            st.success("Checkpoint reloaded!")
        else:
            st.error("No .pth file found in models/.")

    st.markdown("---")
    st.subheader("Live Episode Stats")
    _env = st.session_state.env
    st.metric("Current Step", f"{_env.current_step} / {_env.max_steps}")
    st.metric("Episode Return", f"{_env.episode_return:.1f}")
    st.metric("Collisions", str(_env.collision_count))
    st.metric("Package", "Carrying" if _env.has_package else "Seeking")
    if _env.terminated:
        st.success("Mission Complete!")
    elif _env.truncated:
        st.error("Episode Truncated")

    st.markdown("---")
    st.markdown(
        """<div style="font-size:0.72rem;color:#94a3b8;text-align:center;line-height:1.8;">
        <strong>Reward Structure</strong><br>
        Goal Delivery: <code>+100</code><br>
        Package Pickup: <code>+25</code><br>
        Step Penalty: <code>-1</code><br>
        Obstacle Hit: <code>-10</code><br>
        Boundary Hit: <code>-5</code><br>
        Timeout: <code>-20</code>
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Simulation",
    "Train DQN Agent",
    "Evaluation Benchmark",
    "Bellman Theory Explorer",
    "RL Debugging Analysis",
])

# ===========================================================================
# TAB 1 -- INTERACTIVE SIMULATION
# ===========================================================================
with tab1:
    st.markdown(
        '<div class="section-title">Autonomous Warehouse Robot -- Live Navigation Simulation</div>',
        unsafe_allow_html=True,
    )
    col_left, col_right = st.columns([1.05, 1.2], gap="large")

    with col_left:
        st.markdown("#### Warehouse Floor Grid")
        _sim_env = st.session_state.env
        st.markdown(render_grid_html(_sim_env), unsafe_allow_html=True)
        st.caption("Robot | Package Pickup (P) | Delivery (D) | Obstacle | S = Start")

        st.markdown("---")
        st.markdown("#### Episode Controls")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Step Agent", type="primary", use_container_width=True):
                _obs = _sim_env.get_observation()
                _act = st.session_state.agent.select_action(_obs, evaluate=True)
                step_simulation(_act)
                st.rerun()
        with c2:
            if st.button("Full Episode", use_container_width=True):
                reset_simulation()
                _s = 0
                while not (_sim_env.terminated or _sim_env.truncated) and _s < _sim_env.max_steps:
                    _obs = _sim_env.get_observation()
                    _act = st.session_state.agent.select_action(_obs, evaluate=True)
                    step_simulation(_act)
                    _s += 1
                st.rerun()
        with c3:
            if st.button("Reset Episode", use_container_width=True):
                reset_simulation()
                st.rerun()

        st.markdown("##### Manual Action Override")
        m1, m2, m3, m4 = st.columns(4)
        if m1.button("UP",    use_container_width=True): step_simulation(ACTION_UP);    st.rerun()
        if m2.button("DOWN",  use_container_width=True): step_simulation(ACTION_DOWN);  st.rerun()
        if m3.button("LEFT",  use_container_width=True): step_simulation(ACTION_LEFT);  st.rerun()
        if m4.button("RIGHT", use_container_width=True): step_simulation(ACTION_RIGHT); st.rerun()

    with col_right:
        st.markdown("#### Real-Time Telemetry & MDP State")
        _e = st.session_state.env
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Robot Position", f"({_e.robot_pos[0]}, {_e.robot_pos[1]})")
        rc2.metric("Package", "Carrying" if _e.has_package else "Seeking")
        rc3.metric("Step",    f"{_e.current_step}/{_e.max_steps}")
        rc4.metric("Return",  f"{_e.episode_return:.1f}")
        rd1, rd2, rd3 = st.columns(3)
        rd1.metric("Last Action", st.session_state.last_action)
        rd2.metric("Last Reward", f"{st.session_state.last_reward:.1f}")
        rd3.metric("Collisions",  str(_e.collision_count))

        if _e.terminated:
            st.success("Delivery Mission Accomplished! Package successfully delivered.")
        elif _e.truncated:
            st.error("Episode Truncated -- max step limit reached without delivery.")
        else:
            st.info(f"Last Event: `{st.session_state.step_event}`")

        st.markdown("#### DQN Neural Network -- Predicted Q-Values")
        _obs_vec = _e.get_observation()
        _q_vals  = st.session_state.agent.get_q_values(_obs_vec)
        _q_df = pd.DataFrame({
            "Action": [ACTION_NAMES[i] for i in range(4)],
            "Q-Value": _q_vals,
        }).set_index("Action")
        _best = int(np.argmax(_q_vals))
        st.bar_chart(_q_df, color="#3b82f6")
        st.caption(
            f"Greedy Policy: pi*(s) = **{ACTION_NAMES[_best]}** "
            f"with Q = `{_q_vals[_best]:.3f}` (max over all actions)"
        )

        st.markdown("#### Observation Vector (Neural Network Input)")
        _obs_labels = ["norm_row", "norm_col", "has_package", "dist_pickup", "dist_delivery"]
        st.dataframe(
            pd.DataFrame({"Feature": _obs_labels, "Value": _obs_vec.tolist()}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Step-by-Step Transition Log")
        if st.session_state.sim_history:
            st.dataframe(
                pd.DataFrame(st.session_state.sim_history).tail(10),
                use_container_width=True, hide_index=True,
            )


# ===========================================================================
# TAB 2 -- TRAIN DQN AGENT
# ===========================================================================
with tab2:
    st.markdown('<div class="section-title">Deep Q-Network (DQN) Training Engine</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="theory-box">
        <b>Training Architecture:</b> Experience Replay decouples sample correlations.
        A <i>Target Network</i> (updated every tau steps) stabilizes the Bellman TD-target.
        An <i>epsilon-greedy schedule</i> decays exploration from 1.0 to 0.05.
        <b>Double DQN</b> uses the online network to select actions and the target network
        to evaluate them, eliminating maximization bias.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### Configure Training Hyperparameters")
    tc1, tc2, tc3 = st.columns(3)
    train_eps   = tc1.slider("Training Episodes", 100, 1500, 500, step=50)
    train_lr    = tc2.select_slider("Learning Rate (Adam)", options=[1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)
    train_gamma = tc3.slider("Discount Factor gamma", 0.80, 0.99, 0.95, step=0.01)
    tc4, tc5, tc6 = st.columns(3)
    train_batch = tc4.selectbox("Batch Size", [32, 64, 128], index=1)
    use_double  = tc5.checkbox("Enable Double DQN", value=True)
    decay_rate  = tc6.slider("Epsilon Decay Rate", 0.980, 0.999, 0.992, step=0.001)
    col_tgt, col_hid = st.columns(2)
    target_upd  = col_tgt.slider("Target Net Update Freq (steps)", 50, 500, 200, step=50)
    hidden_dim  = col_hid.selectbox("Hidden Layer Dimension", [64, 128, 256], index=1)

    if st.button("Start Training DQN Agent", type="primary"):
        _sb = st.empty(); _pb = st.progress(0.0); _cs = st.empty()
        _tenv = WarehouseEnv(gamma=train_gamma, max_steps=60)
        _tp   = DQNHyperparameters(
            gamma=train_gamma, lr=train_lr, batch_size=train_batch,
            epsilon_decay=decay_rate, double_dqn=use_double,
            target_update_freq=target_upd, hidden_dim=hidden_dim,
        )
        _nagent = DQNAgent(params=_tp)
        _rews, _succ, _loss, _eps_list = [], [], [], []
        _t0 = time.time()

        for ep in range(1, train_eps + 1):
            _o, _ = _tenv.reset()
            _tr, _ls, _done = 0.0, [], False
            while not _done:
                _a = _nagent.select_action(_o, evaluate=False)
                _no, _r, _term, _trunc, _ = _tenv.step(_a)
                _done = _term or _trunc
                _nagent.replay_buffer.push(_o, _a, _r, _no, _term)
                _lv = _nagent.step_learning()
                if _lv is not None: _ls.append(_lv)
                _o = _no; _tr += _r
            _nagent.decay_epsilon()
            _rews.append(_tr)
            _succ.append(1.0 if _tenv.terminated else 0.0)
            _loss.append(float(np.mean(_ls)) if _ls else 0.0)
            _eps_list.append(_nagent.epsilon)
            _pb.progress(ep / train_eps)
            if ep % 25 == 0 or ep == train_eps:
                _sr = np.mean(_succ[-50:]) * 100
                _sb.markdown(
                    f"**Episode {ep}/{train_eps}** | "
                    f"Success (last 50): `{_sr:.1f}%` | "
                    f"Avg Return: `{np.mean(_rews[-50:]):.1f}` | "
                    f"epsilon: `{_nagent.epsilon:.4f}`"
                )
                _cs.line_chart(pd.DataFrame({
                    "Episode Return": _rews,
                    "Rolling Success %": [np.mean(_succ[max(0,i-50):i+1])*100 for i in range(len(_succ))],
                }))

        _dur = time.time() - _t0
        _nagent.save_checkpoint("models/dqn_warehouse_model.pth")
        st.session_state.agent = _nagent
        _hd = {"episode": list(range(1, train_eps+1)), "reward": _rews,
               "success": _succ, "loss": _loss, "epsilon": _eps_list}
        with open("models/training_history.json", "w") as _fh:
            import json as _json; _json.dump(_hd, _fh)
        _fs = np.mean(_succ[-100:]) * 100
        st.success(
            f"Training done in {_dur:.1f}s | Final 100-ep success rate: {_fs:.1f}% | "
            "Checkpoint saved: models/dqn_warehouse_model.pth"
        )
        st.line_chart(pd.DataFrame({"Episode Return": _rews, "Loss x100": [x*100 for x in _loss]}))

    elif os.path.exists("models/training_history.json"):
        with open("models/training_history.json") as _fh:
            _hist = json.load(_fh)
        st.markdown("#### Previously Recorded Training History")
        st.line_chart(pd.DataFrame({"Episode Return": _hist["reward"], "Loss": _hist["loss"]}))
        st.caption(f"Final 100-episode success rate: **{np.mean(_hist['success'][-100:])*100:.1f}%**")


# ===========================================================================
# TAB 3 -- EVALUATION BENCHMARK
# ===========================================================================
with tab3:
    st.markdown('<div class="section-title">Rigorous Empirical Evaluation -- Untrained vs Trained Agent</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="theory-box">
        Compare an <b>untrained random-policy baseline</b> against the <b>trained Double DQN agent</b>
        under identical warehouse conditions. Metrics: delivery success rate, mean cumulative return,
        episode length, and collision frequency.
        </div>""",
        unsafe_allow_html=True,
    )
    eval_episodes = st.slider("Number of Evaluation Episodes", 20, 300, 100, step=10)
    if st.button("Run Evaluation Benchmark", type="primary"):
        with st.spinner(f"Running {eval_episodes} episodes for each policy..."):
            _um = run_evaluation_episodes(
                st.session_state.agent, st.session_state.env,
                episodes=eval_episodes, is_untrained=True,
            )
            _tm = run_evaluation_episodes(
                st.session_state.agent, st.session_state.env,
                episodes=eval_episodes, is_untrained=False,
            )
        st.markdown("### Benchmark Results")
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Delivery Success Rate", f"{_tm['success_rate']:.1f}%",
                   delta=f"+{_tm['success_rate'] - _um['success_rate']:.1f}% vs Baseline")
        _m2.metric("Mean Cumulative Return", f"{_tm['mean_return']:.1f}",
                   delta=f"{_tm['mean_return'] - _um['mean_return']:.1f}")
        _m3.metric("Avg Steps to Goal", f"{_tm['mean_steps']:.1f}",
                   delta=f"{_tm['mean_steps'] - _um['mean_steps']:.1f}", delta_color="inverse")
        _m4.metric("Avg Collisions", f"{_tm['mean_collisions']:.2f}",
                   delta=f"{_tm['mean_collisions'] - _um['mean_collisions']:.2f}", delta_color="inverse")
        _cdf = pd.DataFrame({
            "Metric": ["Delivery Success Rate (%)", "Mean Episode Return",
                       "Std Dev Return", "Mean Steps / Episode", "Mean Collisions / Episode"],
            "Untrained Baseline (Random Policy)": [
                f"{_um['success_rate']:.1f}%", f"{_um['mean_return']:.2f}",
                f"{_um['std_return']:.2f}", f"{_um['mean_steps']:.1f}", f"{_um['mean_collisions']:.2f}"],
            "Trained Double DQN Agent": [
                f"{_tm['success_rate']:.1f}%", f"{_tm['mean_return']:.2f}",
                f"{_tm['std_return']:.2f}", f"{_tm['mean_steps']:.1f}", f"{_tm['mean_collisions']:.2f}"],
        })
        st.table(_cdf)

    st.markdown("---")
    st.markdown("### Evidence Visualizations")
    _ev1, _ev2 = st.columns(2)
    if os.path.exists("submission/evidence/training_curves.png"):
        _ev1.image("submission/evidence/training_curves.png", caption="DQN Training Convergence Curves")
    if os.path.exists("submission/evidence/evaluation_comparison.png"):
        _ev2.image("submission/evidence/evaluation_comparison.png", caption="Benchmark: Random vs DQN")


# ===========================================================================
# TAB 4 -- BELLMAN THEORY EXPLORER
# ===========================================================================
with tab4:
    st.markdown('<div class="section-title">Bellman Equation & Optimality Theory Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="theory-box">
        <b>Formal MDP:</b> M = (S, A, P, R, gamma) -- discounted infinite-horizon.<br>
        <b>State Space S:</b> (row, col, has_package) for all non-obstacle cells x {0,1}.<br>
        <b>Action Space A:</b> {UP, DOWN, LEFT, RIGHT} -- 4 deterministic actions.<br>
        <b>Rewards:</b> Delivery +100, Pickup +25, Step -1, Obstacle -10, Boundary -5, Timeout -20.
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""
        ### 1. Bellman Expectation Equation
        $$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

        ### 2. Bellman Optimality Equation
        $$V^*(s) = \max_{a} \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$
        $$Q^*(s, a) = R(s, a) + \gamma \max_{a'} Q^*(s', a')$$

        ### 3. Double DQN Corrected TD-Target
        $$y_t^{\text{DDQN}} = r_t + (1-d_t)\,\gamma\,Q\!\left(s_{t+1}, \arg\max_{a'} Q(s_{t+1},a';\theta);\,\theta^-\right)$$
        """
    )
    st.markdown("---")
    st.subheader("Interactive Worked Numerical Example -- Action Comparison")
    _az3 = st.session_state.bellman_analyzer
    _coords = list(_az3.env.free_cells)
    _cs1, _cs2 = st.columns(2)
    _sel_coord = _cs1.selectbox("Coordinate (row, col)", _coords, index=0)
    _sel_pkg   = _cs2.selectbox("Package Status", [0, 1],
                                format_func=lambda x: "0 - Seeking Pickup" if x == 0 else "1 - Carrying Package")
    _tstate = (_sel_coord[0], _sel_coord[1], _sel_pkg)
    _ca1, _ca2 = st.columns(2)
    _act_a = _ca1.selectbox("Action A", [0,1,2,3], index=1, format_func=lambda x: ACTION_NAMES[x])
    _act_b = _ca2.selectbox("Action B", [0,1,2,3], index=0, format_func=lambda x: ACTION_NAMES[x])
    _obs_t = _az3.env.get_observation(_tstate)
    _nn_q  = st.session_state.agent.get_q_values(_obs_t)
    _bd    = _az3.compare_actions_numerical_example(_tstate, _act_a, _act_b, nn_agent_q_values=_nn_q)
    _ra, _rb = _bd["action_A"], _bd["action_B"]

    _bc1, _bc2 = st.columns(2)
    with _bc1:
        st.markdown(f"#### Action A: **{_ra['action_name']}**")
        st.markdown(f"""
| Field | Value |
|---|---|
| Next State s' | `{_ra['next_state']}` (Event: `{_ra['event']}`) |
| Immediate Reward R(s,a) | `{_ra['immediate_reward']:.2f}` |
| V*(s') Next-State Optimal Value | `{_ra['next_state_value_V']:.2f}` |
| Discount Factor gamma | `{_ra['gamma']:.2f}` |
| Discounted Future | `{_ra['discounted_future']:.2f}` |
| **Exact Bellman Q*(s,a)** | **`{_ra['action_value_Q']:.2f}`** |
| Neural Network Q-hat | `{_bd['nn_predictions']['Q_pred_A']:.2f}` |
| Approx Error | `{_bd['nn_predictions']['nn_error_A']:.2f}` |
""")
        st.markdown(f"$$Q^*(s,a) = {_ra['immediate_reward']:.2f} + ({_ra['gamma']:.2f} \\times {_ra['next_state_value_V']:.2f}) = \\mathbf{{{_ra['action_value_Q']:.2f}}}$$")

    with _bc2:
        st.markdown(f"#### Action B: **{_rb['action_name']}**")
        st.markdown(f"""
| Field | Value |
|---|---|
| Next State s' | `{_rb['next_state']}` (Event: `{_rb['event']}`) |
| Immediate Reward R(s,a) | `{_rb['immediate_reward']:.2f}` |
| V*(s') Next-State Optimal Value | `{_rb['next_state_value_V']:.2f}` |
| Discount Factor gamma | `{_rb['gamma']:.2f}` |
| Discounted Future | `{_rb['discounted_future']:.2f}` |
| **Exact Bellman Q*(s,a)** | **`{_rb['action_value_Q']:.2f}`** |
| Neural Network Q-hat | `{_bd['nn_predictions']['Q_pred_B']:.2f}` |
| Approx Error | `{_bd['nn_predictions']['nn_error_B']:.2f}` |
""")
        st.markdown(f"$$Q^*(s,a) = {_rb['immediate_reward']:.2f} + ({_rb['gamma']:.2f} \\times {_rb['next_state_value_V']:.2f}) = \\mathbf{{{_rb['action_value_Q']:.2f}}}$$")

    st.info(
        f"Bellman Decision Rule: Between {_ra['action_name']} and {_rb['action_name']}, "
        f"the criterion selects **{_bd['best_action_between_two']}** "
        f"with an advantage of **{abs(_bd['advantage_A_over_B']):.3f}** return units."
    )
    if os.path.exists("submission/evidence/bellman_value_heatmap.png"):
        st.image("submission/evidence/bellman_value_heatmap.png",
                 caption="Exact Bellman Optimal State-Value Heatmap V*(s)")


# ===========================================================================
# TAB 5 -- RL DEBUGGING ANALYSIS
# ===========================================================================
with tab5:
    st.markdown(
        '<div class="section-title">Advanced RL Debugging Challenge -- Root-Cause Analysis & Algorithmic Fixes</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="theory-box">
        Documents <b>confirmed algorithmic and implementation pathologies</b> in Deep RL and
        demonstrates how systematic debugging -- via Double DQN, exploration warm-starting,
        terminal state masking, and gradient update throttling -- achieves a
        <b>100% delivery success rate</b>.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Confirmed Error Classification Matrix")
    _bugs = pd.DataFrame([
        {"Bug ID": "BUG-01", "Category": "Algorithmic Pathology",
         "Component": "Exploration Schedule",
         "Faulty Behavior": "Premature epsilon-decay without warm-start",
         "Consequence": "0% Success - Safe-Loop Local Optimum",
         "Resolution": "Replay buffer guided exploration seeding"},
        {"Bug ID": "BUG-02", "Category": "Algorithmic Pathology",
         "Component": "Bellman TD-Target",
         "Faulty Behavior": "Single theta- for both select and evaluate",
         "Consequence": "Overestimation bias & high gradient variance",
         "Resolution": "Double DQN - decouple selection from evaluation"},
        {"Bug ID": "BUG-03", "Category": "Implementation Defect",
         "Component": "Training Loop",
         "Faulty Behavior": "Backprop on every single transition",
         "Consequence": "Latency >290s & sample correlation",
         "Resolution": "Gradient update throttling every k=2 steps"},
        {"Bug ID": "BUG-04", "Category": "Implementation Defect",
         "Component": "Terminal State Masking",
         "Faulty Behavior": "Bootstrapping Q at terminal state",
         "Consequence": "Phantom infinite-horizon value accumulation",
         "Resolution": "Strict (1 - done) terminal mask in TD target"},
    ])
    st.table(_bugs)

    st.markdown("---")
    st.subheader("2. Mathematical Formulation: Faulty vs Corrected Bellman Targets")
    st.markdown(
        r"""
        **Faulty Standard DQN:**
        $$y_t^{\text{DQN}} = r_t + \gamma \max_{a'} Q(s_{t+1}, a';\,\theta^-)$$

        **Corrected Double DQN with Terminal Masking:**
        $$y_t^{\text{DDQN}} = r_t + (1-d_t)\,\gamma\,Q\!\left(s_{t+1}, \arg\max_{a'} Q(s_{t+1},a';\theta);\,\theta^-\right)$$
        """
    )

    st.markdown("---")
    st.subheader("3. Empirical Debugging Benchmark: 100 Test Episodes")
    _sp = "submission_debugging/results/debugging_benchmark_summary.json"
    if os.path.exists(_sp):
        with open(_sp) as _fh:
            _ds = json.load(_fh)
        _cd = _ds["corrected_ddqn"]; _pg = _ds["performance_gain"]
        _dc1, _dc2, _dc3, _dc4 = st.columns(4)
        _dc1.metric("Success Rate (DDQN)", f"{_cd['success_rate']:.1f}%",
                    delta=f"+{_pg['success_rate_gain_pct']:.1f}% vs Faulty")
        _dc2.metric("Mean Return", f"{_cd['mean_return']:.1f}",
                    delta=f"+{_pg['return_gain']:.1f}")
        _dc3.metric("Training Time", f"{_cd['training_time_seconds']:.1f}s",
                    delta=f"{_pg['speedup_factor']:.1f}x Faster", delta_color="normal")
        _dc4.metric("Collision Rate", f"{_cd['mean_collisions']:.1f}",
                    delta=f"-{_pg['collision_reduction']:.1f}", delta_color="inverse")
    else:
        st.info("Run `python debugging_experiment/run_experiments.py` to generate benchmark results.")

    st.markdown("#### Diagnostic Evidence Visualizations")
    _dv1, _dv2 = st.columns(2)
    if os.path.exists("submission_debugging/results/debugging_learning_curves.png"):
        _dv1.image("submission_debugging/results/debugging_learning_curves.png",
                   caption="Training: Faulty DQN vs Corrected DDQN")
    if os.path.exists("submission_debugging/results/debugging_benchmark_comparison.png"):
        _dv2.image("submission_debugging/results/debugging_benchmark_comparison.png",
                   caption="Evaluation Benchmark: 100 Test Episodes")
    if os.path.exists("submission_debugging/results/td_loss_comparison.png"):
        st.image("submission_debugging/results/td_loss_comparison.png",
                 caption="TD Loss Dynamics & Exploration Schedules")

    st.markdown("---")
    st.subheader("4. Reproduce Debugging Experiments")
    st.code(
        """# Run the reproducible debugging experiment suite:
python debugging_experiment/run_experiments.py

# Run all regression unit tests:
pytest tests/test_debugging_regression.py -v

# Generate full submission PDF report:
python generate_my_report.py""",
        language="powershell",
    )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
FOOTER = """
<div class="footer-strip">
  <strong>Intelligent Warehouse Robot Navigation -- RL Decision Engine</strong><br>
  Aswin S &nbsp;|&nbsp; RA2311026050067 &nbsp;|&nbsp;
  21CSE417TA / Fundamentals of Reinforcement Learning &nbsp;|&nbsp;
  SRM Institute of Science and Technology, Tiruchirappalli -- 621 105 &nbsp;|&nbsp;
  School of Computing -- B.Tech Artificial Intelligence &amp; Machine Learning
</div>
"""
st.markdown(FOOTER, unsafe_allow_html=True)
