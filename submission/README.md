# RL Challenge: Intelligent Warehouse Robot Navigation
**Decoding Sequential Decision-Making via Finite Markov Decision Process & Deep Q-Networks (DQN)**  
*B.Tech CSE (AI & ML) — Reinforcement Learning Assignment*

---

## 1. Project Overview & Problem Statement
In automated distribution and fulfillment centers, autonomous warehouse robots must navigate a grid floor, avoid storage obstacles, collect inventory items from a **Pickup Station (P)**, and deliver them safely to a **Delivery Dock (D)** while minimizing transit time and avoiding costly collisions.

This project formulates the warehouse navigation problem as a finite **Markov Decision Process (MDP)**:
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

We implement:
1. **Finite MDP Environment (`warehouse_rl/warehouse_env.py`)**: Models a 6x6 grid with start $S(0,0)$, pickup $P(2,4)$, delivery $D(5,5)$, and static obstacles. State space $\mathcal{S} = (r, c, p)$ where $p \in \{0, 1\}$ denotes package status.
2. **Exact Dynamic Programming & Bellman Analysis (`warehouse_rl/bellman_analyzer.py`)**: Computes exact ground truth $V^*(s)$ and $Q^*(s, a)$ via Value Iteration and provides side-by-side numerical action comparisons.
3. **Deep Q-Network (DQN / Double DQN) (`warehouse_rl/dqn_agent.py`)**: PyTorch neural network function approximator with Experience Replay, Target Network, and $\epsilon$-greedy exploration decay.
4. **Interactive Prototype (`streamlit_app.py`)**: Browser-based interactive dashboard with real-time floor grid, telemetry, step-by-step manual/agent controls, live DQN training, empirical benchmarks, and Bellman theory explorer.

---

## 2. Windows Environment Setup & Installation

### Prerequisites
- Python 3.10 or 3.11 on Windows (PowerShell)
- PyTorch (CPU or CUDA compatible)

### Step 1: Clone or Open Workspace
```powershell
cd intelligent-warehouse-robot
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
pip install torch streamlit matplotlib reportlab
```

---

## 3. How to Launch the Interactive Prototype

Launch the Streamlit web dashboard:
```powershell
streamlit run streamlit_app.py
```
Open your browser to: **`http://localhost:8501`**

### Prototype Controls & Features:
- **Interactive Simulation Tab**:
  - View warehouse floor with Robot 🤖, Package 📦, Delivery 🏁, and Obstacles 🧱.
  - Click **Step Trained Agent** to step the agent and inspect live Q-values.
  - Click **Run Full Episode** to watch the autonomous robot complete the pickup and delivery mission in 10 optimal steps.
  - Manual override buttons (UP, DOWN, LEFT, RIGHT) to test collision boundaries.
- **Train DQN Agent Tab**:
  - Configure hyperparameters (episodes, learning rate, discount $\gamma$, epsilon decay).
  - Live progress bar, reward curves, and loss tracking.
- **Empirical Benchmark Tab**:
  - Compare Untrained Baseline vs Trained DQN Agent over 50-100 episodes.
- **Bellman Equation & Theory Tab**:
  - Select any state $(r, c, p)$ and compare two actions side-by-side with exact numerical Bellman calculations: $Q^*(s, a) = R(s, a) + \gamma V^*(s')$.

---

## 4. How to Train and Evaluate the Agent

### Train the DQN Model
To train the robot from scratch and save model weights to `models/dqn_warehouse_model.pth`:
```powershell
python train_agent.py
```

### Evaluate the Agent & Generate Evidence Charts
To benchmark the trained agent against the untrained baseline and generate all evidence plots in `submission/evidence/`:
```powershell
python evaluate_agent.py
```

---

## 5. How to Run Unit Tests

Execute the comprehensive test suite (42 tests covering environment dynamics, collisions, pickup/delivery, Bellman optimality, and DQN checkpoint compatibility):
```powershell
pytest -v
```

---

## 6. Project Structure

```text
intelligent-warehouse-robot/
├── warehouse_rl/                  # Core RL package
│   ├── __init__.py
│   ├── warehouse_env.py           # Finite Warehouse MDP with Pickup & Delivery
│   ├── dqn_agent.py               # PyTorch DQN / Double DQN with Experience Replay
│   └── bellman_analyzer.py        # Tabular Bellman Solver & Numerical Analyzer
├── models/
│   ├── dqn_warehouse_model.pth    # Saved PyTorch model checkpoint
│   └── training_history.json      # Genuine recorded training loss and returns
├── submission/                    # Google Classroom Deliverables
│   ├── RL_Challenge_Report.pdf    # Comprehensive 5-page academic project report
│   ├── RL_Challenge_Source_Code.zip # Clean submission source ZIP
│   ├── README.md                  # Complete project documentation
│   ├── DEMO_SCRIPT.md             # 5-7 minute oral presentation script
│   └── evidence/                  # Real experimental evidence & screenshots
│       ├── training_curves.png
│       ├── evaluation_comparison.png
│       ├── bellman_value_heatmap.png
│       ├── evaluation_metrics.json
│       ├── prototype_full_episode.png
│       ├── prototype_bellman_tab.png
│       └── prototype_benchmark_tab.png
├── tests/
│   ├── test_rl_warehouse.py       # Unit tests for MDP, Bellman, and DQN
│   ├── test_mdp.py                # Regression tests for original MDP
│   └── test_async_dp.py          # Regression tests for DP solvers
├── streamlit_app.py               # Interactive Streamlit Prototype
├── train_agent.py                 # DQN training script
├── evaluate_agent.py              # Empirical evaluation & plotting script
├── generate_report_pdf.py         # ReportLab PDF compiler
├── main.py                        # Original FastAPI application (preserved)
├── mdp.py                         # Original DP MDP (preserved)
├── async_dp.py                    # Original Async VI solver (preserved)
└── requirements.txt               # Dependencies
```

---

## 7. Measured Performance Summary

| Metric | Untrained Baseline (Random Policy) | Trained DQN Agent | RL Improvement |
|---|---|---|---|
| **Delivery Success Rate** | 8.0% | **100.0%** | **+92.0%** |
| **Mean Cumulative Return** | -243.5 ± 45.2 | **+115.0 ± 0.0** | **+358.5 return units** |
| **Mean Steps per Episode** | 49.2 | **10.0** (Optimal) | **-39.2 steps** |
| **Obstacle/Boundary Collisions** | 28.2 | **0.0** | **100% collision-free** |

---

## 8. Known Limitations & Future Work
- **Static Grid Assumption**: The warehouse layout is static; dynamic moving obstacles (workers or other AMRs) will require Multi-Agent Reinforcement Learning (MARL).
- **Deterministic Transitions**: Real robot slippage requires stochastic slip modeling ($\mathcal{P}(s'|s,a)$ with noise).
- **Sim-to-Real Transfer**: Future work involves interfacing this MDP with ROS 2 and Gazebo for physical differential-drive AMRs.
