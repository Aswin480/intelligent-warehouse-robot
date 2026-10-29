# Intelligent Warehouse Robot Navigation Using Asynchronous Dynamic Programming

## Project Overview

This browser-based simulation models a robot navigating a finite warehouse grid. It demonstrates how a warehouse MDP becomes a value function, a greedy policy, and an executable route using **in-place asynchronous value iteration**.

This is a local simulation, not a physical warehouse robot. It does not require ROS 2, Gazebo, hardware, Docker, a database, or cloud services.

## Problem and MDP

The robot travels from `S` to `G` while avoiding `#` obstacles. The simulator generates warehouses, runs asynchronous and synchronous dynamic programming on the same map, shows values and policy arrows, and animates the selected route.

- **State:** `s = (row, column)` for every non-obstacle cell.
- **Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`.
- **Transition:** valid moves enter a neighboring free cell. Boundary and obstacle moves stay in place. The goal is terminal and absorbing.
- **Rewards:** goal `+100`, movement `-1`, obstacle collision `-10`, boundary collision `-5`.
- **Discount:** `gamma = 0.9`, configured in `mdp.py`.

For a deterministic transition, the Bellman update is:

`V(s) = max_a [ R(s,a) + gamma * V(next_state(s,a)) ]`

### Asynchronous Dynamic Programming

`async_dp.py` updates states in row-major order and immediately writes each result into the live value map. Later states in the same sweep can use freshly updated values. Here, “asynchronous” means in-place dynamic programming; it does **not** mean Python `async`/`await`, `asyncio`, threads, or concurrent execution.

The synchronous baseline copies the value map for each sweep. Both use `theta = 0.001` and at most `1000` iterations. The UI reports actual iterations, state updates, runtime, convergence, path length, and total reward; it does not assume one algorithm is always faster.

## Architecture

```text
main.py                    FastAPI routes and solvable random grids
mdp.py                     Warehouse MDP, rewards, transitions, terminal state
async_dp.py                Async VI, synchronous VI, policies, paths
static/index.html          Browser dashboard
static/style.css           Dashboard styling
static/script.js           API calls, grid rendering, animation
tests/test_mdp.py          Original Gridworld regression tests
tests/test_async_dp.py     Warehouse solver and path tests
tests/test_warehouse_mdp.py Warehouse MDP tests
gridworld/                 Original pure-Python MDP foundation retained
```

The original `gridworld/` package and tests remain intact. Its general NumPy tensor and stochastic slip model are useful foundations, while the final simulator uses the smaller deterministic `WarehouseMDP` so the academic demonstration is easy to explain.

## Installation and Running

From Windows PowerShell in this folder:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`, and run tests with `pytest -q`.

## API

- `GET /` serves the dashboard.
- `POST /api/generate` accepts optional `rows`, `cols`, `density`, and `seed`.
- `POST /api/run/async-dp` runs in-place asynchronous value iteration.
- `POST /api/run/value-iteration` runs the synchronous baseline.
- `POST /api/path` accepts `{"algorithm":"async"}` or `{"algorithm":"sync"}`.
- `GET /api/compare` runs both algorithms on the current warehouse.
- `POST /api/reset` restores the default 10 x 10 warehouse.

## Experiment, Limitations, and Future Work

The comparison table measures both algorithms on the same generated warehouse. Results depend on map layout, state order, and machine timing. The current environment is deterministic, single-robot, and static; it does not model inventory, moving workers, multiple robots, uncertainty, or physical motion.

Possible extensions include stochastic transitions, selectable state orders, warehouse tasks and shelves, value-propagation playback, multiple robots, and a physical-robot adapter.

## Features

- **Asynchronous Value Iteration**: Real-time in-place Bellman updates.
- **Deep Q-Learning (DQN)**: Offline and online training for dynamic warehouse navigation.
- **Interactive Streamlit App**: Visualizes Bellman value propagation, policy vectors, and trajectory metrics.
- **FastAPI Backend**: REST endpoints for simulation runs and algorithm benchmarking.
