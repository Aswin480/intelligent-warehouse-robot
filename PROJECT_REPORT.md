# Project Report

## Title

**Intelligent Warehouse Robot Navigation Using Asynchronous Dynamic Programming**

## One-Minute Explanation

This project is a browser-based warehouse robot simulation. The warehouse is represented as a finite Markov Decision Process (MDP). Each free grid cell is a state, and the robot can move up, down, left, or right. The robot receives a large positive reward for reaching the goal, a small negative reward for normal movement, and larger penalties for hitting an obstacle or boundary.

The main algorithm is Asynchronous Value Iteration. It applies the Bellman optimality equation one state at a time and immediately stores each new value. Therefore, later states in the same row-major sweep can use updated information. Synchronous Value Iteration is retained as a baseline: it calculates a complete new table from the previous table and replaces the table after the sweep.

## Problem and Objectives

The robot must travel from `S` to `G` while avoiding `#` obstacles. The objectives are to model this as an MDP, implement in-place asynchronous Bellman updates, compare them with synchronous Value Iteration, extract a greedy policy, and visualize the path in a browser.

This is a local simulation, not physical warehouse hardware. It requires no ROS 2, Gazebo, Docker, database, cloud service, or deep reinforcement learning model.

## MDP Definition

- **State:** `s = (row, column)` for every non-obstacle cell.
- **Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`.
- **Transition:** a free neighbor is entered; an obstacle or boundary leaves the robot in place; the goal is terminal and absorbing.
- **Rewards:** goal `+100`, normal movement `-1`, obstacle collision `-10`, boundary collision `-5`.
- **Discount:** `gamma = 0.9`.

For deterministic movement:

`V(s) = max_a [ R(s,a) + gamma * V(next_state(s,a)) ]`

The greedy policy is:

`pi(s) = argmax_a [ R(s,a) + gamma * V(next_state(s,a)) ]`

## Synchronous Versus Asynchronous

### Synchronous Value Iteration

Synchronous VI reads the old value table for every state and writes a separate new table:

```text
old V -> calculate every state -> new V -> replace old V
```

No state can use a value calculated earlier in the same sweep.

### Asynchronous Value Iteration

Asynchronous VI uses row-major order and writes each state immediately:

```text
update A -> save immediately
update B using current values, including new A
update C using current values, including new B
```

“Asynchronous” here is an algorithmic term. It does not mean Python `async/await`, `asyncio`, threads, or parallel execution.

The implementation in `async_dp.py` is a separate in-place solver. It does not call synchronous VI and rename it.

## Why Comparison Values Can Be Identical

Both algorithms solve the same deterministic MDP. Therefore, identical final policy, path length, total reward, goal result, and even iteration count are valid and expected. The algorithms are still different because their update mechanisms differ.

- **Iterations:** complete Bellman sweeps until convergence.
- **State Updates:** individual non-terminal cell updates. In this implementation, this is usually `non_terminal_states x iterations`, so it can match for both algorithms.
- **Execution Time:** measured wall-clock runtime on the current computer. It can change from run to run, and asynchronous DP is not always guaranteed to be faster.
- **Max Delta:** largest value change during the last sweep. Convergence occurs when it is below `0.001`.
- **Path Length:** number of transitions from start to goal.
- **Total Reward:** sum of actual transition rewards along the generated path.
- **Goal Reached:** whether the generated policy reached the terminal goal safely.

The comparison table also shows the important implementation difference: `Copied table` for synchronous VI and `In-place table` for asynchronous VI.

## Demonstration Workflow

1. Start the server: `python -m uvicorn main:app --reload`.
2. Open `http://127.0.0.1:8000`.
3. Try solving the puzzle with arrow keys.
4. Click **Run Async DP**. Policy arrows and async metrics appear; the route remains hidden.
5. Click **Run Synchronous VI**. Synchronous policy arrows and metrics appear.
6. Click **Compare both** for measured comparison on the same warehouse.
7. Click **Show Solution** to reveal the calculated route.
8. Click **Animate Solution** to animate the robot.
9. Click **Values** to inspect the value function.

## Architecture

```text
main.py                    FastAPI routes and solvable random grids
mdp.py                     Warehouse states, actions, transitions, rewards
async_dp.py                Async VI, Sync VI, policy, path, comparison
static/index.html          Browser dashboard
static/style.css           Dashboard styling
static/script.js           Browser interaction and animation
gridworld/                 Original reusable pure-Python MDP foundation
tests/                     Original and warehouse-specific tests
```

## API Endpoints

- `GET /`: serve dashboard.
- `POST /api/generate`: generate a solvable warehouse.
- `POST /api/run/async-dp`: run asynchronous VI.
- `POST /api/run/value-iteration`: run synchronous VI.
- `POST /api/path`: return a policy path.
- `GET /api/compare`: run and compare both algorithms.
- `POST /api/reset`: restore the default warehouse.

## Testing and Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
pytest -q
python -m uvicorn main:app --reload
```

The verified test suite contains 33 passing tests covering MDP behavior, rewards, terminal handling, convergence, finite values, policies, paths, and comparison consistency.

## Unwanted Files Removed

The original `robot_run.gif`, `value_iteration.gif`, and `vi_vs_pi.png` were removed because the browser dashboard is now the primary visualization. The reusable `gridworld/` code and tests were retained.

## Limitations and Future Work

The current system is deterministic, single-robot, static, and in-memory. It does not model inventory, moving workers, multiple robots, sensor uncertainty, physical dynamics, or hardware. Future work could add stochastic transitions, selectable update orders, value-propagation playback, task assignment, dynamic obstacles, and a ROS or hardware adapter.

## Final Viva Answer

“The project models warehouse navigation as an MDP and uses asynchronous value iteration as its main contribution. Synchronous value iteration creates a new table from the old table for each sweep, while asynchronous value iteration updates the table in place and immediately reuses fresh values. Both are evaluated on the same warehouse, so identical final paths and rewards can be correct. The meaningful difference is the update rule; runtime and convergence are measured experimentally.”
