# dp_nav — DP gridworld navigation for a Gazebo robot

A TurtleBot3 in a Gazebo maze drives a path computed entirely by **Dynamic
Programming** (value iteration / policy iteration over a discretized gridworld
MDP — Sutton & Barto Ch. 4). The pure-Python DP core lives in `gridworld/` at
the repo root; this ROS 2 package wraps it: `planner_node` reads `/odom`, looks
up π\*(current cell), and publishes `/cmd_vel`.

## Verified environment

Confirmed working **headless** in a container with **ROS 2 Kilted + Gazebo
(gz) 9.5.0** (the `ros_gz` vendor gz), Apple-silicon `linux/arm64`. The robot
followed the DP policy cell-to-cell — `(5,0)→E→(5,5)→N→(3,5)→E→(3,6)→N→(0,6)` —
and reached the goal, tracking each lane centerline with no wall contact.

## Run it

Inside the container, with this repo mounted at `/workspace`:

```bash
# one-time: build ros_gz (no Kilted apt binaries yet), fetch TB3, build dp_nav
ros2_ws/src/dp_nav/scripts/setup_container.sh

# run the demo (GUI=1 shows Gazebo for screen-recording; GUI=0 is headless)
GUI=1 ros2_ws/src/dp_nav/scripts/run_demo.sh
```

To **record the LinkedIn clip**: start the container with XQuartz/DISPLAY
forwarding (same pattern as the ArduPilot repo's `run-container.sh`), run with
`GUI=1`, and screen-record the Gazebo window as the robot solves the maze.
Pair it with the `viz/` heatmap-convergence GIF and the VI-vs-PI figure.

## Things the verification pinned down (don't relearn them)

- **`ros_gz` is built from source** (`kilted` branch) — there are no Kilted apt
  binaries yet. `rosdep` pulls the `gz-{sim,transport,msgs}-vendor` packages,
  which provide **gz 9.5.0**; run the gz *server* from the same (vendored) gz so
  `create`/`bridge` interoperate. Build with system `/usr/bin/python3`, not any
  active ArduPilot venv (it lacks `catkin_pkg`).
- **Ground-truth odometry, not wheel odometry.** The stock burger's DiffDrive
  odom dead-reckons and drifts in *yaw* during in-place turns (wheel slip),
  which steered the planner diagonally into walls. `make_burger_gt.py` swaps in a
  gz `OdometryPublisher` (true sim pose) on `/odom`. It reports **absolute**
  world pose, so run the planner with **`spawn_relative_odom:=false`**.
  (With real wheel odom, set it `:=true` to add the spawn-cell offset.)
- **2 m cells** (`scenarios/maze.yaml`) give the small burger generous lane
  clearance; the cell-follower turns in place at cell centers so motion stays
  axis-aligned.
- The `planner_node` `scenario`/`spawn_relative_odom` parameters and the
  `cell_follower` `FollowerParams` (speed, tolerances) are the knobs to tune.
