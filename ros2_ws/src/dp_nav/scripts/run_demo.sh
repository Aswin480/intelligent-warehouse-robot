#!/usr/bin/env bash
# Run the DP-navigation demo INSIDE the container (after setup_container.sh).
# Launches Gazebo + the TurtleBot3, bridges /cmd_vel and /odom, and starts the
# DP planner. The robot computes V*/pi* by value iteration and drives the policy
# cell-to-cell to the goal.
#
#   GUI=1 (default) shows the Gazebo window (needs XQuartz + DISPLAY) — use this
#         to screen-record the demo. GUI=0 runs headless (CI / quick check).
#
# Ground-truth odometry is ABSOLUTE world pose, so the planner runs with
# spawn_relative_odom:=false.
set -eo pipefail   # not -u: ROS setup.bash references unbound vars

REPO=${REPO:-/workspace}
EXTRA_WS=${EXTRA_WS:-/ros2_extra_ws}
TB3=${TB3:-$EXTRA_WS/src/turtlebot3_simulations}
GUI=${GUI:-1}

source /opt/ros/kilted/setup.bash
source "$EXTRA_WS/install/setup.bash"
source /tmp/dpnav/install/setup.bash
export PATH=/usr/bin:$PATH
export PYTHONPATH="$REPO:$REPO/ros2_ws/src:${PYTHONPATH:-}"
export GZ_SIM_RESOURCE_PATH="$TB3/turtlebot3_gazebo/models"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}   # isolate from other ROS sessions if set

WORLD="$REPO/ros2_ws/src/dp_nav/worlds/maze.sdf"
SCENARIO="$REPO/scenarios/maze.yaml"
# spawn pose = world coords of the scenario's start cell
read -r SX SY < <(python3 -c "from gridworld.scenario import load_scenario as L; from gridworld.grid_coords import cell_to_world as W; s=L('$SCENARIO'); x,y=W(s.start[0],s.start[1],s.coords); print(x,y)")

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

GZFLAGS="-r -v2"; [ "$GUI" = "0" ] && GZFLAGS="-s -r -v2"
gz sim $GZFLAGS "$WORLD" & pids+=($!)

echo ">>> waiting for the maze world..."
for _ in $(seq 1 30); do gz topic -l 2>/dev/null | grep -q /world/maze && break; sleep 1; done

echo ">>> spawning TurtleBot3 at ($SX, $SY)"
ros2 run ros_gz_sim create -world maze -file /tmp/burger_gt.sdf -name burger -x "$SX" -y "$SY" -z 0.01

ros2 run ros_gz_bridge parameter_bridge \
  "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist" \
  "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry" & pids+=($!)
sleep 2

echo ">>> starting DP planner (spawn_relative_odom:=false for ground-truth odom)"
python3 "$REPO/ros2_ws/src/dp_nav/dp_nav/planner_node.py" --ros-args \
  -p scenario:="$SCENARIO" -p spawn_relative_odom:=false & pids+=($!)

echo ">>> running. Ctrl-C to stop."
wait "${pids[0]}"   # block on gz; cleanup kills bridge+planner on exit
