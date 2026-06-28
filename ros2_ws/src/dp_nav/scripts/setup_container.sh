#!/usr/bin/env bash
# One-time setup INSIDE a ROS 2 Kilted + Gazebo (gz Ionic/Jetty) container with
# this repo mounted (default /workspace). Builds the pieces that have no Kilted
# apt binaries yet, and prepares the TurtleBot3 model. Idempotent: re-running
# skips work that's already done.
#
#   ros_gz (kilted) ........ built from source -> $EXTRA_WS/install
#   turtlebot3 burger ...... cloned + ground-truth OdometryPublisher -> /tmp/burger_gt.sdf
#   dp_nav ................. built -> /tmp/dpnav/install
#
# Verified in gz-ardupilot:dev (ROS 2 Kilted, gz 9.5.0 vendor for ros_gz).
set -eo pipefail   # not -u: ROS setup.bash references unbound vars

REPO=${REPO:-/workspace}
EXTRA_WS=${EXTRA_WS:-/ros2_extra_ws}
TB3=${TB3:-$EXTRA_WS/src/turtlebot3_simulations}

source /opt/ros/kilted/setup.bash
export GZ_VERSION=jetty
export PATH=/usr/bin:$PATH        # system python3 (the ArduPilot venv lacks ament deps)

# 1) ros_gz from source (no Kilted binaries published yet)
if [ ! -d "$EXTRA_WS/install/ros_gz_sim" ]; then
  echo ">>> building ros_gz (kilted) from source"
  mkdir -p "$EXTRA_WS/src"; cd "$EXTRA_WS/src"
  [ -d ros_gz ] || git clone -b kilted https://github.com/gazebosim/ros_gz.git
  cd "$EXTRA_WS"
  rosdep install --from-paths src/ros_gz --ignore-src -y --rosdistro kilted
  colcon build --packages-up-to ros_gz_sim ros_gz_bridge \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
else
  echo ">>> ros_gz already built"
fi

# 2) TurtleBot3 burger model + ground-truth odometry
[ -d "$TB3" ] || git clone -b main https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git "$TB3"
python3 "$REPO/ros2_ws/src/dp_nav/scripts/make_burger_gt.py" \
  "$TB3/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf" /tmp/burger_gt.sdf

# 3) dp_nav (artifacts in /tmp, not the mounted repo)
source "$EXTRA_WS/install/setup.bash"
colcon build --paths "$REPO/ros2_ws/src/dp_nav" \
  --build-base /tmp/dpnav/build --install-base /tmp/dpnav/install

echo ">>> setup complete. Run the demo with: $REPO/ros2_ws/src/dp_nav/scripts/run_demo.sh"
