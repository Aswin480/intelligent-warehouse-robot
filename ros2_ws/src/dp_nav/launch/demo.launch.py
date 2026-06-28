# =============================================================================
# ROBOT DECISION (Task 9 Step 2) — READ BEFORE EDITING
# =============================================================================
# DEFAULT: fallback diff-drive robot (diff_drive.sdf) — distro-proof, works
# on any ROS2/Gazebo Harmonic setup without extra packages.
#
# TO SWAP IN TURTLEBOT3 (once you confirm TB3 packages exist on Kilted):
#   1. Inside the container, run:
#        apt-cache search ros-kilted-turtlebot3
#        ros2 pkg list | grep -i turtlebot3
#      If "TB3 NOT INSTALLED", stay with the fallback below.
#   2. If TB3 IS available:
#        sudo apt install ros-kilted-turtlebot3 ros-kilted-turtlebot3-gazebo
#        export TURTLEBOT3_MODEL=burger
#   3. In generate_launch_description() below:
#      a. Remove the `gz` and `spawn` actions.
#      b. Replace them with the TB3 gazebo launch include, e.g.:
#           from launch.actions import IncludeLaunchDescription
#           from launch.launch_description_sources import PythonLaunchDescriptionSource
#           from ament_index_python.packages import get_package_share_directory
#           tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
#           tb3_launch = IncludeLaunchDescription(
#               PythonLaunchDescriptionSource(
#                   os.path.join(tb3_gazebo, "launch", "turtlebot3_world.launch.py")
#               ),
#               launch_arguments={
#                   "x_pose": "0.5",
#                   "y_pose": "-5.5",   # cell_to_world(5, 0) = (0.5, -5.5)
#               }.items(),
#           )
#      c. Add `tb3_launch` to the LaunchDescription list (keep the `bridge`).
#      NOTE: if the TB3 path spawns the robot via a separate spawn/create
#      action (rather than the gazebo include doing it), wrap that spawn in
#      `TimerAction(period=3.0, actions=[...])` too, so it waits for gz sim's
#      create service to come up (same silently-dropped-spawn gotcha).
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

# EDIT this to your mounted repo path inside the container if different:
REPO_ROOT = "/workspace/rl-explorations"
SCENARIO = os.path.join(REPO_ROOT, "scenarios", "maze.yaml")


def generate_launch_description():
    share = get_package_share_directory("dp_nav")
    world = os.path.join(share, "worlds", "maze.sdf")
    robot = os.path.join(share, "worlds", "diff_drive.sdf")  # fallback robot

    gz = ExecuteProcess(cmd=["gz", "sim", "-r", world], output="screen")

    spawn = Node(
        package="ros_gz_sim", executable="create", output="screen",
        arguments=["-file", robot, "-name", "dpbot", "-x", "0.5", "-y", "-5.5", "-z", "0.1"],
    )

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
        ],
    )

    planner = Node(
        package="dp_nav", executable="planner_node", output="screen",
        parameters=[{"scenario": SCENARIO}],
    )

    # Delay spawn so gz sim's create service is up (avoids silently-dropped spawns).
    # Delay planner so robot is up (5s > spawn's 3s). IMPORTANT: set REPO_ROOT above to match container mount.
    return LaunchDescription([gz, TimerAction(period=3.0, actions=[spawn]), bridge, TimerAction(period=5.0, actions=[planner])])
