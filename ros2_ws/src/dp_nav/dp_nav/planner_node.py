"""ROS2 node: /odom -> DP policy lookup -> /cmd_vel. Thin adapter over Follower."""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from gridworld.scenario import load_scenario
from gridworld.grid_coords import cell_to_world
from dp_nav.cell_follower import Follower


def _yaw_from_quaternion(q):
    # yaw (z-axis rotation) from a geometry_msgs Quaternion
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class PlannerNode(Node):
    def __init__(self):
        super().__init__("planner_node")
        self.declare_parameter("scenario", "scenarios/maze.yaml")
        # gz DiffDrive odom is spawn-relative (starts at (0,0) at the spawn pose),
        # so we add the spawn world-position to recover true world coordinates.
        # If your odom source already publishes world/ground-truth pose, set
        # spawn_relative_odom:=false on launch.
        # Symptom: if the robot immediately heads toward the wrong corner, flip this param.
        self.declare_parameter("spawn_relative_odom", True)
        path = self.get_parameter("scenario").get_parameter_value().string_value
        scenario = load_scenario(path)
        spawn_relative = self.get_parameter("spawn_relative_odom").get_parameter_value().bool_value
        if spawn_relative:
            offset = cell_to_world(scenario.start[0], scenario.start[1], scenario.coords)
        else:
            offset = (0.0, 0.0)
        self.follower = Follower(scenario, world_offset=offset)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self._reached = False
        self.get_logger().info(f"planner_node up; scenario={path}")

    def _on_odom(self, msg: Odometry):
        if self._reached:
            self.pub.publish(Twist())
            return
        p = msg.pose.pose
        pose = (p.position.x, p.position.y, _yaw_from_quaternion(p.orientation))
        lin, ang, reached = self.follower.step(pose)
        cmd = Twist()
        cmd.linear.x = float(lin)
        cmd.angular.z = float(ang)
        self.pub.publish(cmd)
        if reached:
            self._reached = True
            self.get_logger().info("Goal reached — holding position.")


def main():
    rclpy.init()
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
