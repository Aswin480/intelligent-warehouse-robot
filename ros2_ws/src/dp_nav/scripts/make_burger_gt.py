#!/usr/bin/env python3
"""Adapt the stock TurtleBot3 burger gz model for the DP-navigation demo.

Three edits to the upstream `turtlebot3_burger/model.sdf`:
  1. Strip the IMU + gpu_lidar <sensor> blocks — the demo doesn't use them and
     gpu_lidar needs GPU rendering we don't have in a headless container.
  2. Repoint the DiffDrive plugin's odometry to `wheel_odom` so it no longer
     owns `/odom`.
  3. Add a gz OdometryPublisher on `/odom` that reports the robot's *true* sim
     pose. Wheel-dead-reckoned odom drifts in yaw during in-place turns (wheel
     slip), which sends the planner diagonally into walls; ground-truth pose is
     drift-free and gives the planner perfect localization — appropriate for a
     planning demo. NOTE: OdometryPublisher reports ABSOLUTE world pose, so run
     planner_node with `spawn_relative_odom:=false`.

Usage: make_burger_gt.py <upstream model.sdf> <output burger_gt.sdf>
"""
import sys
import xml.etree.ElementTree as ET


def main():
    src, dst = sys.argv[1], sys.argv[2]
    tree = ET.parse(src)
    root = tree.getroot()
    parents = {c: p for p in root.iter() for c in p}

    for sensor in root.findall(".//sensor"):
        parents[sensor].remove(sensor)

    for plugin in root.findall(".//plugin"):
        if "DiffDrive" in plugin.get("name", ""):
            ot = plugin.find("odom_topic")
            if ot is None:
                ot = ET.SubElement(plugin, "odom_topic")
            ot.text = "wheel_odom"

    model = root.find(".//model")
    op = ET.SubElement(model, "plugin")
    op.set("filename", "gz-sim-odometry-publisher-system")
    op.set("name", "gz::sim::systems::OdometryPublisher")
    for tag, val in [("odom_topic", "odom"), ("odom_frame", "odom"),
                     ("robot_base_frame", "base_footprint"),
                     ("dimensions", "2"), ("odom_publish_frequency", "30")]:
        ET.SubElement(op, tag).text = val

    tree.write(dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
