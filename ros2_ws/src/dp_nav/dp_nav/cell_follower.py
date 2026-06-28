"""Pure control logic: map a pose to a /cmd_vel command following a DP policy.

ROS2-free so it can be unit-tested on the host. The ROS2 node (planner_node)
is a thin adapter over this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from gridworld.grid_coords import cell_to_world, world_to_cell
from gridworld.mdp import ACTIONS, _DELTA
from gridworld.solvers import value_iteration, policy_actions


@dataclass
class FollowerParams:
    angle_tol: float = 0.06     # rad (~3.5°): align tightly before driving
    center_tol: float = 0.12    # m: "centered in the cell" threshold
    k_ang: float = 2.5
    k_lin: float = 0.6
    max_ang: float = 1.2
    max_lin: float = 0.22       # m/s: TurtleBot3 burger's nominal max; safe with 2 m lanes


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def next_cell(row, col, action_idx, n_rows, n_cols):
    dr, dc = _DELTA[ACTIONS[action_idx]]
    return (_clamp(row + dr, 0, n_rows - 1), _clamp(col + dc, 0, n_cols - 1))


def compute_command(pose, target_xy, params):
    x, y, yaw = pose
    tx, ty = target_xy
    heading = math.atan2(ty - y, tx - x)
    err = _wrap(heading - yaw)
    if abs(err) > params.angle_tol:
        return 0.0, _clamp(params.k_ang * err, -params.max_ang, params.max_ang)
    dist = math.hypot(tx - x, ty - y)
    lin = _clamp(params.k_lin * dist, 0.0, params.max_lin)
    ang = _clamp(params.k_ang * err, -params.max_ang, params.max_ang)
    return lin, ang


class Follower:
    """Drives a robot cell-to-cell along a DP policy.

    To stay clear of walls in a tight grid, the robot first centers itself in
    the current cell, then turns in place toward the policy's next cell and
    advances. Turns therefore happen at cell centers and motion is
    axis-aligned, so the robot never cuts a corner into a wall.
    """

    def __init__(self, scenario, params=None, world_offset=(0.0, 0.0)):
        self.scenario = scenario
        self.cfg = scenario.coords
        self.params = params or FollowerParams()
        self.world_offset = world_offset
        mdp = scenario.build_mdp()
        self.mdp = mdp
        _, policy, _ = value_iteration(mdp, theta=1e-9)
        self.actions = policy_actions(policy)
        self._last_cell = None     # cell we last entered
        self._centering = True     # busy centering in the current cell?

    def step(self, pose):
        x, y, yaw = pose
        # odom is spawn-relative; shift into world frame so cells/targets line up.
        x = x + self.world_offset[0]
        y = y + self.world_offset[1]
        rc = world_to_cell(x, y, self.cfg)
        if rc not in self.mdp.index:  # standing on a wall cell: stop safely
            return 0.0, 0.0, False
        s = self.mdp.index[rc]
        if self.mdp.terminal[s]:
            return 0.0, 0.0, True

        # Entering a new cell restarts the center-then-advance cycle.
        if rc != self._last_cell:
            self._last_cell = rc
            self._centering = True

        cx, cy = cell_to_world(rc[0], rc[1], self.cfg)
        if self._centering and math.hypot(cx - x, cy - y) <= self.params.center_tol:
            self._centering = False   # centered: commit to advancing this cell

        if self._centering:
            target = (cx, cy)         # drive to the current cell's center first
        else:
            nrc = next_cell(rc[0], rc[1], int(self.actions[s]),
                            self.cfg.n_rows, self.cfg.n_cols)
            target = cell_to_world(nrc[0], nrc[1], self.cfg)

        lin, ang = compute_command((x, y, yaw), target, self.params)
        return lin, ang, False
