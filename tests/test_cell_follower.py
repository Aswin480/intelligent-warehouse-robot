import math
from gridworld.scenario import load_scenario
import sys, os
sys.path.insert(0, os.path.join("ros2_ws", "src", "dp_nav"))
from dp_nav.cell_follower import compute_command, FollowerParams, Follower, next_cell


def test_rotates_when_facing_wrong_way():
    p = FollowerParams()
    # Target is straight ahead in +x, but robot faces +y (yaw=pi/2): must rotate.
    lin, ang = compute_command((0.0, 0.0, math.pi / 2), (1.0, 0.0), p)
    assert lin == 0.0
    assert ang < 0.0  # turn clockwise toward +x


def test_drives_forward_when_aligned():
    p = FollowerParams()
    lin, ang = compute_command((0.0, 0.0, 0.0), (1.0, 0.0), p)
    assert lin > 0.0
    assert abs(ang) < 1e-6


def test_next_cell_clamps():
    assert next_cell(0, 0, 0, 6, 7) == (0, 0)  # N from top row stays
    assert next_cell(5, 0, 2, 6, 7) == (5, 1)  # E moves right


def test_world_offset_maps_spawn_relative_odom_to_start_cell():
    sc = load_scenario("scenarios/maze.yaml")
    from gridworld.grid_coords import cell_to_world, world_to_cell
    offset = cell_to_world(sc.start[0], sc.start[1], sc.coords)
    # arithmetic: a zero odom reading + offset must land in the start cell
    assert world_to_cell(0.0 + offset[0], 0.0 + offset[1], sc.coords) == sc.start
    # Follower wired with the offset must not treat the zero pose as a terminal/crash
    f = Follower(sc, world_offset=offset)
    lin, ang, reached = f.step((0.0, 0.0, 0.0))
    assert reached is False  # the start cell is not the goal
    # Without the offset, the same zero pose resolves to cell (0,0) instead:
    assert world_to_cell(0.0, 0.0, sc.coords) != sc.start


def test_follower_reaches_goal_flag_at_terminal():
    sc = load_scenario("scenarios/maze.yaml")
    f = Follower(sc)
    from gridworld.grid_coords import cell_to_world
    # Place the robot at the goal cell center.
    goal_rc = next(rc for rc, ch in
                   ((r_c, sc.grid[r_c[0]][r_c[1]])
                    for r_c in [(r, c) for r in range(sc.coords.n_rows)
                                for c in range(sc.coords.n_cols)])
                   if ch == "G")
    gx, gy = cell_to_world(goal_rc[0], goal_rc[1], sc.coords)
    lin, ang, reached = f.step((gx, gy, 0.0))
    assert reached is True
    assert lin == 0.0 and ang == 0.0


def test_follower_advances_from_cell_center_toward_policy():
    # Centered in the start cell and already facing the policy direction
    # (East, yaw=0), the follower commits to advancing (drives forward) rather
    # than continuing to center.
    sc = load_scenario("scenarios/maze.yaml")
    from gridworld.grid_coords import cell_to_world
    f = Follower(sc)  # world_offset defaults to (0,0): pose treated as absolute
    cx, cy = cell_to_world(sc.start[0], sc.start[1], sc.coords)
    lin, ang, reached = f.step((cx, cy, 0.0))
    assert reached is False
    assert lin > 0.0  # advancing toward the next cell, not stalled
