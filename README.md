# Autonomous Goal Navigation (ROS 2 / TurtleBot3)

This repository contains the course project developed for **CEN449: Introduction to Autonomous Robots**.  
The project demonstrates an autonomous navigation behavior in ROS 2 where a mobile robot attempts to reach a specified goal position using the Nav2 stack, while applying a lightweight goal-replanning strategy based on occupancy grid information.

## Summary

- **Platform:** ROS 2 (Humble), TurtleBot3-compatible setup
- **Core capability:** autonomous navigation to a target pose via Nav2 (`navigate_to_pose`)
- **Enhancement:** when the direct goal is unsafe/unreachable, the node selects an intermediate sub-goal using a scored search around the target based on:
  - distance-to-goal,
  - obstacle clearance,
  - frontier preference (unknown-map boundary),
  - safety checks in the occupancy grid.

## Presentation & Demo

A recorded presentation explaining the project and demonstrating a sample operational scenario is available here:  
- https://youtu.be/TPuX7nkHmnk

## Repository Layout

Key items in the package:
- `launch/full_project_launch.py` — main launch file used to run the project
- `cen449_project/smart_goal_pose_v1.6.py` — main navigation node (used by the launch file)
- `cen449_project/versions/scripts/` — archived development versions (`smart_goal_pose_v1.0` … `v1.6`)
- `test/` — style and lint tests (`flake8`, `pep257`, etc.)
- Standard ROS 2 Python package files: `package.xml`, `setup.py`, `setup.cfg`, `resource/`, etc.

> **Note on naming:**  
> The ROS 2 package name used by `colcon`/`ros2 launch` is **`cen449_project1`**, while the Python module directory is **`cen449_project/`**.

## Prerequisites

### Software
- Ubuntu + ROS 2 Humble environment properly installed and sourced
- A working TurtleBot3 + Nav2 setup (simulation or real robot), including:
  - `nav2_msgs` (NavigateToPose action)
  - TF frames available between `map` and the robot base (`base_link` or `base_footprint`)
  - `/map` published as `nav_msgs/OccupancyGrid`

### ROS 2 Workspace
This package is assumed to be placed inside a ROS 2 workspace (example: `~/turtlebot3_ws/src/`).

## Build & Run

From your workspace directory:

1) Navigate to your workspace:
```bash
cd ~/turtlebot3_ws
```
2) Source ROS 2:
```bash
source /opt/ros/humble/setup.bash
```
3) Build only this package:
```bash
colcon build --symlink-install --packages-select cen449_project1
```
4) Source the workspace overlay:
```bash
source install/setup.bash
```
5) Launch:
```bash
ros2 launch cen449_project1 full_project_launch.py
```
Note: Replace `~/turtlebot3_ws` with your workspace path if different.

## Configuration (Target Goal)

The goal pose is defined at the top of the navigation script:

- `GOAL_X`, `GOAL_Y` (meters, in the `map` frame)
- `GOAL_YAW_DEG` (degrees)

Example (as currently set in the code):
- `GOAL_X = 0.0148`
- `GOAL_Y = 2.31`
- `GOAL_YAW_DEG = 90.0`

To change the target, edit those constants and rebuild (or re-run if using `--symlink-install` and your environment reloads correctly).

## How the Navigation Node Works (High-Level)

The node sends a `NavigateToPose` goal to Nav2. If the direct target cell is not suitable, it performs a local search around the goal and selects an intermediate waypoint.

Core steps:
1. Wait for `/map` and a valid TF transform (`map -> base_link` or `map -> base_footprint`).
2. Evaluate whether the goal cell is **known free** and sufficiently clear of obstacles.
3. If not, sample candidate positions in expanding rings around the goal and score each candidate using:
   - distance to the final goal (prefer smaller),
   - obstacle clearance (prefer larger),
   - frontier presence (optionally prefer candidates adjacent to unknown cells).
4. Send the chosen target (goal or sub-goal) to Nav2 and wait for completion (with a timeout).
5. If a sub-goal is reached, enable a “final mode” that more aggressively retries the main goal.
6. If Nav2 aborts with minimal movement, optionally apply a simple recovery (brief backup + rotate) and retry.

## Runtime Notes / Assumptions

- **Frame conventions:** the goal is expressed in the `map` frame.
- **Topics/Interfaces used:**
  - `/map` (`nav_msgs/OccupancyGrid`)
  - `/cmd_vel` (`geometry_msgs/Twist`) for simple recovery motion
  - Nav2 action server: `navigate_to_pose` (`nav2_msgs/action/NavigateToPose`)
  - TF lookup: `map -> base_link` or `map -> base_footprint`

## Troubleshooting

Common console messages produced by `cen449_project/smart_goal_pose_v1.6.py` include:

- **`Map not received (timeout). Please verify that /map is being published.`**  
  `/map` is not available. Ensure SLAM/localization + map publishing are running (or that your launch file starts them).

- **`Nav2 action server is not available.`**  
  Nav2 is not running (or the action name differs). Confirm Nav2 is launched and provides `navigate_to_pose`.

- **`Robot pose unavailable (map -> base_* TF). Is SLAM/Nav2 running?`**  
  TF from `map` to `base_link`/`base_footprint` is not ready. Verify localization/SLAM and robot state publishers are active.

- **`Nav2 goal timed out. Canceling.`**  
  Navigation did not complete within the configured timeout. Check for blocked paths, planner issues, or increase the timeout if appropriate.

- **`MAX_REPLANS exhausted. Unable to reach the goal safely.`**  
  The node attempted multiple replans but could not find a safe/feasible route. Consider adjusting safety/search parameters or the goal location.

Additional messages you may see:
- **`Goal was rejected.`** (Nav2 rejected the goal request)
- **`Recovery: backing up briefly and rotating...`** (simple recovery motion before retry)
