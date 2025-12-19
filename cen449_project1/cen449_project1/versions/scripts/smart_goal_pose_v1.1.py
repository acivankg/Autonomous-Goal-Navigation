#!/usr/bin/env python3

import math
import random
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan


class SmartGoalPoseClient(Node):
    def __init__(self) -> None:
        super().__init__('smart_goal_pose_client')

        # Nav2 action client
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ------------- Parameters -------------
        # Use different goal_x/goal_y for static map vs SLAM.
        self.declare_parameter('goal_x', 1.03)
        self.declare_parameter('goal_y', 0.6)
        self.declare_parameter('goal_yaw_deg', 90.0)

        # Max number of explore + navigate cycles
        self.declare_parameter('max_attempts', 3)

        # Base explore duration per attempt (each attempt will be attempt * explore_duration)
        self.declare_parameter('explore_duration', 60.0)  # seconds

        # Map topic (from map_server or SLAM)
        self.declare_parameter('map_topic', '/map')

        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.goal_yaw_deg = float(self.get_parameter('goal_yaw_deg').value)
        self.max_attempts = int(self.get_parameter('max_attempts').value)
        self.explore_duration = float(self.get_parameter('explore_duration').value)
        self.map_topic = str(self.get_parameter('map_topic').value)

        # ------------- Map & perception data -------------
        self._map = None

        # Laser / obstacle info
        self._scan_min_range = float('inf')
        self._obstacle_threshold = 0.35  # m

        # Odometry for stuck detection
        self._odom_pose = None  # (x, y)

        # Subscriptions
        self._map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self._map_callback,
            10
        )

        self._scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self._scan_callback,
            10
        )

        self._odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self._odom_callback,
            10
        )

        # Publisher for random wandering
        self._cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    # ------------------- Callbacks -------------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _scan_callback(self, msg: LaserScan) -> None:
        valid = [r for r in msg.ranges if not math.isinf(r) and not math.isnan(r)]
        if valid:
            self._scan_min_range = min(valid)

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    # ------------------- Wait helpers -------------------

    def _wait_for_nav2(self, timeout_sec: float = 10.0) -> bool:
        self.get_logger().info('Waiting for Nav2 action server...')
        if not self._client.wait_for_server(timeout_sec=timeout_sec):
            self.get_logger().error('Nav2 action server not available.')
            return False
        return True

    def _wait_for_map(self, timeout_sec: float = 10.0) -> bool:
        self.get_logger().info(f'Waiting for map on "{self.map_topic}"...')
        start = time.time()
        while self._map is None and (time.time() - start) < timeout_sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._map is None:
            self.get_logger().error('No map received.')
            return False

        self.get_logger().info('Map received.')
        return True

    # ------------------- Map query: goal cell -------------------

    def _goal_cell_value(self):
        """Return occupancy value at the goal cell, or None if outside map."""
        if self._map is None:
            return None

        grid = self._map
        res = grid.info.resolution
        width = grid.info.width
        height = grid.info.height
        origin = grid.info.origin

        ox = origin.position.x
        oy = origin.position.y

        q = origin.orientation
        # yaw from quaternion (assuming planar map)
        yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))

        # world → map-origin frame
        dx = self.goal_x - ox
        dy = self.goal_y - oy

        mx = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        my = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        col = int(mx / res)
        row = int(my / res)

        if col < 0 or row < 0 or col >= width or row >= height:
            self.get_logger().warn('Goal is outside current map bounds.')
            return None

        idx = row * width + col
        if idx < 0 or idx >= len(grid.data):
            return None

        return grid.data[idx]

    def _goal_is_known_and_free(self) -> bool:
        val = self._goal_cell_value()
        if val is None:
            self.get_logger().warn('Cannot evaluate goal cell.')
            return False

        if val < 0:
            self.get_logger().info('Goal cell is UNKNOWN (-1) in map.')
            return False

        if val >= 50:
            self.get_logger().info(f'Goal cell appears OCCUPIED (value={val}).')
            return False

        self.get_logger().info('Goal cell is known and free.')
        return True

    # ------------------- Send Nav2 goal -------------------

    def _send_single_goal(self) -> bool:
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        gx = self.goal_x
        gy = self.goal_y
        yaw_deg = self.goal_yaw_deg
        yaw = math.radians(yaw_deg)

        goal.pose.pose.position.x = gx
        goal.pose.pose.position.y = gy
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f'Sending goal: x={gx:.2f}, y={gy:.2f}, yaw={yaw_deg:.1f} deg'
        )

        send_future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by Nav2.')
            return False

        self.get_logger().info('Goal accepted. Waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        if result is None:
            self.get_logger().error('No result returned from Nav2.')
            return False

        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal reached successfully!')
            return True

        self.get_logger().warn(f'Goal finished with status: {status}')
        return False

    # ------------------- Random wandering with stuck detection -------------------

    def _wander_randomly(self, duration_sec: float) -> None:
        """
        Wander for `duration_sec` seconds:
        - Use random forward / rotate modes.
        - Avoid obstacles using laser.
        - Detect if robot is stuck using odometry; if stuck, force rotation.
        """
        self.get_logger().info(
            f'Starting random wander for {duration_sec:.1f} s to explore...'
        )

        start = time.time()
        rate_hz = 10.0
        period = 1.0 / rate_hz

        # Simple "state machine": 'forward' or 'rotate'
        mode = 'forward'
        mode_end_time = start

        forward_speed = 0.15
        steer = 0.0
        rotate_speed = 0.8

        # For stuck detection
        last_progress_check = start
        last_progress_pose = self._odom_pose
        stuck_counter = 0
        progress_interval = 1.0  # seconds
        min_progress = 0.02      # meters in interval to consider as "moving"

        while rclpy.ok() and (time.time() - start) < duration_sec:
            now = time.time()

            # ---------- Stuck detection using odom ----------
            if (
                self._odom_pose is not None
                and (now - last_progress_check) > progress_interval
            ):
                if last_progress_pose is not None:
                    dx = self._odom_pose[0] - last_progress_pose[0]
                    dy = self._odom_pose[1] - last_progress_pose[1]
                    dist = math.hypot(dx, dy)

                    if dist < min_progress and mode == 'forward' and forward_speed > 0.05:
                        stuck_counter += 1
                        self.get_logger().debug(
                            f'Possible stuck detected (dist={dist:.3f}), count={stuck_counter}'
                        )
                    else:
                        stuck_counter = 0

                    last_progress_pose = self._odom_pose

                last_progress_check = now

            # ---------- Obstacle or stuck -> force rotate mode ----------
            if self._scan_min_range < self._obstacle_threshold or stuck_counter >= 3:
                mode = 'rotate'
                mode_end_time = now + random.uniform(1.0, 3.0)
                rotate_speed = random.uniform(0.8, 1.2) * (
                    1.0 if random.random() < 0.5 else -1.0
                )
                stuck_counter = 0
                self.get_logger().debug(
                    'Obstacle or stuck detected -> switching to rotate mode.'
                )

            # ---------- Randomly change mode when its time expires ----------
            if now > mode_end_time:
                if mode == 'forward':
                    # Switch to rotate
                    mode = 'rotate'
                    mode_end_time = now + random.uniform(1.0, 3.0)
                    rotate_speed = random.uniform(0.5, 1.0) * (
                        1.0 if random.random() < 0.5 else -1.0
                    )
                else:
                    # Switch to forward
                    mode = 'forward'
                    mode_end_time = now + random.uniform(2.0, 5.0)
                    forward_speed = random.uniform(0.10, 0.25)
                    steer = random.uniform(-0.5, 0.5)

            # ---------- Build and publish Twist ----------
            twist = Twist()
            if mode == 'forward':
                twist.linear.x = forward_speed
                twist.angular.z = steer
            else:
                twist.linear.x = 0.0
                twist.angular.z = rotate_speed

            self._cmd_pub.publish(twist)

            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        # Stop robot after wandering
        self._cmd_pub.publish(Twist())
        self.get_logger().info('Random wander finished.')

    # ------------------- Main logic -------------------

    def navigate_with_exploration(self) -> None:
        if not self._wait_for_nav2():
            return
        if not self._wait_for_map():
            return

        for attempt in range(1, self.max_attempts + 1):
            self.get_logger().info(
                f'===== Navigation attempt {attempt}/{self.max_attempts} ====='
            )

            # 1) Check if goal cell is known & free
            if not self._goal_is_known_and_free():
                if attempt == self.max_attempts:
                    self.get_logger().error(
                        'Target area is still unknown or occupied after '
                        f'{self.max_attempts} attempts. Aborting.'
                    )
                    break

                self.get_logger().warn(
                    'Target is in unknown/occupied space. Exploring before retrying...'
                )
                # Each attempt explores longer: attempt * base duration
                explore_time = self.explore_duration * attempt
                self._wander_randomly(explore_time)
                continue

            # 2) Goal seems reachable: send Nav2 goal
            succeeded = self._send_single_goal()
            if succeeded:
                break

            # 3) Nav failed (e.g. no global path) -> explore and retry
            if attempt < self.max_attempts:
                self.get_logger().warn(
                    'Navigation failed. Exploring and will retry...'
                )
                explore_time = self.explore_duration * attempt
                self._wander_randomly(explore_time)
            else:
                self.get_logger().error(
                    'Navigation failed after maximum number of attempts. Aborting.'
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SmartGoalPoseClient()
    try:
        node.navigate_with_exploration()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()