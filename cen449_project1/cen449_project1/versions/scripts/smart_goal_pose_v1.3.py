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

        # ---------------- Nav2 action client ----------------
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ---------------- Parameters ----------------
        # Goal pose (in the "map" frame)
        self.declare_parameter('goal_x', 1.03)
        self.declare_parameter('goal_y', 0.6)
        self.declare_parameter('goal_yaw_deg', 90.0)

        # Max number of "explore + navigate" cycles
        self.declare_parameter('max_attempts', 3)

        # Base explore duration per attempt (each attempt uses attempt * explore_duration)
        self.declare_parameter('explore_duration', 120.0)  # seconds, fairly long

        # Map topic (from map_server or SLAM)
        self.declare_parameter('map_topic', '/map')

        # Escape / sweep behavior parameters
        self.declare_parameter('sweep_step_deg', 10.0)   # small angle step (≈ your "10%")
        self.declare_parameter('sweep_max_deg', 170.0)   # max small sweep angle
        self.declare_parameter('big_turn_deg', 90.0)     # big turn when super-stuck

        self.goal_x = float(self.get_parameter('goal_x').value)
        self.goal_y = float(self.get_parameter('goal_y').value)
        self.goal_yaw_deg = float(self.get_parameter('goal_yaw_deg').value)
        self.max_attempts = int(self.get_parameter('max_attempts').value)
        self.explore_duration = float(self.get_parameter('explore_duration').value)
        self.map_topic = str(self.get_parameter('map_topic').value)

        self.sweep_step_deg = float(self.get_parameter('sweep_step_deg').value)
        self.sweep_max_deg = float(self.get_parameter('sweep_max_deg').value)
        self.big_turn_deg = float(self.get_parameter('big_turn_deg').value)

        # ---------------- Map & perception data ----------------
        self._map = None

        # Laser data
        self._scan = None
        self._scan_min_range = float('inf')
        self._obstacle_threshold = 0.35  # m (front too close)

        # Odometry for stuck detection
        self._odom_pose = None  # (x, y)

        # Direction to rotate when sweeping (kept consistent, flipped when super-stuck)
        self._escape_dir = 1.0 if random.random() < 0.5 else -1.0

        # ---------------- Subscriptions & Publishers ----------------
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

        self._cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    # ---------------- Callbacks ----------------

    def _map_callback(self, msg: OccupancyGrid) -> None:
        self._map = msg

    def _scan_callback(self, msg: LaserScan) -> None:
        self._scan = msg
        valid = [r for r in msg.ranges if math.isfinite(r)]
        if valid:
            self._scan_min_range = min(valid)

    def _odom_callback(self, msg: Odometry) -> None:
        self._odom_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    # ---------------- Wait helpers ----------------

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

    def _wait_for_scan(self, timeout_sec: float = 10.0) -> bool:
        self.get_logger().info('Waiting for LaserScan on "scan"...')
        start = time.time()
        while self._scan is None and (time.time() - start) < timeout_sec and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._scan is None:
            self.get_logger().error('No LaserScan received.')
            return False

        self.get_logger().info('LaserScan received.')
        return True

    # ---------------- Map query: goal cell ----------------

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
        # yaw from quaternion (assuming planar map, z-w quaternion)
        yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))

        # Transform goal from world into map-origin frame
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

    # ---------------- Nav2 goal sending ----------------

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

    # ---------------- Laser helpers ----------------

    def _is_heading_clear(self, heading_rad: float, min_clearance: float) -> bool:
        """
        Check if the direction `heading_rad` (0 rad = straight ahead) is clear
        within a small angular window, using LaserScan.
        """
        if self._scan is None:
            return False

        scan = self._scan
        angle_min = scan.angle_min
        angle_max = scan.angle_max
        inc = scan.angle_increment

        # Heading must be within scan FOV
        if heading_rad < angle_min or heading_rad > angle_max:
            return False

        # Index of the central ray for this heading
        idx_center = int(round((heading_rad - angle_min) / inc))
        window = 5  # rays on each side

        min_range = float('inf')
        for i in range(idx_center - window, idx_center + window + 1):
            if 0 <= i < len(scan.ranges):
                r = scan.ranges[i]
                if math.isfinite(r):
                    min_range = min(min_range, r)

        if min_range == float('inf'):
            # No valid readings in this window -> treat as clear
            return True

        return min_range > min_clearance

    def _front_is_blocked(self, clearance: float = None) -> bool:
        if clearance is None:
            clearance = self._obstacle_threshold
        # Straight ahead is heading_rad = 0
        return not self._is_heading_clear(0.0, clearance) if self._scan is not None else False

    def _rotate_in_place(self, angle_rad: float, angular_speed: float = 0.7) -> None:
        """
        Rotate the robot in place by `angle_rad` (approximate, using time).
        Positive angle is CCW, negative is CW.
        """
        if angle_rad == 0.0:
            return

        direction = 1.0 if angle_rad > 0.0 else -1.0
        duration = abs(angle_rad) / angular_speed

        rate_hz = 20.0
        period = 1.0 / rate_hz
        start = time.time()

        self.get_logger().debug(
            f'Rotating in place for {duration:.2f} s to change heading by '
            f'{math.degrees(angle_rad):.1f} deg'
        )

        while rclpy.ok() and (time.time() - start) < duration:
            twist = Twist()
            twist.angular.z = direction * angular_speed
            self._cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        # Stop rotation
        self._cmd_pub.publish(Twist())

    # ---------------- Escape by sweeping angles ----------------

    def _escape_by_angle_sweep(self, super_stuck: bool) -> None:
        """
        Escape behavior when front is blocked or we are stuck:
        - Small backup.
        - Sweep headings in one rotational direction in small steps (e.g. 10°),
          checking with the laser if that heading is clear enough.
        - If no small heading works and we're super-stuck, do a bigger 90° turn.
        """
        self.get_logger().info(
            'Obstacle / stuck detected -> starting angle sweep escape...'
        )

        rate_hz = 10.0
        period = 1.0 / rate_hz

        # 1) Small backup to get away from the obstacle
        backup_duration = 0.8  # seconds
        start = time.time()
        while rclpy.ok() and (time.time() - start) < backup_duration:
            twist = Twist()
            twist.linear.x = -0.10
            self._cmd_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        self._cmd_pub.publish(Twist())

        # Maybe flip sweep direction if we've been super-stuck
        if super_stuck:
            self._escape_dir *= -1.0
            self.get_logger().info(
                'Super-stuck: flipping sweep direction for a bigger change.'
            )

        if self._scan is None:
            # No scan, just do a rough big turn
            big_angle = self._escape_dir * math.radians(self.big_turn_deg)
            self._rotate_in_place(big_angle)
            return

        step_rad = math.radians(self.sweep_step_deg)
        max_sweep_rad = math.radians(self.sweep_max_deg)
        max_k = max(1, int(max_sweep_rad / step_rad))

        min_clear_forward = self._obstacle_threshold * 1.2

        # 2) Small-step sweep: keep adjusting heading until we find a clear one
        found_angle = None
        for k in range(1, max_k + 1):
            ang = self._escape_dir * step_rad * k
            if self._is_heading_clear(ang, min_clear_forward):
                found_angle = ang
                self.get_logger().info(
                    f'Found clear heading at {math.degrees(ang):.1f} deg '
                    f'after {k} small steps.'
                )
                break

        if found_angle is not None:
            self._rotate_in_place(found_angle)
            return

        # 3) No small-angle success; if super-stuck, try a bigger 90° change
        if super_stuck:
            big_angle = self._escape_dir * math.radians(self.big_turn_deg)
            if self._is_heading_clear(big_angle, min_clear_forward):
                self.get_logger().info(
                    f'Super-stuck: using big turn of {self.big_turn_deg:.1f} deg.'
                )
                self._rotate_in_place(big_angle)
                return

            # Try opposite 90° as well
            alt_angle = -self._escape_dir * math.radians(self.big_turn_deg)
            if self._is_heading_clear(alt_angle, min_clear_forward):
                self.get_logger().info(
                    f'Super-stuck: using opposite big turn of {-self.big_turn_deg:.1f} deg.'
                )
                self._rotate_in_place(alt_angle)
                return

        # 4) Last resort: 180° turn
        self.get_logger().warn(
            'No clear direction found from laser; performing 180° turn as last resort.'
        )
        self._rotate_in_place(self._escape_dir * math.radians(180.0))

    # ---------------- Random wandering with stuck detection ----------------

    def _wander_randomly(self, duration_sec: float) -> None:
        """
        Wander for `duration_sec` seconds:

        - Start by choosing a direction and going straight while the path is clear.
        - If the robot sees an obstacle in front or is not making progress, it:
          * stops,
          * performs an angle-sweep escape (small 10° steps in one direction),
          * if super-stuck, allows a bigger 90° turn.
        - Periodically checks whether the goal cell has become known & free;
          if so, it stops exploring early.
        """
        self.get_logger().info(
            f'Starting exploration wander for {duration_sec:.1f} s...'
        )

        if not self._wait_for_scan(timeout_sec=5.0):
            self.get_logger().error('Cannot wander: no LaserScan.')
            return

        start = time.time()
        rate_hz = 10.0
        period = 1.0 / rate_hz

        forward_speed = 0.15

        # Stuck detection
        last_progress_time = start
        last_progress_pose = self._odom_pose
        stuck_counter = 0
        super_stuck_counter = 0
        progress_interval = 1.0   # seconds between checks
        min_progress = 0.03       # meters in interval to be considered "moving"

        # Goal reachability check
        last_goal_check = start
        goal_check_interval = 2.0  # seconds

        while rclpy.ok() and (time.time() - start) < duration_sec:
            now = time.time()

            # ---------- Check if goal became reachable during exploration ----------
            if (now - last_goal_check) > goal_check_interval:
                if self._goal_is_known_and_free():
                    self.get_logger().info(
                        'Goal became known & free during exploration; '
                        'stopping wander early.'
                    )
                    break
                last_goal_check = now

            # ---------- Stuck detection using odom ----------
            if (
                self._odom_pose is not None
                and last_progress_pose is not None
                and (now - last_progress_time) > progress_interval
            ):
                dx = self._odom_pose[0] - last_progress_pose[0]
                dy = self._odom_pose[1] - last_progress_pose[1]
                dist = math.hypot(dx, dy)

                if dist < min_progress and forward_speed > 0.05:
                    stuck_counter += 1
                    super_stuck_counter += 1
                    self.get_logger().debug(
                        f'Possible stuck (dist={dist:.3f}), count={stuck_counter}'
                    )
                else:
                    stuck_counter = 0

                last_progress_pose = self._odom_pose
                last_progress_time = now

            # ---------- Obstacle or stuck -> escape ----------
            front_blocked = self._front_is_blocked()
            if front_blocked or stuck_counter >= 2:
                self.get_logger().info(
                    f'Front_blocked={front_blocked}, stuck_counter={stuck_counter} '
                    '-> performing escape.'
                )
                # Stop before escaping
                self._cmd_pub.publish(Twist())

                super_stuck = super_stuck_counter >= 5
                self._escape_by_angle_sweep(super_stuck=super_stuck)

                # Reset local stuck counter after an escape
                stuck_counter = 0

                # Give some time to process new scan/odom data
                last_progress_time = time.time()
                last_progress_pose = self._odom_pose
                continue

            # ---------- Path is clear -> move straight ----------
            twist = Twist()
            twist.linear.x = forward_speed
            twist.angular.z = 0.0
            self._cmd_pub.publish(twist)

            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        # Stop at the end of wandering
        self._cmd_pub.publish(Twist())
        self.get_logger().info('Exploration wander finished.')

    # ---------------- Main logic: navigate with exploration ----------------

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
