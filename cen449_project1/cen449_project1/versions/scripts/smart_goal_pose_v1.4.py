#!/usr/bin/env python3
import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time
from rclpy.duration import Duration


# =========================
# TEK YERDEN AYARLA
# =========================
GOAL_X = 0.0148
GOAL_Y = 2.31
GOAL_YAW_DEG = 90.0

SAFETY_CELLS = 4          # 3 -> 4 (engel dibi olmasın)
SEARCH_MAX_RADIUS = 0.9   # 0.7 -> 0.9 (goal unknown/occupied ise daha geniş ara)
SEARCH_STEP = 0.05

MAX_REPLANS = 20

# Recovery (basit)
BACKUP_SPEED = 0.12
BACKUP_DIST = 0.20
ROTATE_DEG = 25.0
ROTATE_SPEED = 0.8


class GoalPoseClient(Node):
    def __init__(self) -> None:
        super().__init__("goal_pose_client")

        # sim time
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        # Nav2 action
        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # TF (thread yok -> shutdown exception gürültüsü de yok)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        # cmd_vel recovery
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # /map QoS: transient_local
        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, map_qos)

    # ---------------- MAP ----------------
    def map_callback(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg

    def wait_for_map(self, timeout_sec: float = 20.0) -> bool:
        self.get_logger().info("Harita (/map) bekleniyor...")
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.map_msg is not None and self.map_msg.info.width > 0:
                m = self.map_msg
                w_m = m.info.width * m.info.resolution
                h_m = m.info.height * m.info.resolution
                ox = m.info.origin.position.x
                oy = m.info.origin.position.y
                self.get_logger().info(
                    f"Harita geldi. size={m.info.width}x{m.info.height} res={m.info.resolution:.3f} "
                    f"origin=({ox:.2f},{oy:.2f}) span=({w_m:.2f}m,{h_m:.2f}m)"
                )
                return True
        self.get_logger().error("Harita gelmedi (timeout). /map yayınını kontrol et.")
        return False

    def map_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        m = self.map_msg
        if m is None:
            return None
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        maxx = ox + (m.info.width - 1) * m.info.resolution
        maxy = oy + (m.info.height - 1) * m.info.resolution
        return ox, oy, maxx, maxy

    def clamp_to_map(self, x: float, y: float) -> Tuple[float, float]:
        b = self.map_bounds()
        if b is None:
            return x, y
        minx, miny, maxx, maxy = b
        cx = min(max(x, minx), maxx)
        cy = min(max(y, miny), maxy)
        return cx, cy

    def world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        m = self.map_msg
        if m is None:
            return None
        res = m.info.resolution
        ox = m.info.origin.position.x
        oy = m.info.origin.position.y
        mx = int((x - ox) / res)
        my = int((y - oy) / res)
        if 0 <= mx < m.info.width and 0 <= my < m.info.height:
            return mx, my
        return None

    def cell_value(self, mx: int, my: int) -> Optional[int]:
        m = self.map_msg
        if m is None:
            return None
        w = m.info.width
        h = m.info.height
        if not (0 <= mx < w and 0 <= my < h):
            return None
        return m.data[my * w + mx]

    def cell_known_free(self, mx: int, my: int) -> bool:
        v = self.cell_value(mx, my)
        return (v is not None) and (0 <= v < 50)

    def clear_of_obstacles(self, mx: int, my: int, safety_cells: int) -> bool:
        """
        SADECE engel (>=50) genişlet. Unknown (-1) burada engel sayılmıyor.
        """
        m = self.map_msg
        if m is None:
            return False
        w = m.info.width
        h = m.info.height
        data = m.data

        for ix in range(max(0, mx - safety_cells), min(w, mx + safety_cells + 1)):
            for iy in range(max(0, my - safety_cells), min(h, my + safety_cells + 1)):
                v = data[iy * w + ix]
                if v >= 50:
                    return False
        return True

    def has_unknown_neighbor(self, mx: int, my: int) -> bool:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                v = self.cell_value(mx + dx, my + dy)
                if v == -1:
                    return True
        return False

    def pick_goal_or_frontier(self, gx: float, gy: float, safety_cells: int, max_radius: float) -> Optional[Tuple[float, float]]:
        """
        - Eğer (gx,gy) known free + obstacle clearance => direkt git
        - Değilse en yakındaki:
            1) frontier (known free, obstacle cleared, neighbor unknown)
            2) yoksa normal known free + obstacle cleared
        """
        if self.map_msg is None:
            return None

        # goal map dışında olabilir -> map içine clamp edip aramayı oradan başlat
        cgx, cgy = self.clamp_to_map(gx, gy)

        idx = self.world_to_map(cgx, cgy)
        if idx is None:
            return None
        mx, my = idx

        if self.cell_known_free(mx, my) and self.clear_of_obstacles(mx, my, safety_cells):
            return (cgx, cgy)

        best_frontier = None
        best_free = None

        step = SEARCH_STEP
        max_steps = int(max_radius / step)

        for r_step in range(1, max_steps + 1):
            r = r_step * step
            samples = max(12, int(2 * math.pi * r / step))
            for k in range(samples):
                a = 2 * math.pi * k / samples
                sx = cgx + r * math.cos(a)
                sy = cgy + r * math.sin(a)
                sx, sy = self.clamp_to_map(sx, sy)

                idx2 = self.world_to_map(sx, sy)
                if idx2 is None:
                    continue
                mx2, my2 = idx2

                if not self.cell_known_free(mx2, my2):
                    continue
                if not self.clear_of_obstacles(mx2, my2, safety_cells):
                    continue

                if best_free is None:
                    best_free = (sx, sy)

                if self.has_unknown_neighbor(mx2, my2):
                    best_frontier = (sx, sy)
                    break

            if best_frontier is not None:
                break

        return best_frontier if best_frontier is not None else best_free

    # ---------------- TF / POSE ----------------
    def get_robot_pose(self) -> Optional[Tuple[float, float]]:
        timeout = Duration(seconds=0.3)
        for base in ("base_footprint", "base_link"):
            try:
                if not self.tf_buffer.can_transform("map", base, Time(), timeout):
                    continue
                tf = self.tf_buffer.lookup_transform("map", base, Time(), timeout)
                return tf.transform.translation.x, tf.transform.translation.y
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue
        return None

    # ---------------- RECOVERY ----------------
    def stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def backup_and_rotate(self) -> None:
        self.get_logger().warn("Recovery: azıcık geri + azıcık dönüyorum...")

        # backup
        t = Twist()
        t.linear.x = -BACKUP_SPEED
        dur = BACKUP_DIST / BACKUP_SPEED
        t0 = time.time()
        while time.time() - t0 < dur:
            self.cmd_pub.publish(t)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.stop_robot()
        time.sleep(0.2)

        # rotate
        t = Twist()
        t.angular.z = ROTATE_SPEED
        rot_dur = math.radians(ROTATE_DEG) / ROTATE_SPEED
        t0 = time.time()
        while time.time() - t0 < rot_dur:
            self.cmd_pub.publish(t)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.stop_robot()
        time.sleep(0.2)

    # ---------------- NAV2 ----------------
    def navigate_to(self, x: float, y: float, yaw_deg: float) -> int:
        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Nav2 action server yok.")
            return GoalStatus.STATUS_ABORTED

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        yaw = math.radians(yaw_deg)
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Goal gönderiliyor: x={x:.2f}, y={y:.2f}, yaw={yaw_deg:.1f} deg")
        send_fut = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)

        gh = send_fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn("Goal reddedildi.")
            return GoalStatus.STATUS_ABORTED

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)

        res = res_fut.result()
        return res.status if res is not None else GoalStatus.STATUS_ABORTED

    # ---------------- RUN ----------------
    def run(self) -> None:
        if not self.wait_for_map(timeout_sec=20.0):
            return

        # TF gelmesi için azıcık bekleme (Nav2 yeni açıldıysa)
        for _ in range(30):
            if self.get_robot_pose() is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        for attempt in range(1, MAX_REPLANS + 1):
            pose = self.get_robot_pose()
            if pose is None:
                self.get_logger().warn("Robot pozu alınamadı (map->base_* TF yok). SLAM/Nav2 hazır mı?")
                time.sleep(0.3)
                continue

            max_r = SEARCH_MAX_RADIUS + 0.2 * (attempt - 1)

            tgt = self.pick_goal_or_frontier(GOAL_X, GOAL_Y, safety_cells=SAFETY_CELLS, max_radius=max_r)
            if tgt is None:
                self.get_logger().warn(f"[{attempt}/{MAX_REPLANS}] Aday hedef bulunamadı. radius={max_r:.2f} -> büyütüp deneyeceğim.")
                continue

            tx, ty = tgt
            yaw_to = math.degrees(math.atan2(GOAL_Y - ty, GOAL_X - tx))

            self.get_logger().info(f"[{attempt}/{MAX_REPLANS}] Seçilen hedef: ({tx:.2f},{ty:.2f})  asıl: ({GOAL_X:.2f},{GOAL_Y:.2f})")
            status = self.navigate_to(tx, ty, yaw_to)

            if status == GoalStatus.STATUS_SUCCEEDED:
                if math.hypot(tx - GOAL_X, ty - GOAL_Y) < 0.05:
                    self.get_logger().info("Asıl hedefe ulaşıldı ✅")
                    return
                self.get_logger().info("Ara hedefe ulaşıldı. Asıl hedefi tekrar deniyorum...")
                continue

            self.get_logger().warn(f"Nav2 status={status}. Recovery + tekrar...")
            self.backup_and_rotate()

        self.get_logger().error("MAX_REPLANS bitti. Hedefe güvenli şekilde gidemedim.")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalPoseClient()
    try:
        node.run()
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
