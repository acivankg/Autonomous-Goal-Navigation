#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
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
# TEK YERDEN AYARLA (BUNLARA DOKUNMADIM)
# =========================
GOAL_X = 1.38
GOAL_Y = 1.16
GOAL_YAW_DEG = 90.0

MAX_REPLANS = 30  # bunu da değiştirmedim

# Ara hedef seçerken engelden uzak durma
SAFETY_CELLS = 4
SEARCH_MAX_RADIUS = 0.9
SEARCH_STEP = 0.05

# Skor ağırlıkları
W_GOAL = 1.0
W_CLEAR = 1.8
W_FRONTIER = 0.35
CLEARANCE_CAP_M = 2.0

# "Ben tamam sayacağım" eşiği (Nav2'nin kendi toleransı ayrı!)
FINAL_EPS_M = 0.03  # 3cm

# FINAL MODE'da (en az 1 ara hedef sonrası) Nav2 son santimi yapamıyorsa "close enough"
FINAL_SOFT_EPS_M = 0.05  # 5cm  (0.045m'de takıldığın için)

# Bu ayar ON olursa: robot goal civarında takılıp bitiremiyorsa,
# mesafe FINAL_EPS_M içindeyse "başardı" sayıp goal'ü cancel eder.
ALLOW_CANCEL_WHEN_CLOSE = True

# Nav2 action bekleme / stuck algılama
NAV2_GOAL_TIMEOUT_SEC = 45.0
STUCK_NO_IMPROV_SEC = 8.0
STUCK_IMPROV_EPS_M = 0.02

# Aynı ara hedefi seçip durmasın diye
MIN_SUBGOAL_DIST_FROM_ROBOT_M = 0.18

# "SUCCESS dedi ama gitmedi" filtresi
FAKE_SUCCESS_SUBGOAL_MAX_M = 0.15   # success sonrası subgoal'a hâlâ >15cm uzaksan şüpheli
FAKE_SUCCESS_MOVED_MAX_M = 0.05     # ve toplam hareket <5cm ise "fake success" say

# Recovery (basit)
BACKUP_SPEED = 0.12
BACKUP_DIST = 0.20
ROTATE_DEG = 25.0
ROTATE_SPEED = 0.8


@dataclass
class Candidate:
    x: float
    y: float
    dist_to_goal: float
    clearance_m: float
    is_frontier: bool
    cost: float


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class GoalPoseClient(Node):
    def __init__(self) -> None:
        super().__init__("goal_pose_client")

        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self._client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, map_qos)

        self.reached_intermediate: bool = False

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
                self.get_logger().info(
                    f"Harita geldi. size={m.info.width}x{m.info.height} res={m.info.resolution:.3f} "
                    f"origin=({m.info.origin.position.x:.2f},{m.info.origin.position.y:.2f})"
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
        return (min(max(x, minx), maxx), min(max(y, miny), maxy))

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
        m = self.map_msg
        if m is None:
            return False
        w = m.info.width
        h = m.info.height
        data = m.data
        for ix in range(max(0, mx - safety_cells), min(w, mx + safety_cells + 1)):
            for iy in range(max(0, my - safety_cells), min(h, my + safety_cells + 1)):
                if data[iy * w + ix] >= 50:
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

    def obstacle_clearance_m(self, mx: int, my: int, max_cells: int = 35) -> float:
        m = self.map_msg
        if m is None:
            return 0.0
        w = m.info.width
        h = m.info.height
        data = m.data
        res = m.info.resolution

        x0 = max(0, mx - max_cells)
        x1 = min(w - 1, mx + max_cells)
        y0 = max(0, my - max_cells)
        y1 = min(h - 1, my + max_cells)

        best_d2: Optional[int] = None
        for ix in range(x0, x1 + 1):
            dx = ix - mx
            for iy in range(y0, y1 + 1):
                if data[iy * w + ix] >= 50:
                    dy = iy - my
                    d2 = dx * dx + dy * dy
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2

        if best_d2 is None:
            return max_cells * res
        return math.sqrt(best_d2) * res

    # ---------------- TF / POSE ----------------
    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        timeout = Duration(seconds=0.3)
        for base in ("base_footprint", "base_link"):
            try:
                if not self.tf_buffer.can_transform("map", base, Time(), timeout):
                    continue
                tf = self.tf_buffer.lookup_transform("map", base, Time(), timeout)
                x = tf.transform.translation.x
                y = tf.transform.translation.y
                q = tf.transform.rotation
                yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
                return x, y, yaw
            except (LookupException, ConnectivityException, ExtrapolationException):
                continue
        return None

    # ---------------- RECOVERY ----------------
    def stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def backup_and_rotate(self) -> None:
        self.get_logger().warn("Recovery: azıcık geri + azıcık dönüyorum...")

        t = Twist()
        t.linear.x = -BACKUP_SPEED
        dur = BACKUP_DIST / BACKUP_SPEED
        t0 = time.time()
        while time.time() - t0 < dur:
            self.cmd_pub.publish(t)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.stop_robot()
        time.sleep(0.15)

        t = Twist()
        t.angular.z = ROTATE_SPEED
        rot_dur = math.radians(ROTATE_DEG) / ROTATE_SPEED
        t0 = time.time()
        while time.time() - t0 < rot_dur:
            self.cmd_pub.publish(t)
            rclpy.spin_once(self, timeout_sec=0.05)
        self.stop_robot()
        time.sleep(0.15)

    # ---------------- ADAY SEÇİMİ ----------------
    def pick_goal_or_frontier_scored(
        self,
        gx: float,
        gy: float,
        safety_cells: int,
        max_radius: float,
        force_goal: bool,
        robot_xy: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if self.map_msg is None:
            return None

        cgx, cgy = self.clamp_to_map(gx, gy)
        idx = self.world_to_map(cgx, cgy)
        if idx is None:
            return None
        mx, my = idx
        gv = self.cell_value(mx, my)

        if self.cell_known_free(mx, my) and self.clear_of_obstacles(mx, my, safety_cells):
            return (cgx, cgy)

        if force_goal and (gv is not None) and (0 <= gv < 50):
            self.get_logger().warn(f"FINAL MODE: goal bilinen boş (cell={gv}) -> safety bypass ile goal'u direkt deniyorum.")
            return (cgx, cgy)

        step = SEARCH_STEP
        max_steps = int(max_radius / step)
        best: Optional[Candidate] = None

        rx, ry = robot_xy if robot_xy is not None else (None, None)

        for r_step in range(1, max_steps + 1):
            r = r_step * step
            samples = max(24, int(2 * math.pi * r / step))
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

                if rx is not None and math.hypot(sx - rx, sy - ry) < MIN_SUBGOAL_DIST_FROM_ROBOT_M:
                    continue

                dist_goal = math.hypot(gx - sx, gy - sy)
                clearance = min(self.obstacle_clearance_m(mx2, my2, max_cells=35), CLEARANCE_CAP_M)
                frontier = self.has_unknown_neighbor(mx2, my2)

                cost = (W_GOAL * dist_goal) - (W_CLEAR * clearance) - (W_FRONTIER * (1.0 if frontier else 0.0))

                cand = Candidate(x=sx, y=sy, dist_to_goal=dist_goal, clearance_m=clearance, is_frontier=frontier, cost=cost)

                if best is None or cand.cost < best.cost:
                    best = cand

        if best is None:
            return None

        self.get_logger().info(
            f"Seçim: ({best.x:.2f},{best.y:.2f}) dist_to_goal={best.dist_to_goal:.2f}m "
            f"clearance={best.clearance_m:.2f}m frontier={best.is_frontier} cost={best.cost:.3f}"
        )
        return (best.x, best.y)

    # ---------------- NAV2 (TIMEOUT + STUCK) ----------------
    def navigate_to_monitored(self, x: float, y: float, yaw_deg: float) -> int:
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

        t_deadline = time.time() + NAV2_GOAL_TIMEOUT_SEC
        best_dist = float("inf")
        last_improv_t = time.time()

        while time.time() < t_deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            if res_fut.done():
                res = res_fut.result()
                return res.status if res is not None else GoalStatus.STATUS_ABORTED

            pose = self.get_robot_pose()
            if pose is None:
                continue
            rx, ry, _ = pose
            d = math.hypot(x - rx, y - ry)

            if d + STUCK_IMPROV_EPS_M < best_dist:
                best_dist = d
                last_improv_t = time.time()

            if ALLOW_CANCEL_WHEN_CLOSE and d <= FINAL_EPS_M:
                self.get_logger().warn(f"Subgoal'e çok yakın (d={d:.3f}m). Goal cancel + success sayıyorum.")
                cancel_fut = gh.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_fut, timeout_sec=1.0)
                return GoalStatus.STATUS_SUCCEEDED

            if time.time() - last_improv_t > STUCK_NO_IMPROV_SEC:
                self.get_logger().warn(f"Stuck: {STUCK_NO_IMPROV_SEC:.0f}s iyileşme yok (best_dist={best_dist:.2f}m). Goal cancel.")
                cancel_fut = gh.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_fut, timeout_sec=1.0)
                return GoalStatus.STATUS_ABORTED

        self.get_logger().warn("Nav2 goal timeout. Cancel ediyorum.")
        cancel_fut = gh.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_fut, timeout_sec=1.0)
        return GoalStatus.STATUS_ABORTED

    # ---------------- RUN ----------------
    def run(self) -> None:
        if not self.wait_for_map(timeout_sec=20.0):
            return

        for _ in range(40):
            if self.get_robot_pose() is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        for attempt in range(1, MAX_REPLANS + 1):
            pose = self.get_robot_pose()
            if pose is None:
                self.get_logger().warn("Robot pozu yok (map->base_* TF). SLAM/Nav2 hazır mı?")
                time.sleep(0.2)
                continue

            rx, ry, _ = pose
            d_goal_now = math.hypot(GOAL_X - rx, GOAL_Y - ry)
            if d_goal_now <= FINAL_EPS_M:
                self.get_logger().info("Asıl hedefe ulaşıldı ✅")
                return

            max_r = SEARCH_MAX_RADIUS + 0.2 * (attempt - 1)

            tgt = self.pick_goal_or_frontier_scored(
                GOAL_X, GOAL_Y,
                safety_cells=SAFETY_CELLS,
                max_radius=max_r,
                force_goal=self.reached_intermediate,
                robot_xy=(rx, ry),
            )

            if tgt is None:
                # KRİTİK FIX: burada attempt zaten artıyor -> radius büyüyecek. Spam sonsuz döngü yok.
                self.get_logger().warn(f"[{attempt}/{MAX_REPLANS}] Aday hedef yok. radius={max_r:.2f} -> büyütüp deneyeceğim.")
                time.sleep(0.15)
                continue

            tx, ty = tgt
            if math.hypot(tx - GOAL_X, ty - GOAL_Y) <= 1e-6:
                yaw_cmd = GOAL_YAW_DEG
            else:
                yaw_cmd = math.degrees(math.atan2(GOAL_Y - ty, GOAL_X - tx))

            self.get_logger().info(
                f"[{attempt}/{MAX_REPLANS}] Seçilen hedef: ({tx:.2f},{ty:.2f})  asıl: ({GOAL_X:.2f},{GOAL_Y:.2f}) "
                f"final_mode={self.reached_intermediate}"
            )

            pose_before = self.get_robot_pose()
            status = self.navigate_to_monitored(tx, ty, yaw_cmd)
            pose_after = self.get_robot_pose()

            moved = 0.0
            if pose_before is not None and pose_after is not None:
                moved = math.hypot(pose_after[0] - pose_before[0], pose_after[1] - pose_before[1])

            # Son duruma göre mesafeleri ölç
            pose2 = self.get_robot_pose()
            if pose2 is not None:
                rx2, ry2, _ = pose2
                d_goal = math.hypot(GOAL_X - rx2, GOAL_Y - ry2)
                d_sub = math.hypot(tx - rx2, ty - ry2)
                self.get_logger().info(f"Sonuç sonrası: d_goal={d_goal:.3f}m d_sub={d_sub:.3f}m moved={moved:.3f}m")

                if d_goal <= FINAL_EPS_M:
                    self.get_logger().info("Asıl hedefe ulaşıldı ✅")
                    return

            # Fake success filtresi
            if status == GoalStatus.STATUS_SUCCEEDED and pose2 is not None:
                rx2, ry2, _ = pose2
                d_sub = math.hypot(tx - rx2, ty - ry2)
                if d_sub > FAKE_SUCCESS_SUBGOAL_MAX_M and moved < FAKE_SUCCESS_MOVED_MAX_M:
                    self.get_logger().warn(
                        f"Nav2 SUCCESS dedi ama subgoal hâlâ uzak (d_sub={d_sub:.2f}m) ve hareket yok (moved={moved:.2f}). "
                        f"Fake success -> ABORT gibi davranıyorum."
                    )
                    status = GoalStatus.STATUS_ABORTED

            if status == GoalStatus.STATUS_SUCCEEDED:
                # Ara hedefe ulaştık -> final mode aç
                self.reached_intermediate = True
                self.get_logger().info("Ara hedefe ulaşıldı. Asıl hedefi tekrar deniyorum (FINAL MODE açık).")
                continue

            # FINAL MODE'da ve goal dibindeysek: inflation yüzünden son 1-2cm imkansız olabilir -> close enough say
            if pose2 is not None and self.reached_intermediate:
                rx2, ry2, _ = pose2
                d_goal = math.hypot(GOAL_X - rx2, GOAL_Y - ry2)
                if d_goal <= FINAL_SOFT_EPS_M:
                    self.get_logger().warn(
                        f"FINAL MODE: goal dibindeyim (d_goal={d_goal:.3f}m) ama Nav2 bitiremiyor. Close-enough sayıyorum ✅"
                    )
                    return

            self.get_logger().warn(f"Nav2 status={status}, moved={moved:.3f}m")
            if status == GoalStatus.STATUS_ABORTED and moved < 0.03:
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
