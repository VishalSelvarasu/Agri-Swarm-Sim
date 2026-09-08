#!/usr/bin/env python3
import csv
import math
import random

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from agri_swarm_msgs.msg import WeedDetection


class DetectorNode(Node):
    """Noisy weed detector.

    Frame convention, shared with lane_follower_node, task_executor_node and
    allocator_node.cpp: gz DiffDrive odometry is spawn-relative, so the pose is
    lifted with world = origin + odom on arrival. Everything downstream --
    visibility tests, emitted positions -- is world frame, matching the oracle
    CSV and WeedDetection.position.
    """

    def __init__(self):
        super().__init__("detector")

        self.declare_parameter("robot_id", "robot_0")
        self.declare_parameter("ground_truth_csv", "")
        self.declare_parameter("seed", 0)
        # Spawn pose, supplied by the launch file.
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)

        # --- sensor geometry ---
        self.declare_parameter("sensor_range", 1.6)      # [m]
        self.declare_parameter("sensor_fov", 1.4)        # [rad], forward cone
        self.declare_parameter("rate_hz", 5.0)

        # --- noise model (the experiment axes) ---
        self.declare_parameter("recall_near", 0.95)      # P(detect) at range 0
        self.declare_parameter("recall_far", 0.55)       # P(detect) at max range
        # Spurious detections per second, per robot. At 0.25 a four-robot run
        # generates hundreds of phantom weeds against 79 real ones, so the
        # swarm spends its time treating noise and the experiment measures the
        # detector rather than the allocator. 0.02 is roughly one false
        # positive per robot per minute.
        self.declare_parameter("fp_per_sec", 0.02)       # spurious detections/s
        self.declare_parameter("position_sigma", 0.03)   # [m]

        # Beta shape params. TP mean = a/(a+b).
        self.declare_parameter("conf_tp_alpha", 6.0)     # TP mean ~0.75
        self.declare_parameter("conf_tp_beta", 2.0)
        self.declare_parameter("conf_fp_alpha", 2.0)     # FP mean ~0.40 -> OVERLAP
        self.declare_parameter("conf_fp_beta", 3.0)

        g = lambda n: self.get_parameter(n).value
        self.robot_id = g("robot_id")
        self.origin_x = float(g("origin_x"))
        self.origin_y = float(g("origin_y"))
        self.range = float(g("sensor_range"))
        self.fov = float(g("sensor_fov"))
        self.recall_near = float(g("recall_near"))
        self.recall_far = float(g("recall_far"))
        self.fp_per_sec = float(g("fp_per_sec"))
        self.pos_sigma = float(g("position_sigma"))
        self.tp_a, self.tp_b = float(g("conf_tp_alpha")), float(g("conf_tp_beta"))
        self.fp_a, self.fp_b = float(g("conf_fp_alpha")), float(g("conf_fp_beta"))

        if self.robot_id != "robot_0" and self.origin_x == 0.0 and self.origin_y == 0.0:
            self.get_logger().error(
                f"{self.robot_id}: origin_x and origin_y are both zero. The "
                "pose will stay in odom frame, so visibility is tested against "
                "the wrong position and false positives are emitted at odom "
                "coordinates. The launch file is not passing the spawn pose.")

        # Per-robot stream derived from the global seed. Two robots must not
        # share a stream, and a rerun of the same config must reproduce.
        self.rng = random.Random(f"{g('seed')}::{self.robot_id}")

        self.weeds = self._load_oracle(g("ground_truth_csv"))
        self.get_logger().info(
            f"{self.robot_id}: oracle has {len(self.weeds)} patches, "
            f"origin ({self.origin_x:+.2f}, {self.origin_y:+.2f})")

        self.pose = None          # (x, y, yaw) in WORLD frame
        self.logged_first_pose = False
        self.dt = 1.0 / float(g("rate_hz"))

        self.pub = self.create_publisher(WeedDetection, "/weed_detections", 20)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.create_timer(self.dt, self._tick)

    # ------------------------------------------------------------------

    def _load_oracle(self, path):
        if not path:
            raise RuntimeError("ground_truth_csv parameter is required")
        out = []
        with open(path) as f:
            for row in csv.DictReader(f):
                out.append({
                    "id": int(row["id"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "radius": float(row["radius"]),
                })
        return out

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # Odometry is spawn-relative; the oracle is world. Lift here, once, so
        # that no downstream computation has to know which frame it is in.
        self.pose = (self.origin_x + p.x, self.origin_y + p.y, yaw)

        if not self.logged_first_pose:
            self.logged_first_pose = True
            self.get_logger().info(
                f"first pose: world ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}) "
                f"= odom ({p.x:+.3f}, {p.y:+.3f}) "
                f"+ origin ({self.origin_x:+.3f}, {self.origin_y:+.3f})")

    # ------------------------------------------------------------------

    def _in_view(self, wx, wy):
        x, y, yaw = self.pose
        dx, dy = wx - x, wy - y
        r = math.hypot(dx, dy)
        if r > self.range:
            return None
        bearing = math.atan2(dy, dx) - yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        if abs(bearing) > self.fov / 2.0:
            return None
        return r

    def _recall_at(self, r):
        t = min(1.0, max(0.0, r / self.range))
        return self.recall_near + t * (self.recall_far - self.recall_near)

    def _emit(self, patch_id, x, y, radius, confidence):
        m = WeedDetection()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.patch_id = patch_id
        m.observer_id = self.robot_id
        m.position.x = x + self.rng.gauss(0.0, self.pos_sigma)
        m.position.y = y + self.rng.gauss(0.0, self.pos_sigma)
        m.position.z = 0.0
        m.radius = float(radius)
        m.confidence = float(min(1.0, max(0.0, confidence)))
        self.pub.publish(m)

    def _tick(self):
        if self.pose is None:
            return

        # True positives (and misses).
        for w in self.weeds:
            r = self._in_view(w["x"], w["y"])
            if r is None:
                continue
            # Per-tick trial. Longer dwell => more chances, which is the
            # intended coupling between speed and recall.
            if self.rng.random() < self._recall_at(r) * self.dt:
                conf = self.rng.betavariate(self.tp_a, self.tp_b)
                self._emit(w["id"], w["x"], w["y"], w["radius"], conf)

        # False positives: bare soil reported as weed. patch_id = 2**32-1
        # marks "no oracle correspondence" for the offline scorer. The pose is
        # already world frame, so the emitted position is too.
        if self.rng.random() < self.fp_per_sec * self.dt:
            x, y, yaw = self.pose
            b = yaw + self.rng.uniform(-self.fov / 2.0, self.fov / 2.0)
            d = self.rng.uniform(0.2, self.range)
            conf = self.rng.betavariate(self.fp_a, self.fp_b)
            self._emit(0xFFFFFFFF, x + d * math.cos(b), y + d * math.sin(b),
                       self.rng.uniform(0.05, 0.12), conf)


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()