#!/usr/bin/env python3

from __future__ import annotations

import math
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from agri_swarm_msgs.msg import RobotState, Treatment


class EnergyMonitor(Node):

    def __init__(self):
        super().__init__("energy_monitor")

        self.declare_parameter("robot_id", "robot_0")
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("energy_capacity_j", 40000.0)
        self.declare_parameter("energy_per_m_j", 12.0)
        self.declare_parameter("treat_cost_j", 30.0)
        self.declare_parameter("reserve_fraction", 0.15)
        self.declare_parameter("idle_w", 0.2)
        self.declare_parameter("fail_at_s", -1.0)

        g = lambda n: self.get_parameter(n).value
        self.robot_id = g("robot_id")
        self.origin_x = float(g("origin_x"))
        self.origin_y = float(g("origin_y"))
        self.capacity = float(g("energy_capacity_j"))
        self.energy_per_m = float(g("energy_per_m_j"))
        self.treat_cost = float(g("treat_cost_j"))
        self.reserve_frac = float(g("reserve_fraction"))
        self.idle_w = float(g("idle_w"))
        self.fail_at_s = float(g("fail_at_s"))

        self.energy_j = self.capacity
        self.spent_travel_j = 0.0
        self.spent_treat_j = 0.0
        self.spent_idle_j = 0.0
        self.distance_m = 0.0
        self.treatments = 0

        self.pose = None          
        self.orientation = None
        self.last_xy = None
        self.speed = 0.0
        self.failed = False
        self.warned_empty = False

        self.dt = 1.0 / float(g("rate_hz"))
        self.start_s = self._now_s()
        self.last_tick_s = self.start_s
        self.last_report_s = self.start_s

        self.get_logger().info(
            f"{self.robot_id}: capacity {self.capacity:.0f} J, "
            f"{self.energy_per_m:.1f} J/m, {self.treat_cost:.1f} J/treatment, "
            f"reserve {self.reserve_frac * 100:.0f}%")

        self.pub = self.create_publisher(RobotState, "/robot_states", 50)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.create_subscription(Treatment, "/treatments", self._on_treatment, 50)
        self.create_timer(self.dt, self._tick)

    # ------------------------------------------------------------------ clock

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @property
    def total_spent_j(self) -> float:
        """Energy actually consumed. Unbounded: keeps counting past capacity."""
        return self.spent_travel_j + self.spent_treat_j + self.spent_idle_j

    # ------------------------------------------------------------------ input

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        xy = (self.origin_x + p.x, self.origin_y + p.y)

        if self.last_xy is not None:
            step = math.dist(xy, self.last_xy)
            self.distance_m += step
            self.spent_travel_j += step * self.energy_per_m
        self.last_xy = xy

        self.pose = xy
        self.orientation = msg.pose.pose.orientation
        self.speed = abs(msg.twist.twist.linear.x)

    def _on_treatment(self, m: Treatment):
        if m.robot_id != self.robot_id:
            return
        self.treatments += 1
        self.spent_treat_j += self.treat_cost

    # ------------------------------------------------------------------- tick

    def _tick(self):
        now = self._now_s()
        elapsed = now - self.last_tick_s
        self.last_tick_s = now

        if self.fail_at_s >= 0.0 and not self.failed:
            if now - self.start_s >= self.fail_at_s:
                self.failed = True
                self.get_logger().warn(
                    f"{self.robot_id} failed at t+{self.fail_at_s:.1f}s "
                    "(injected); heartbeat stops here")

        
        if self.failed:
            return

        self.spent_idle_j += self.idle_w * max(0.0, elapsed)
        self.energy_j = max(0.0, self.capacity - self.total_spent_j)

        if self.energy_j <= 0.0 and not self.warned_empty:
            self.warned_empty = True
            self.get_logger().warn(
                f"{self.robot_id} is flat after {self.distance_m:.1f} m and "
                f"{self.treatments} treatments. Capacity is likely undersized "
                "for the mission.")

        if self.pose is None:
            return

        
        if now - self.last_report_s >= 30.0:
            self.last_report_s = now
            self.get_logger().info(
                f"travelled {self.distance_m:.1f} m, {self.treatments} treatments, "
                f"{self.total_spent_j:.0f} J spent of {self.capacity:.0f} J "
                f"(travel {self.spent_travel_j:.0f}, treat "
                f"{self.spent_treat_j:.0f}, idle {self.spent_idle_j:.0f})")

        m = RobotState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.robot_id = self.robot_id
        m.pose.position.x = self.pose[0]
        m.pose.position.y = self.pose[1]
        m.pose.position.z = 0.0
        if self.orientation is not None:
            m.pose.orientation = self.orientation
        m.remaining_energy_j = float(self.energy_j)
        usable = max(0.0, self.energy_j - self.reserve_frac * self.capacity)
        m.estimated_range_m = float(usable / max(1e-6, self.energy_per_m))
        m.status = self._status()
        m.committed_task_ids = []
        self.pub.publish(m)

    def _status(self) -> int:
        if self.energy_j <= self.reserve_frac * self.capacity:
            return RobotState.STATUS_LOW_BATTERY
        if self.speed > 0.02:
            return RobotState.STATUS_TRANSIT
        return RobotState.STATUS_IDLE


def main(args=None):
    rclpy.init(args=args)
    node = EnergyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().warn(f"shutdown race in spin(): {e}")
    finally:
        node.get_logger().info(
            f"{node.robot_id}: {node.distance_m:.1f} m travelled, "
            f"{node.treatments} treatments, {node.total_spent_j:.0f} J spent "
            f"of {node.capacity:.0f} J")
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()