#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from agri_swarm_msgs.msg import RobotState, Treatment


class EnergyMonitor(Node):
    """Publishes this robot's RobotState heartbeat on /robot_states.

    Two jobs, and the second is the one that was missing:

    1. Energy. Integrates travelled distance at energy_per_m_j and subtracts
       treat_cost_j per treatment, so the allocator's state-of-charge term and
       its reserve gate have something real to read. The two cost constants
       MUST match the allocator's, or predicted and actual consumption diverge
       and the feasibility gate fires at the wrong time.

    2. Liveness. allocator_node's isSilent() treats a robot with no entry in
       last_seen_ as dead. With nothing publishing RobotState, every winner
       looks dead to its peers, every award is re-announced, and every task
       burns all max_rounds and is abandoned. The heartbeat is what makes the
       auction converge at all.

    Fault injection is the absence of this message: after fail_at_s the node
    stops publishing and its peers observe a silent winner.

    Frame convention: world = origin + odom, as everywhere else.
    """

    def __init__(self):
        super().__init__("energy_monitor")

        self.declare_parameter("robot_id", "robot_0")
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)
        self.declare_parameter("rate_hz", 5.0)
        # Must match allocator_node's parameters of the same names.
        self.declare_parameter("energy_capacity_j", 40000.0)
        self.declare_parameter("energy_per_m_j", 12.0)
        self.declare_parameter("treat_cost_j", 30.0)
        self.declare_parameter("reserve_fraction", 0.15)
        # Standby draw, so a stationary robot is not free to run forever.
        self.declare_parameter("idle_w", 0.2)
        # Stop publishing at t seconds to simulate failure. Negative disables.
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
        self.distance_m = 0.0
        self.treatments = 0

        self.pose = None          # world (x, y)
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

        # No heartbeat is the failure signal. Peers see a silent winner and
        # re-announce whatever this robot was holding.
        if self.failed:
            return

        idle_j = self.idle_w * max(0.0, elapsed)
        self.energy_j = max(
            0.0,
            self.capacity - self.spent_travel_j - self.spent_treat_j - idle_j)
        self.spent_travel_j += 0.0   # travel is integrated in the odom callback

        if self.energy_j <= 0.0 and not self.warned_empty:
            self.warned_empty = True
            self.get_logger().warn(
                f"{self.robot_id} is flat after {self.distance_m:.1f} m and "
                f"{self.treatments} treatments. Capacity is likely undersized "
                "for the mission.")

        if self.pose is None:
            return

        # Logged periodically rather than at shutdown: SIGINT invalidates the
        # context before a final log line can be published.
        if now - self.last_report_s >= 30.0:
            self.last_report_s = now
            self.get_logger().info(
                f"travelled {self.distance_m:.1f} m, {self.treatments} treatments, "
                f"{self.capacity - self.energy_j:.0f} J consumed of "
                f"{self.capacity:.0f} J")

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
    finally:
        node.get_logger().info(
            f"{node.robot_id}: {node.distance_m:.1f} m travelled, "
            f"{node.treatments} treatments, "
            f"{node.capacity - node.energy_j:.0f} J of "
            f"{node.capacity:.0f} J consumed")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()