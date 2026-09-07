#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from agri_swarm_core.pure_pursuit import (
    Pose2D, PursuitLimits, assign_lanes, find_lookahead, is_finished,
    lane_clearance_m, lane_waypoints, load_lanes, pursuit_command,
)


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class LaneFollower(Node):
    """Drives one robot's assigned lanes, in world frame.

    Frame convention, shared with allocator_node.cpp and task_executor_node.py:
    gz DiffDrive odometry reads (0, 0) at the spawn point, so the pose is lifted
    into world frame on arrival with world = origin + odom. Paths, task
    positions and every distance stay in world coordinates throughout.
    """

    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("robot_index", 0)
        self.declare_parameter("n_robots", 4)
        self.declare_parameter("lanes_csv", "")
        self.declare_parameter("waypoint_step_m", 1.0)
        self.declare_parameter("v_nom", 0.6)
        self.declare_parameter("omega_max", 1.2)
        self.declare_parameter("lookahead_m", 0.7)
        # Spawn pose, supplied by the launch file. See the frame convention
        # above; without it the odometry origin is mistaken for the world one.
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)

        idx = self.get_parameter("robot_index").value
        n = self.get_parameter("n_robots").value
        lanes_csv = self.get_parameter("lanes_csv").value
        if not lanes_csv:
            raise RuntimeError("lanes_csv parameter is required")

        self.lim = PursuitLimits(
            v_nom=self.get_parameter("v_nom").value,
            omega_max=self.get_parameter("omega_max").value,
            lookahead_m=self.get_parameter("lookahead_m").value,
        )

        self.origin_x = self.get_parameter("origin_x").value
        self.origin_y = self.get_parameter("origin_y").value

        lanes = load_lanes(lanes_csv)
        mine = assign_lanes(len(lanes), n, idx)
        # World frame, used as-is. The pose is lifted to match in on_odom().
        self.path = lane_waypoints(
            lanes, mine, step_m=self.get_parameter("waypoint_step_m").value)
        self.index = 0
        self.pose = None
        self.done = False
        self.checked_first_pose = False

        clearance = lane_clearance_m()
        self.get_logger().info(
            f"robot {idx}/{n}: lanes {mine}, {len(self.path)} waypoints, "
            f"origin ({self.origin_x:+.2f}, {self.origin_y:+.2f}), "
            f"first waypoint in world "
            f"({self.path[0][0]:+.2f}, {self.path[0][1]:+.2f}), "
            f"lane clearance {clearance * 100:.1f} cm per side")
        if idx != 0 and self.origin_x == 0.0 and self.origin_y == 0.0:
            self.get_logger().error(
                f"robot {idx} has origin_x = origin_y = 0. Odometry is "
                "spawn-relative, so the pose will be wrong by the spawn offset "
                "and the robot will drive across crop rows. The launch file is "
                "not passing the spawn pose.")
        if clearance <= 0.05:
            self.get_logger().warn(
                "lane clearance is under 5 cm. Any odometry drift puts a wheel "
                "in the crop row. This is geometry, not tuning.")

        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.sub = self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.timer = self.create_timer(0.05, self.tick)

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        self.pose = Pose2D(self.origin_x + p.position.x,
                           self.origin_y + p.position.y,
                           yaw_from_quaternion(p.orientation))

        if not self.checked_first_pose:
            self.checked_first_pose = True
            lateral = abs(self.path[0][1] - self.pose.y)
            self.get_logger().info(
                f"first pose: world ({self.pose.x:+.3f}, {self.pose.y:+.3f}) "
                f"= odom ({p.position.x:+.3f}, {p.position.y:+.3f}) "
                f"+ origin ({self.origin_x:+.3f}, {self.origin_y:+.3f}); "
                f"lateral offset to first waypoint {lateral:.3f} m")
            # Reaching the first waypoint must be a longitudinal move. A large
            # lateral offset means the robot would cross crop rows to start.
            if lateral > 0.5:
                self.get_logger().error(
                    f"first waypoint is {lateral:.2f} m LATERAL of the spawn "
                    "pose. The robot will drive across crop rows to reach it. "
                    "origin_y does not match the spawn pose.")

    def tick(self):
        if self.pose is None or self.done:
            return

        self.index, target = find_lookahead(
            self.path, (self.pose.x, self.pose.y), self.lim.lookahead_m, self.index)

        if is_finished(self.pose, self.path, self.index, self.lim):
            self.done = True
            self.pub.publish(Twist())
            self.get_logger().info("lane sweep complete")
            return

        cmd = pursuit_command(self.pose, target, self.lim)
        msg = Twist()
        msg.linear.x = cmd.v
        msg.angular.z = cmd.omega
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.pub.publish(Twist())  # do not leave a robot driving
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()