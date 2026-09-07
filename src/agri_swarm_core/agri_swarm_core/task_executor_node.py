#!/usr/bin/env python3

from __future__ import annotations
import math
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from agri_swarm_msgs.msg import TaskAnnouncement, TaskAward, Treatment

from agri_swarm_core.pure_pursuit import (
    Pose2D, PursuitLimits, assign_lanes, find_lookahead, is_finished,
    lane_waypoints, load_lanes, pursuit_command,
)
from agri_swarm_core.task_execution import (
    ExecState, advance, headland_crossing_x, lane_centres, needs_headland_transit,
    plan_in_lane, plan_via_headland, return_waypoints, select_next_task,
    travel_direction, within,
)


def yaw_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class TaskExecutor(Node):
    """Drives the lane sweep and interrupts it to service awarded tasks.

    Replaces lane_follower_node for runs with use_allocator:=true. It owns
    cmd_vel outright: two nodes publishing velocity to one robot is a race.

    Frame convention, shared with lane_follower_node and allocator_node.cpp:
    world = origin + odom. Task positions arrive in world coordinates and are
    used unchanged.

    Robots never drive to a weed. Treating means moving along a lane to the
    weed's x and spraying laterally; the outer wheel track (0.34 m) cannot
    enter a 0.22 m crop row. Cross-lane tasks route via a headland.
    """

    def __init__(self):
        super().__init__("task_executor")

        self.declare_parameter("robot_id", "robot_0")
        self.declare_parameter("robot_index", 0)
        self.declare_parameter("n_robots", 4)
        self.declare_parameter("lanes_csv", "")
        self.declare_parameter("waypoint_step_m", 1.0)
        self.declare_parameter("v_nom", 0.6)
        self.declare_parameter("omega_max", 1.2)
        self.declare_parameter("lookahead_m", 0.7)
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)
        self.declare_parameter("row_spacing_m", 0.75)
        self.declare_parameter("n_rows", 10)
        # Lateral reach of the boom. Must be at least row_spacing/2 or the
        # intra-row weeds (70% of the field) are unreachable from any lane.
        self.declare_parameter("spray_reach_m", 0.45)
        self.declare_parameter("treat_duration_s", 2.0)
        self.declare_parameter("station_tol_m", 0.15)
        # Added beyond each end of the lane span to find a legal crossing x.
        # Zero if lanes_csv already includes the headland.
        self.declare_parameter("headland_margin_m", 0.0)
        # Fault injection: freeze this robot at t seconds. Negative disables.
        self.declare_parameter("fail_at_s", -1.0)

        self.robot_id = self.get_parameter("robot_id").value
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
        self.row_spacing = self.get_parameter("row_spacing_m").value
        self.spray_reach = self.get_parameter("spray_reach_m").value
        self.treat_duration = self.get_parameter("treat_duration_s").value
        self.station_tol = self.get_parameter("station_tol_m").value
        self.fail_at_s = self.get_parameter("fail_at_s").value

        if self.spray_reach < self.row_spacing / 2.0:
            self.get_logger().error(
                f"spray_reach_m {self.spray_reach:.2f} is below "
                f"row_spacing/2 {self.row_spacing / 2.0:.2f}. Intra-row weeds "
                "are unreachable from any lane and most of the field cannot "
                "be treated.")

        lanes = load_lanes(lanes_csv)
        mine = assign_lanes(len(lanes), n, idx)
        self.path = lane_waypoints(
            lanes, mine, step_m=self.get_parameter("waypoint_step_m").value)
        self.lane_ys = lane_centres(
            self.get_parameter("n_rows").value, self.row_spacing)
        margin = self.get_parameter("headland_margin_m").value
        self.headland_x = headland_crossing_x(
            min(ln.x_min for ln in lanes),
            max(ln.x_max for ln in lanes),
            margin)
        # Sanity bounds for incoming task positions, one lane spacing of slack.
        self.field_x = (self.headland_x[0] - 1.0, self.headland_x[1] + 1.0)
        self.field_y = (min(self.lane_ys) - self.row_spacing,
                        max(self.lane_ys) + self.row_spacing)

        self.index = 0
        self.pose = None
        self.done = False
        self.failed = False
        self.checked_first_pose = False

        self.state = ExecState.LANE
        self.known = {}        # task_id -> (x, y, confidence)
        self.queue = []        # awarded task_ids, in award order
        self.treated = set()
        self.pending_awards = []   # awards heard before their announcement

        self.detour = []
        self.detour_index = 0
        self.plan = None
        self.active_task = None
        self.resume_point = None
        self.treat_started_s = None
        self.start_s = self._now_s()

        self.get_logger().info(
            f"{self.robot_id}: lanes {mine}, {len(self.path)} waypoints, "
            f"origin ({self.origin_x:+.2f}, {self.origin_y:+.2f}), "
            f"headland crossings at x = {self.headland_x[0]:+.2f} / "
            f"{self.headland_x[1]:+.2f}, spray reach {self.spray_reach:.2f} m")

        self.pub_cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.pub_treat = self.create_publisher(Treatment, "/treatments", 50)
        self.create_subscription(Odometry, "odom", self.on_odom, 10)
        self.create_subscription(
            TaskAnnouncement, "/task_announcements", self.on_announcement, 50)
        self.create_subscription(TaskAward, "/task_awards", self.on_award, 50)
        self.timer = self.create_timer(0.05, self.tick)

    # ------------------------------------------------------------------ clock

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------- pose

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
            if lateral > 0.5:
                self.get_logger().error(
                    f"first waypoint is {lateral:.2f} m LATERAL of the spawn "
                    "pose. origin_y does not match the spawn pose.")

    # ------------------------------------------------------------------ tasks

    def on_announcement(self, a: TaskAnnouncement):
        self.known[a.task_id] = (a.position.x, a.position.y, float(a.confidence))
        if a.task_id in self.pending_awards:
            self.pending_awards.remove(a.task_id)
            self._enqueue(a.task_id)

    def on_award(self, aw: TaskAward):
        if aw.winner_id != self.robot_id:
            return
        if aw.task_id in self.treated or aw.task_id in self.queue:
            return
        if aw.task_id not in self.known:
            # Cross-topic ordering is not guaranteed even under reliable QoS.
            # Buffer rather than drop, or the task is silently never serviced.
            if aw.task_id not in self.pending_awards:
                self.pending_awards.append(aw.task_id)
            return
        self._enqueue(aw.task_id)

    def _enqueue(self, task_id: int):
        if task_id in self.treated or task_id in self.queue:
            return
        self.queue.append(task_id)
        self.get_logger().info(
            f"awarded task {task_id}, queue depth {len(self.queue)}")

    # ------------------------------------------------------------------- tick

    def tick(self):
        if self.pose is None:
            return

        if self.fail_at_s >= 0.0 and not self.failed:
            if self._now_s() - self.start_s >= self.fail_at_s:
                self.failed = True
                self.pub_cmd.publish(Twist())
                self.get_logger().warn(
                    f"{self.robot_id} failed at t+{self.fail_at_s:.1f}s "
                    "(injected); holding position")

        if self.failed:
            self.pub_cmd.publish(Twist())
            return

        here = (self.pose.x, self.pose.y)
        at_station = (
            self.plan is not None
            and within(here, (self.plan.station.x, self.plan.station.y),
                       self.station_tol))
        at_resume = (
            self.resume_point is not None
            and within(here, self.resume_point, self.station_tol))
        dwell = (0.0 if self.treat_started_s is None
                 else self._now_s() - self.treat_started_s)

        new_state = advance(
            self.state,
            has_task=bool(self.queue),
            at_station=at_station,
            dwell_elapsed_s=dwell,
            treat_duration_s=self.treat_duration,
            at_resume_point=at_resume,
            failed=self.failed,
        )
        if new_state is not self.state:
            self._on_transition(self.state, new_state)
            self.state = new_state

        if self.state is ExecState.LANE:
            self._drive_lane()
        elif self.state is ExecState.DETOUR:
            self._drive_list(self.detour)
        elif self.state is ExecState.TREAT:
            self.pub_cmd.publish(Twist())
        elif self.state is ExecState.RESUME:
            self._drive_list(self.detour)

    def _on_transition(self, old: ExecState, new: ExecState):
        if new is ExecState.DETOUR:
            self._begin_detour()
        elif new is ExecState.TREAT:
            self.treat_started_s = self._now_s()
            self.pub_cmd.publish(Twist())
        elif new is ExecState.RESUME:
            self._finish_treatment()
        elif new is ExecState.LANE:
            self.plan = None
            self.resume_point = None
            self.detour = []

    # ---------------------------------------------------------------- detours

    def _in_field(self, x: float, y: float) -> bool:
        return (self.field_x[0] <= x <= self.field_x[1]
                and self.field_y[0] <= y <= self.field_y[1])

    def _begin_detour(self):
        if not self.queue:
            return
        here0 = (self.pose.x, self.pose.y)
        task_id = select_next_task(
            self.queue,
            {t: (self.known[t][0], self.known[t][1])
             for t in self.queue if t in self.known},
            here0, self.lane_ys, travel_direction(self.path, self.index))
        if task_id is None:
            return
        tx, ty, _conf = self.known[task_id]

        # A task outside the field means an upstream frame error, not a weed.
        # Driving to it would take the robot off the headland; drop it and say
        # so, rather than letting a coordinate bug present as poor tracking.
        if not self._in_field(tx, ty):
            self.get_logger().error(
                f"task {task_id} at ({tx:.2f}, {ty:.2f}) is outside the field "
                f"x[{self.field_x[0]:.1f}, {self.field_x[1]:.1f}] "
                f"y[{self.field_y[0]:.1f}, {self.field_y[1]:.1f}]; dropping. "
                "Check that every publisher of task positions is in world frame.")
            self.queue.remove(task_id)
            self.state = ExecState.LANE
            return

        here = (self.pose.x, self.pose.y)
        direction = travel_direction(self.path, self.index)

        if needs_headland_transit(self.pose.y, ty, self.lane_ys,
                                  self.row_spacing, self.spray_reach):
            plan = plan_via_headland(
                here, tx, ty, self.lane_ys, self.headland_x,
                self.path, self.index, direction, self.spray_reach)
        else:
            plan = plan_in_lane(
                here, tx, ty, self.lane_ys, self.path, self.index, direction,
                self.spray_reach)

        if plan is None:
            self.get_logger().warn(
                f"task {task_id} at ({tx:.2f}, {ty:.2f}) is unreachable; "
                "dropping it from the queue")
            self.queue.remove(task_id)
            self.state = ExecState.LANE
            return

        self.plan = plan
        self.active_task = task_id
        self.detour = list(plan.waypoints)
        self.detour_index = 0
        i = min(plan.resume_index, len(self.path) - 1)
        self.resume_point = self.path[i]
        self.index = i

    def _finish_treatment(self):
        task_id = self.active_task
        self.queue.remove(task_id)
        self.active_task = None
        tx, ty, conf = self.known[task_id]
        self.treated.add(task_id)

        m = Treatment()
        m.header.stamp = self.get_clock().now().to_msg()
        m.task_id = task_id
        m.robot_id = self.robot_id
        m.position.x = tx
        m.position.y = ty
        m.confidence = float(conf)
        self.pub_treat.publish(m)
        self.get_logger().info(
            f"treated task {task_id} at ({tx:.2f}, {ty:.2f}) conf={conf:.2f} "
            f"from station ({self.plan.station.x:.2f}, {self.plan.station.y:.2f})")

        self.treat_started_s = None
        self.detour = return_waypoints(
            self.plan.station, self.resume_point, self.headland_x)
        self.detour_index = 0

    # ---------------------------------------------------------------- driving

    def _drive_lane(self):
        if self.done:
            self.pub_cmd.publish(Twist())
            return
        self.index, target = find_lookahead(
            self.path, (self.pose.x, self.pose.y), self.lim.lookahead_m, self.index)
        if is_finished(self.pose, self.path, self.index, self.lim):
            self.done = True
            self.pub_cmd.publish(Twist())
            self.get_logger().info("lane sweep complete")
            return
        self._publish(target)

    def _drive_list(self, points):
        if not points:
            self.pub_cmd.publish(Twist())
            return
        here = (self.pose.x, self.pose.y)
        while (self.detour_index < len(points) - 1
               and within(here, points[self.detour_index], self.station_tol)):
            self.detour_index += 1
        self._publish(points[self.detour_index])

    def _publish(self, target):
        cmd = pursuit_command(self.pose, target, self.lim)
        msg = Twist()
        msg.linear.x = cmd.v
        msg.angular.z = cmd.omega
        self.pub_cmd.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TaskExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # After SIGINT the context is already invalid; publishing or shutting
        # down again raises and the process exits non-zero, which run_batch
        # would read as a failed run.
        if rclpy.ok():
            node.pub_cmd.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()