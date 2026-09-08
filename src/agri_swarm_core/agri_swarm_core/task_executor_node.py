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
    ExecState, Trigger, decide, headland_crossing_x, lane_centres,
    needs_headland_transit, plan_in_lane, plan_via_headland, return_waypoints,
    select_next_task, travel_direction, within,
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

    All transition logic lives in task_execution.decide(), which is pure and
    unit-tested. This class only reads sensors, applies the side effects the
    trigger implies, and publishes velocity.
    """

    def __init__(self):
        super().__init__("task_executor")

        # ------------------------------------------------------- parameters
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
        # Must not be tighter than PursuitLimits.goal_tolerance_m (0.25), or
        # pure pursuit circles the station without arriving and the state
        # machine never leaves DETOUR.
        self.declare_parameter("station_tol_m", 0.30)
        # Must exceed max_detour_m / v_nom, or legitimate detours are aborted
        # before they finish: 12 m at 0.6 m/s is 20 s one way.
        self.declare_parameter("detour_timeout_s", 30.0)
        # RESUME is bounded for the same reason DETOUR is. A robot that cannot
        # reach its resume point otherwise drives at it for the whole run.
        self.declare_parameter("resume_timeout_s", 30.0)
        # Only interrupt the sweep for work within this radius. The queue never
        # empties while the detector runs, so without a cap the robot leaves
        # LANE on the tick after entering it and never sweeps its lanes.
        self.declare_parameter("max_detour_m", 12.0)
        # Added beyond each end of the lane span to find a legal crossing x.
        # Zero if lanes_csv already includes the headland.
        self.declare_parameter("headland_margin_m", 0.0)
        # Fault injection: freeze this robot at t seconds. Negative disables.
        self.declare_parameter("fail_at_s", -1.0)

        # ------------------------------------------------------------ config
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
        self.detour_timeout = self.get_parameter("detour_timeout_s").value
        self.resume_timeout = self.get_parameter("resume_timeout_s").value
        self.max_detour = self.get_parameter("max_detour_m").value
        self.fail_at_s = self.get_parameter("fail_at_s").value

        if self.spray_reach < self.row_spacing / 2.0:
            self.get_logger().error(
                f"spray_reach_m {self.spray_reach:.2f} is below "
                f"row_spacing/2 {self.row_spacing / 2.0:.2f}. Intra-row weeds "
                "are unreachable from any lane and most of the field cannot "
                "be treated.")
        if self.detour_timeout * self.lim.v_nom < self.max_detour:
            self.get_logger().warn(
                f"detour_timeout_s {self.detour_timeout:.0f} is shorter than "
                f"max_detour_m {self.max_detour:.0f} / v_nom "
                f"{self.lim.v_nom:.2f} = "
                f"{self.max_detour / self.lim.v_nom:.0f}s. Legitimate detours "
                "will be aborted before they arrive.")

        # ---------------------------------------------------------- geometry
        lanes = load_lanes(lanes_csv)
        mine = assign_lanes(len(lanes), n, idx)
        self.path = lane_waypoints(
            lanes, mine, step_m=self.get_parameter("waypoint_step_m").value)
        self.lane_ys = lane_centres(
            self.get_parameter("n_rows").value, self.row_spacing)
        self.headland_x = headland_crossing_x(
            min(ln.x_min for ln in lanes),
            max(ln.x_max for ln in lanes),
            self.get_parameter("headland_margin_m").value)
        # Sanity bounds for incoming task positions, one lane spacing of slack.
        self.field_x = (self.headland_x[0] - 1.0, self.headland_x[1] + 1.0)
        self.field_y = (min(self.lane_ys) - self.row_spacing,
                        max(self.lane_ys) + self.row_spacing)

        # ------------------------------------------------------------- state
        self.index = 0
        self.pose = None
        self.done = False
        self.failed = False
        self.checked_first_pose = False

        self.state = ExecState.LANE
        self.known = {}            # task_id -> (x, y, confidence)
        self.queue = []            # awarded task_ids, serviced in path order
        self.treated = set()
        self.pending_awards = []   # awards heard before their announcement

        self.plan = None
        self.active_task = None
        self.resume_point = None
        self.detour = []
        self.detour_index = 0
        self.detour_started_s = None
        self.resume_started_s = None
        self.treat_started_s = None
        self.timeouts = 0
        self.start_s = self._now_s()

        self.get_logger().info(
            f"{self.robot_id}: lanes {mine}, {len(self.path)} waypoints, "
            f"origin ({self.origin_x:+.2f}, {self.origin_y:+.2f}), "
            f"headland crossings at x = {self.headland_x[0]:+.2f} / "
            f"{self.headland_x[1]:+.2f}, spray reach {self.spray_reach:.2f} m, "
            f"detour cap {self.max_detour:.0f} m")

        # ---------------------------------------------------------------- io
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
                self.get_logger().warn(
                    f"{self.robot_id} failed at t+{self.fail_at_s:.1f}s "
                    "(injected); holding position")

        if self.failed:
            self.pub_cmd.publish(Twist())
            return

        now = self._now_s()
        here = (self.pose.x, self.pose.y)

        state, trigger = decide(
            self.state,
            has_task=bool(self.queue),
            at_station=(self.plan is not None and within(
                here, (self.plan.station.x, self.plan.station.y),
                self.station_tol)),
            at_resume=(self.resume_point is not None
                       and within(here, self.resume_point, self.station_tol)),
            dwell_elapsed_s=(0.0 if self.treat_started_s is None
                             else now - self.treat_started_s),
            treat_duration_s=self.treat_duration,
            detour_elapsed_s=(0.0 if self.detour_started_s is None
                              else now - self.detour_started_s),
            detour_timeout_s=self.detour_timeout,
            resume_elapsed_s=(0.0 if self.resume_started_s is None
                              else now - self.resume_started_s),
            resume_timeout_s=self.resume_timeout,
            failed=self.failed,
        )

        # Commit the state BEFORE applying side effects. The previous order let
        # a failed detour set-up write ExecState.LANE and then be overwritten
        # by the assignment, pinning the robot in DETOUR with no plan until the
        # timeout fired.
        self.state = state
        if trigger is not Trigger.NONE:
            self._apply(trigger, now)

        if self.state is ExecState.LANE:
            self._drive_lane()
        elif self.state is ExecState.TREAT:
            self.pub_cmd.publish(Twist())
        else:
            self._drive_list(self.detour)

    def _apply(self, trigger: Trigger, now: float):
        if trigger is Trigger.TAKE_TASK:
            self.detour_started_s = now
            self.resume_started_s = None
            if not self._begin_detour():
                # Nothing serviceable right now. Sweeping is always a valid
                # fallback, so drop straight back into the lane.
                self._clear_detour()
                self.state = ExecState.LANE

        elif trigger is Trigger.ARRIVED:
            self.treat_started_s = now
            self.pub_cmd.publish(Twist())

        elif trigger is Trigger.DWELL_DONE:
            self._finish_treatment()
            self.resume_started_s = now
            self.detour_started_s = None

        elif trigger is Trigger.REJOINED:
            self._clear_detour()

        elif trigger in (Trigger.DETOUR_TIMEOUT, Trigger.RESUME_TIMEOUT):
            self.timeouts += 1
            self.get_logger().warn(
                f"{trigger.value} on task {self.active_task}; rejoining the "
                f"lane ({self.timeouts} so far)")
            if self.active_task is not None and self.active_task in self.queue:
                self.queue.remove(self.active_task)
            self._clear_detour()

    def _clear_detour(self):
        self.plan = None
        self.active_task = None
        self.resume_point = None
        self.detour = []
        self.detour_index = 0
        self.detour_started_s = None
        self.resume_started_s = None
        self.treat_started_s = None

    # ---------------------------------------------------------------- detours

    def _in_field(self, x: float, y: float) -> bool:
        return (self.field_x[0] <= x <= self.field_x[1]
                and self.field_y[0] <= y <= self.field_y[1])

    def _begin_detour(self) -> bool:
        """Set up a detour. Returns False when there is nothing to service."""
        if not self.queue or self.pose is None:
            return False

        here = (self.pose.x, self.pose.y)
        direction = travel_direction(self.path, self.index)
        task_id = select_next_task(
            self.queue,
            {t: (self.known[t][0], self.known[t][1])
             for t in self.queue if t in self.known},
            here, self.lane_ys, direction, self.max_detour)
        if task_id is None:
            return False

        tx, ty, _conf = self.known[task_id]

        # A task outside the field means an upstream frame error, not a weed.
        # Driving to it would take the robot off the headland.
        if not self._in_field(tx, ty):
            self.get_logger().error(
                f"task {task_id} at ({tx:.2f}, {ty:.2f}) is outside the field "
                f"x[{self.field_x[0]:.1f}, {self.field_x[1]:.1f}] "
                f"y[{self.field_y[0]:.1f}, {self.field_y[1]:.1f}]; dropping. "
                "Check that every publisher of task positions is in world frame.")
            self.queue.remove(task_id)
            return False

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
            return False

        self.plan = plan
        self.active_task = task_id
        self.detour = list(plan.waypoints)
        self.detour_index = 0
        # Never rewind the sweep: the lane index is monotone.
        i = min(plan.resume_index, len(self.path) - 1)
        self.resume_point = self.path[i]
        self.index = i
        return True

    def _finish_treatment(self):
        # The task being treated, not the head of the queue: selection is by
        # path order, so the queue is not FIFO and popping index 0 would emit
        # a treatment for a weed the robot never visited.
        task_id = self.active_task
        if task_id is None or self.plan is None or task_id not in self.known:
            return
        if task_id in self.queue:
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
        if self.resume_point is not None:
            self.detour = return_waypoints(
                self.plan.station, self.resume_point, self.headland_x)
        else:
            self.detour = []
        self.detour_index = 0

    # ---------------------------------------------------------------- driving

    def _drive_lane(self):
        if self.done:
            self.pub_cmd.publish(Twist())
            return
        self.index, target = find_lookahead(
            self.path, (self.pose.x, self.pose.y), self.lim.lookahead_m,
            self.index)
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
        node.get_logger().info(
            f"{node.robot_id}: {len(node.treated)} treated, "
            f"{len(node.queue)} queued, {node.timeouts} timeouts, "
            f"lane index {node.index}/{len(node.path)}")
        # After SIGINT the context is already invalid; publishing or shutting
        # down again raises and the process exits non-zero, which run_batch
        # would read as a failed run.
        if rclpy.ok():
            node.pub_cmd.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()