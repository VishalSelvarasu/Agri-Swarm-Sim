#!/usr/bin/env python3

from __future__ import annotations

import csv
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from agri_swarm_msgs.msg import SwarmEvent, TaskAward, Treatment

# Fixed by analysis/score_run.py. One row per treatment, no aggregation:
# aggregating here is what would force the threshold sweep back into simulation.
FIELDNAMES = ["t_s", "robot_id", "task_id", "x", "y", "confidence"]

# events.csv sits beside treatments.csv and carries the contention record:
# re-announcements with their cause, abandonments, split-brain observations,
# and the awards that form the denominator for any rate derived from them.
#
# One row per OBSERVATION. A single split-brain is reported independently by
# every allocator that holds the task, so counting rows overstates the event
# count by up to n_robots. De-duplicate on (event, task_id, round) in the
# scorer; the observer spread is itself a message-loss indicator.
EVENT_FIELDNAMES = [
    "t_s",
    "event",
    "observer_id",
    "task_id",
    "round",
    "x",
    "y",
    "confidence",
    "winner_a",
    "winner_b",
    "n_bids_a",
    "n_bids_b",
    "detail",
]

EVENT_NAMES = {
    SwarmEvent.EVENT_REANNOUNCE: "reannounce",
    SwarmEvent.EVENT_ABANDON: "abandon",
    SwarmEvent.EVENT_SPLIT_BRAIN: "split_brain",
}


def _stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class TreatmentLogger(Node):
    """Writes treatments.csv and events.csv, and supervises the run.

    One instance per run, not per robot: a single writer avoids interleaved
    partial rows. Each row is flushed on arrival so that a run killed by a
    timeout still leaves a readable file.

    It also ends the mission. Every executor publishes mission_idle; once all
    n_robots have held idle continuously for quiet_period_s, or max_mission_s
    elapses, this node exits. The launch file turns that exit into a shutdown
    of the whole graph, which is what makes run_batch possible: without it a
    run only ends when a human notices four 'lane sweep complete' lines in
    forty nodes' worth of interleaved log.

    The supervisor lives here rather than in a node of its own so that no new
    setup.py entry point is needed. The trade is that a node named for logging
    also decides when the run is over.
    """

    def __init__(self):
        super().__init__("treatment_logger")

        self.declare_parameter("out_csv", "/tmp/treatments.csv")
        # Empty means "events.csv beside treatments.csv", so no launch file
        # needs to know about this file.
        self.declare_parameter("events_csv", "")
        # Zero disables termination entirely and the node runs until Ctrl-C.
        self.declare_parameter("n_robots", 0)
        # Must exceed the longest gap a robot can sit idle mid-mission. A robot
        # that has swept its lanes goes idle between awards, so this needs to
        # outlast the allocator's bid window plus a detour: 30 s is roughly
        # detour_timeout_s and has margin over bid_window_s + award_grace_s.
        self.declare_parameter("quiet_period_s", 30.0)
        # Backstop for a wedged run. Zero disables. In sim-time seconds.
        self.declare_parameter("max_mission_s", 3600.0)

        path = self.get_parameter("out_csv").value
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        events_path = self.get_parameter("events_csv").value
        if not events_path:
            events_path = os.path.join(directory or ".", "events.csv")
        events_dir = os.path.dirname(events_path)
        if events_dir:
            os.makedirs(events_dir, exist_ok=True)

        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDNAMES)
        self.writer.writeheader()
        self.file.flush()
        self.rows = 0

        self.events_file = open(events_path, "w", newline="")
        self.events_writer = csv.DictWriter(
            self.events_file, fieldnames=EVENT_FIELDNAMES)
        self.events_writer.writeheader()
        self.events_file.flush()
        self.event_rows = 0
        self.event_counts = {name: 0 for name in EVENT_NAMES.values()}
        self.event_counts["award"] = 0

        self.create_subscription(Treatment, "/treatments", self.on_treatment, 50)
        self.create_subscription(SwarmEvent, "/swarm_events", self.on_event, 50)
        # The award stream is the denominator. Only the winning allocator
        # publishes an award, so one row here is one awarded (task, round).
        self.create_subscription(TaskAward, "/task_awards", self.on_award, 50)

        self.get_logger().info(f"writing treatments to {path}")
        self.get_logger().info(f"writing events to {events_path}")

        # ------------------------------------------------------- supervision
        self.finished = False
        self.finish_reason = "interrupted"
        self.n_robots = int(self.get_parameter("n_robots").value)
        self.quiet_period_s = float(self.get_parameter("quiet_period_s").value)
        self.max_mission_s = float(self.get_parameter("max_mission_s").value)
        self.idle = {}              # robot_id -> bool, last reported
        self.all_idle_since = None  # sim-time seconds, or None
        self.start_s = self._now_s()

        if self.n_robots > 0:
            for i in range(self.n_robots):
                rid = f"robot_{i}"
                self.create_subscription(
                    Bool, f"/{rid}/mission_idle",
                    self._make_idle_cb(rid), 10)
            self.create_timer(1.0, self.check_mission)
            self.get_logger().info(
                f"supervising {self.n_robots} robots; ending the mission after "
                f"{self.quiet_period_s:.0f}s of all-idle"
                + (f", or {self.max_mission_s:.0f}s elapsed"
                   if self.max_mission_s > 0 else ""))
        else:
            self.get_logger().warn(
                "n_robots is 0, so this run will not end on its own")

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _make_idle_cb(self, robot_id: str):
        def cb(msg: Bool):
            self.idle[robot_id] = bool(msg.data)
        return cb

    def check_mission(self):
        now = self._now_s()
        elapsed = now - self.start_s

        if self.max_mission_s > 0 and elapsed > self.max_mission_s:
            busy = [r for r, v in sorted(self.idle.items()) if not v]
            missing = [f"robot_{i}" for i in range(self.n_robots)
                       if f"robot_{i}" not in self.idle]
            self.get_logger().error(
                f"mission timeout at {elapsed:.0f}s. Still busy: "
                f"{busy or 'none'}. Never reported: {missing or 'none'}.")
            self.finish_reason = f"timeout at {elapsed:.0f}s"
            self.finished = True
            return

        heard_from_all = len(self.idle) >= self.n_robots
        all_idle = heard_from_all and all(self.idle.values())

        if not all_idle:
            self.all_idle_since = None
            return

        if self.all_idle_since is None:
            self.all_idle_since = now
            self.get_logger().info(
                f"all {self.n_robots} robots idle at {elapsed:.0f}s; holding "
                f"for {self.quiet_period_s:.0f}s before ending the mission")
            return

        if now - self.all_idle_since >= self.quiet_period_s:
            self.get_logger().info(
                f"mission complete at {elapsed:.0f}s: every robot idle for "
                f"{self.quiet_period_s:.0f}s")
            self.finish_reason = f"complete at {elapsed:.0f}s"
            self.finished = True

    def on_treatment(self, m: Treatment):
        self.writer.writerow({
            "t_s": f"{_stamp_s(m.header):.3f}",
            "robot_id": m.robot_id,
            "task_id": m.task_id,
            "x": f"{m.position.x:.4f}",
            "y": f"{m.position.y:.4f}",
            "confidence": f"{m.confidence:.4f}",
        })
        self.file.flush()
        self.rows += 1

    def _write_event(self, row: dict):
        self.events_writer.writerow(row)
        self.events_file.flush()
        self.event_rows += 1
        self.event_counts[row["event"]] = self.event_counts.get(row["event"], 0) + 1

    def on_event(self, m: SwarmEvent):
        name = EVENT_NAMES.get(m.event_type, f"unknown_{m.event_type}")
        self._write_event({
            "t_s": f"{_stamp_s(m.header):.3f}",
            "event": name,
            "observer_id": m.observer_id,
            "task_id": m.task_id,
            "round": m.round,
            "x": f"{m.position.x:.4f}",
            "y": f"{m.position.y:.4f}",
            "confidence": f"{m.confidence:.4f}",
            "winner_a": m.winner_a,
            "winner_b": m.winner_b,
            "n_bids_a": m.n_bids_a,
            "n_bids_b": m.n_bids_b,
            "detail": m.detail,
        })

    def on_award(self, m: TaskAward):
        # Position is not carried on TaskAward; task_id joins to the
        # announcement and to treatments.csv, which is enough for scoring.
        self._write_event({
            "t_s": f"{_stamp_s(m.header):.3f}",
            "event": "award",
            "observer_id": m.winner_id,
            "task_id": m.task_id,
            "round": m.round,
            "x": "",
            "y": "",
            "confidence": "",
            "winner_a": m.winner_id,
            "winner_b": "",
            "n_bids_a": m.n_bids_seen,
            "n_bids_b": "",
            "detail": f"utility={m.winning_utility:.4f}",
        })

    def destroy_node(self):
        self.get_logger().info(f"wrote {self.rows} treatments")
        summary = ", ".join(
            f"{k}={v}" for k, v in sorted(self.event_counts.items()))
        self.get_logger().info(f"wrote {self.event_rows} events ({summary})")
        self.get_logger().info(f"run ended: {self.finish_reason}")
        try:
            self.file.close()
        finally:
            try:
                self.events_file.close()
            finally:
                super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TreatmentLogger()
    try:
        # Spun a slice at a time rather than with rclpy.spin(), so that the
        # supervisor can end the run from inside a timer callback without
        # tearing down the context from within that callback.
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # After SIGINT the context is already invalid; shutting down again
        # raises and the process exits non-zero, which run_batch would read as
        # a failed run.
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()