#!/usr/bin/env python3

from __future__ import annotations

import csv
import os

import rclpy
from rclpy.node import Node

from agri_swarm_msgs.msg import Treatment

# Fixed by analysis/score_run.py. One row per treatment, no aggregation:
# aggregating here is what would force the threshold sweep back into simulation.
FIELDNAMES = ["t_s", "robot_id", "task_id", "x", "y", "confidence"]


class TreatmentLogger(Node):
    """Writes every Treatment message to treatments.csv.

    One instance per run, not per robot: a single writer avoids interleaved
    partial rows. Each row is flushed on arrival so that a run killed by a
    timeout still leaves a readable file.
    """

    def __init__(self):
        super().__init__("treatment_logger")

        self.declare_parameter("out_csv", "/tmp/treatments.csv")
        path = self.get_parameter("out_csv").value

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDNAMES)
        self.writer.writeheader()
        self.file.flush()
        self.rows = 0

        self.create_subscription(Treatment, "/treatments", self.on_treatment, 50)
        self.get_logger().info(f"writing treatments to {path}")

    def on_treatment(self, m: Treatment):
        t_s = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.writer.writerow({
            "t_s": f"{t_s:.3f}",
            "robot_id": m.robot_id,
            "task_id": m.task_id,
            "x": f"{m.position.x:.4f}",
            "y": f"{m.position.y:.4f}",
            "confidence": f"{m.confidence:.4f}",
        })
        self.file.flush()
        self.rows += 1

    def destroy_node(self):
        self.get_logger().info(f"wrote {self.rows} treatments")
        try:
            self.file.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TreatmentLogger()
    try:
        rclpy.spin(node)
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