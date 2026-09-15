#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Point = Tuple[float, float]



@dataclass(frozen=True)
class Lane:
    index: int
    y: float
    x_min: float
    x_max: float


def load_lanes(path: str) -> List[Lane]:
    """Read lanes_<seed>.csv into Lane records, ordered by lane_index."""
    lanes: List[Lane] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lanes.append(
                Lane(
                    index=int(row["lane_index"]),
                    y=float(row["y"]),
                    x_min=float(row["x_min"]),
                    x_max=float(row["x_max"]),
                )
            )
    return sorted(lanes, key=lambda ln: ln.index)


def lane_clearance_m(
    row_spacing: float = 0.75,
    crop_row_width: float = 0.22,
    robot_half_width: float = 0.17,
) -> float:
    """Lateral margin from lane centreline to crop, per side.

    robot_half_width default is max(body_w/2, wheel_sep/2 + wheel_w/2)
    = max(0.12, 0.17) = 0.17 from agri_bot.urdf.xacro.

    A negative or near-zero result means the week-1 milestone ("drive the lanes
    without touching a crop row") is geometrically impossible and no amount of
    controller tuning will fix it. Widen row_spacing or narrow the robot.
    """
    return row_spacing / 2.0 - crop_row_width / 2.0 - robot_half_width


def assign_lanes(n_lanes: int, n_robots: int, robot_index: int) -> List[int]:
    """Contiguous, balanced block of lane indices for one robot.

    Contiguous rather than round-robin: round-robin makes every robot cross the
    whole field between lanes, which inflates mission time and energy for no
    benefit and would quietly flatter or punish whichever bid_mode you happen
    to run second.

    Remainder lanes go to the lowest-numbered robots, so the assignment is a
    partition: every lane belongs to exactly one robot, and no lane is dropped.
    """
    if n_robots <= 0:
        raise ValueError("n_robots must be positive")
    if not 0 <= robot_index < n_robots:
        raise ValueError(f"robot_index {robot_index} out of range for {n_robots} robots")
    if n_lanes < 0:
        raise ValueError("n_lanes must be non-negative")

    base, rem = divmod(n_lanes, n_robots)
    start = robot_index * base + min(robot_index, rem)
    count = base + (1 if robot_index < rem else 0)
    return list(range(start, start + count))


def lane_waypoints(
    lanes: Sequence[Lane],
    lane_indices: Sequence[int],
    step_m: float = 1.0,
    serpentine: bool = True,
) -> List[Point]:
    """Boustrophedon path over the assigned lanes.

    Serpentine ordering (down one lane, back along the next) means the transit
    between lanes is one lane-spacing, not one field-length. Turn it off only
    if you want a deliberately worse baseline.
    """
    if step_m <= 0.0:
        raise ValueError("step_m must be positive")

    by_index = {ln.index: ln for ln in lanes}
    path: List[Point] = []

    for k, idx in enumerate(lane_indices):
        if idx not in by_index:
            raise KeyError(f"lane_index {idx} not present in lanes file")
        ln = by_index[idx]

        n_steps = max(1, int(math.ceil((ln.x_max - ln.x_min) / step_m)))
        xs = [ln.x_min + i * (ln.x_max - ln.x_min) / n_steps for i in range(n_steps + 1)]
        if serpentine and (k % 2 == 1):
            xs.reverse()
        path.extend((x, ln.y) for x in xs)

    return path


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float  # radians, +CCW, 0 = +x


@dataclass(frozen=True)
class Command:
    v: float      # m/s
    omega: float  # rad/s, +CCW


@dataclass(frozen=True)
class PursuitLimits:
    v_nom: float = 0.6
    v_min: float = 0.05
    omega_max: float = 1.2
    lookahead_m: float = 0.7
    goal_tolerance_m: float = 0.25
    # Beyond this heading error the robot turns in place rather than arcing.
    # Without it, a robot spawned facing the wrong way carves a wide arc
    # straight through a crop row on the very first control tick.
    turn_in_place_rad: float = 1.05


def find_lookahead(
    path: Sequence[Point],
    position: Point,
    lookahead_m: float,
    start_index: int = 0,
) -> Tuple[int, Point]:
    """Return (index, point) of the first path point at least lookahead away.

    start_index is monotone: never search backwards, or a serpentine path lets
    the robot latch onto the parallel lane it already finished.
    """
    if not path:
        raise ValueError("empty path")

    i = min(max(start_index, 0), len(path) - 1)

    # Advance the anchor to the closest point at or after start_index, so that
    # progress is tracked even when the robot is pushed off the line.
    best_i, best_d = i, math.dist(position, path[i])
    for j in range(i, len(path)):
        d = math.dist(position, path[j])
        if d < best_d:
            best_i, best_d = j, d
        if d > 4.0 * lookahead_m:
            break

    for j in range(best_i, len(path)):
        if math.dist(position, path[j]) >= lookahead_m:
            return j, path[j]

    return len(path) - 1, path[-1]


def pursuit_command(pose: Pose2D, target: Point, lim: PursuitLimits) -> Command:
    """Standard pure pursuit, with an explicit turn-in-place regime."""
    dx = target[0] - pose.x
    dy = target[1] - pose.y

    cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
    x_r = cos_t * dx + sin_t * dy    # forward, robot frame
    y_r = -sin_t * dx + cos_t * dy   # left, robot frame

    dist_sq = x_r * x_r + y_r * y_r
    if dist_sq < 1e-9:
        return Command(0.0, 0.0)

    heading_err = math.atan2(y_r, x_r)

    if abs(heading_err) > lim.turn_in_place_rad:
        omega = math.copysign(lim.omega_max, heading_err)
        return Command(0.0, omega)

    curvature = 2.0 * y_r / dist_sq
    v = lim.v_nom
    omega = v * curvature

    # Preserve the arc when omega saturates: scale v down rather than clipping
    # omega alone, which would straighten the path into the crop.
    if abs(omega) > lim.omega_max:
        scale = lim.omega_max / abs(omega)
        omega = math.copysign(lim.omega_max, omega)
        v = max(lim.v_min, v * scale)

    return Command(v, omega)


def is_finished(pose: Pose2D, path: Sequence[Point], index: int, lim: PursuitLimits) -> bool:
    """True once the final waypoint is reached."""
    return index >= len(path) - 1 and math.dist((pose.x, pose.y), path[-1]) <= lim.goal_tolerance_m
