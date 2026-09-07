from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]


class ExecState(Enum):
    """What the robot is currently doing."""

    LANE = "lane"        # following the assigned boustrophedon path
    DETOUR = "detour"    # driving to a treatment station
    TREAT = "treat"      # stationary, dispensing
    RESUME = "resume"    # returning to the lane path


@dataclass(frozen=True)
class Station:
    """A pose in a lane from which a task is treatable."""

    x: float
    y: float
    lateral_offset_m: float   # distance from the station to the weed


@dataclass(frozen=True)
class Plan:
    """A detour, and where to pick the lane path back up afterwards."""

    waypoints: List[Point]
    station: Station
    resume_index: int


# --------------------------------------------------------------------- geometry


def clearance_from_track_m(
    row_spacing_m: float,
    crop_row_width_m: float,
    wheel_separation_m: float,
    wheel_width_m: float,
) -> float:
    """Free space per side between the outer wheel edge and the crop row.

    Derived from the URDF rather than from a pre-computed half width. The
    binding dimension is the wheel track, not the body: outer track is
    wheel_separation + wheel_width because the wheels straddle the body.

    pure_pursuit.lane_clearance_m() computes the same quantity from an
    already-reduced robot_half_width. The two must agree; a test asserts it.
    """
    corridor = row_spacing_m - crop_row_width_m
    outer_track = wheel_separation_m + wheel_width_m
    return (corridor - outer_track) / 2.0


def lane_centres(n_rows: int, row_spacing_m: float) -> List[float]:
    """Lane centre y values, matching generate_field.py."""
    return [-row_spacing_m / 2.0 + i * row_spacing_m for i in range(n_rows + 1)]


def nearest_lane_centre(y: float, lane_ys: Sequence[float]) -> float:
    if not lane_ys:
        raise ValueError("no lanes")
    return min(lane_ys, key=lambda ly: abs(ly - y))


def same_lane(y_a: float, y_b: float, row_spacing_m: float) -> bool:
    """True when two y values fall in the same lane."""
    return abs(y_a - y_b) < row_spacing_m / 2.0


def headland_crossing_x(
    lane_x_min: float,
    lane_x_max: float,
    margin_m: float,
) -> Tuple[float, float]:
    """The two x values at which lateral movement between lanes is legal.

    Lateral movement is only safe clear of the crop row extent. `margin_m` is
    added beyond each end of the lane span, so this is correct whether the
    lanes CSV already includes the headland (pass 0.0) or stops at the row
    extent (pass the field's headland width).
    """
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative")
    lo, hi = min(lane_x_min, lane_x_max), max(lane_x_min, lane_x_max)
    return (lo - margin_m, hi + margin_m)


# ------------------------------------------------------------------- stations


def treatment_station(
    task_x: float,
    task_y: float,
    lane_y: float,
    spray_reach_m: float,
) -> Optional[Station]:
    """Station in lane `lane_y` from which the task can be treated.

    Returns None when the task is beyond lateral reach, which is the signal
    that the task belongs to a different lane and needs a headland transit.
    """
    lateral = abs(task_y - lane_y)
    if lateral > spray_reach_m:
        return None
    return Station(x=task_x, y=lane_y, lateral_offset_m=lateral)


def needs_headland_transit(
    pose_y: float,
    task_y: float,
    lane_ys: Sequence[float],
    row_spacing_m: float,
    spray_reach_m: float,
) -> bool:
    """True when the task cannot be reached from the robot's current lane.

    Judged against the centre of the lane the robot is in, not its actual y.
    Path-tracking error would otherwise push a reachable task over the
    threshold and send the robot on a headland transit it does not need.
    """
    lane_y = nearest_lane_centre(pose_y, lane_ys)
    return abs(task_y - lane_y) > min(spray_reach_m, row_spacing_m / 2.0)


# ------------------------------------------------------------------ path plans


def index_ahead(
    waypoints: Sequence[Point],
    x: float,
    direction: int,
    start: int = 0,
) -> int:
    """First waypoint at or after `x` along the travel direction.

    `direction` is +1 or -1 along x. Returns len(waypoints) when the whole
    remaining path lies behind the given x.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    for i in range(start, len(waypoints)):
        if (waypoints[i][0] - x) * direction > 0.0:
            return i
    return len(waypoints)


def travel_direction(waypoints: Sequence[Point], index: int) -> int:
    """Sign of travel along x at the given waypoint index.

    Returns +1 when the path runs in +x, -1 otherwise. A serpentine path
    alternates per lane, so this must be read from the path rather than
    assumed.
    """
    if len(waypoints) < 2:
        return 1
    i = min(max(index, 0), len(waypoints) - 2)
    dx = waypoints[i + 1][0] - waypoints[i][0]
    if dx == 0.0:
        for j in range(i + 1, len(waypoints) - 1):
            dx = waypoints[j + 1][0] - waypoints[j][0]
            if dx != 0.0:
                break
    return 1 if dx >= 0.0 else -1


def lane_segment_end(waypoints: Sequence[Point], index: int) -> int:
    """Last index of the contiguous run of waypoints sharing index's y.

    A boustrophedon path is a sequence of lanes. Any search for "the waypoint
    ahead" must stop at the end of the current lane, or it can return a
    waypoint in the NEXT lane, teleporting the robot's resume point across a
    crop row.
    """
    if not waypoints:
        raise ValueError("empty path")
    i = min(max(index, 0), len(waypoints) - 1)
    y = waypoints[i][1]
    j = i
    while j + 1 < len(waypoints) and abs(waypoints[j + 1][1] - y) <= 1e-9:
        j += 1
    return j


def resume_after_station(
    waypoints: Sequence[Point],
    station_x: float,
    current_index: int,
    direction: int,
) -> int:
    """Index at which to rejoin the lane path after treating.

    Bounded by the end of the current lane. When the station lies beyond that
    -- behind the robot, or past the end of the sweep -- the robot rejoins
    where it left off rather than skipping forward.

    The unbounded version of this returned len(waypoints) for a station behind
    the robot, which clamped to the LAST waypoint of the entire sweep: the
    robot drove to the end of the field and every later detour resumed there
    too, so it never covered its lanes.
    """
    if not waypoints:
        raise ValueError("empty path")
    current_index = min(max(current_index, 0), len(waypoints) - 1)
    end = lane_segment_end(waypoints, current_index)
    i = index_ahead(waypoints[:end + 1], station_x, direction, start=current_index)
    return current_index if i > end else i


def select_next_task(
    queue: Sequence[int],
    positions: Mapping[int, Point],
    pose: Point,
    lane_ys: Sequence[float],
    direction: int,
) -> Optional[int]:
    """Which queued task to service next.

    Path order, not arrival order. A FIFO queue sends a robot backwards to a
    task announced minutes ago, then forwards again, so it oscillates instead
    of sweeping. Ranked: reachable from this lane and ahead, nearest first;
    then reachable from this lane behind; then everything else by distance.
    """
    lane_y = nearest_lane_centre(pose[1], lane_ys)

    def rank(task_id: int):
        x, y = positions[task_id]
        same = abs(y - lane_y) <= abs(lane_ys[1] - lane_ys[0]) / 2.0 if len(lane_ys) > 1 else True
        ahead = (x - pose[0]) * direction > 0.0
        tier = 0 if (same and ahead) else (1 if same else 2)
        return (tier, math.dist(pose, (x, y)), task_id)

    known = [t for t in queue if t in positions]
    if not known:
        return None
    return min(known, key=rank)


def plan_in_lane(
    pose: Point,
    task_x: float,
    task_y: float,
    lane_ys: Sequence[float],
    lane_path: Sequence[Point],
    current_index: int,
    direction: int,
    spray_reach_m: float,
) -> Optional[Plan]:
    """Detour to a task reachable from the current lane.

    The detour is a single longitudinal move at the LANE CENTRE, not at the
    robot's current y. Using the pose directly would bake path-tracking error
    into the station: a robot 9 cm off centreline would treat from 9 cm off
    centreline, which is the whole clearance budget.

    The station may lie behind the robot, in which case the returned path
    reverses along the lane.
    """
    lane_y = nearest_lane_centre(pose[1], lane_ys)
    station = treatment_station(task_x, task_y, lane_y, spray_reach_m)
    if station is None:
        return None

    resume = resume_after_station(lane_path, station.x, current_index, direction)
    return Plan(
        waypoints=[(station.x, station.y)],
        station=station,
        resume_index=resume,
    )


def plan_via_headland(
    pose: Point,
    task_x: float,
    task_y: float,
    lane_ys: Sequence[float],
    headland_x: Tuple[float, float],
    lane_path: Sequence[Point],
    current_index: int,
    direction: int,
    spray_reach_m: float,
) -> Optional[Plan]:
    """Detour to a task in a different lane, routing around the crop rows.

    The path leaves via the nearer headland, crosses laterally there, runs down
    the target lane to the task's x, and is expected to return the same way.
    No segment crosses a crop row.
    """
    target_lane = nearest_lane_centre(task_y, lane_ys)
    station = treatment_station(task_x, task_y, target_lane, spray_reach_m)
    if station is None:
        return None

    low, high = min(headland_x), max(headland_x)
    exit_x = low if abs(pose[0] - low) <= abs(pose[0] - high) else high

    waypoints = [
        # Leave along the current lane's centre, not the drifted pose.
        (exit_x, nearest_lane_centre(pose[1], lane_ys)),
        (exit_x, target_lane),      # across, clear of the rows
        (station.x, target_lane),   # down the target lane to the task
    ]
    # Cross-lane work resumes where the robot left its own lane.
    resume = min(max(current_index, 0), len(lane_path) - 1)
    return Plan(waypoints=waypoints, station=station, resume_index=resume)


def return_waypoints(
    station: Station,
    resume_point: Point,
    headland_x: Tuple[float, float],
) -> List[Point]:
    """Path from a treatment station back to a point on the lane path.

    Longitudinal when the station and the resume point share a lane. Otherwise
    the route goes back out via the nearer headland, so no segment crosses a
    crop row.
    """
    if abs(resume_point[1] - station.y) <= 1e-9:
        return [resume_point]

    low, high = min(headland_x), max(headland_x)
    exit_x = low if abs(station.x - low) <= abs(station.x - high) else high
    return [
        (exit_x, station.y),
        (exit_x, resume_point[1]),
        resume_point,
    ]


# ---------------------------------------------------------------- state machine


def advance(
    state: ExecState,
    *,
    has_task: bool,
    at_station: bool,
    dwell_elapsed_s: float,
    treat_duration_s: float,
    at_resume_point: bool,
    failed: bool = False,
) -> ExecState:
    """Single transition of the executor state machine.

    Kept total and side-effect free so that every path is reachable in a test.
    A failed robot holds its state; the node stops publishing heartbeats and
    the allocator re-announces the task.
    """
    if failed:
        return state

    if state is ExecState.LANE:
        return ExecState.DETOUR if has_task else ExecState.LANE

    if state is ExecState.DETOUR:
        return ExecState.TREAT if at_station else ExecState.DETOUR

    if state is ExecState.TREAT:
        if dwell_elapsed_s >= treat_duration_s:
            return ExecState.RESUME
        return ExecState.TREAT

    if state is ExecState.RESUME:
        if at_resume_point:
            return ExecState.DETOUR if has_task else ExecState.LANE
        return ExecState.RESUME

    raise ValueError(f"unhandled state: {state}")


def within(pose: Point, target: Point, tol_m: float) -> bool:
    return math.dist(pose, target) <= tol_m


# ---------------------------------------------------------------------- energy


def path_length_m(points: Sequence[Point]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def detour_cost_j(
    pose: Point,
    plan: Plan,
    resume_point: Point,
    energy_per_m_j: float,
    treat_cost_j: float,
) -> float:
    """Energy for a detour, treatment and return, along the planned path.

    Uses the actual routed path rather than straight-line distance, so it
    exceeds the allocator's Euclidean estimate for cross-lane tasks.
    """
    out = path_length_m([pose] + list(plan.waypoints))
    back = math.dist((plan.station.x, plan.station.y), resume_point)
    return (out + back) * energy_per_m_j + treat_cost_j