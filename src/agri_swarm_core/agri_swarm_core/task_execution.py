from __future__ import annotations
 
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple
 
Point = Tuple[float, float]
 
 
class ExecState(Enum):
    """What the robot is currently doing."""
 
    LANE = "lane"        # following the assigned boustrophedon path
    DETOUR = "detour"    # driving to a treatment station
    TREAT = "treat"      # stationary, dispensing
    RESUME = "resume"    # returning to the lane path
 
 
@dataclass(frozen=True)
class Station:
    """A pose in the robot's own lane from which a task is treatable."""
 
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
 
 
def lane_clearance_m(
    row_spacing_m: float,
    crop_row_width_m: float,
    wheel_separation_m: float,
    wheel_width_m: float,
) -> float:
    """Free space per side between the outer wheel edge and the crop row.
 
    The binding dimension is the wheel track, not the body: outer track is
    wheel_separation + wheel_width because the wheels straddle the body.
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
    row_spacing_m: float,
    spray_reach_m: float,
) -> bool:
    """True when the task cannot be reached from the robot's current lane."""
    lane_y = pose_y
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
 
 
def plan_in_lane(
    pose: Point,
    task_x: float,
    task_y: float,
    lane_waypoints: Sequence[Point],
    current_index: int,
    direction: int,
    spray_reach_m: float,
) -> Optional[Plan]:
    """Detour to a task reachable from the current lane.
 
    The detour is a single longitudinal move; the y of every waypoint is the
    lane centre the robot is already on, so lane clearance is preserved by
    construction rather than by controller behaviour. The station may lie
    behind the robot, in which case the returned path reverses along the lane.
    """
    lane_y = pose[1]
    station = treatment_station(task_x, task_y, lane_y, spray_reach_m)
    if station is None:
        return None
 
    resume = index_ahead(lane_waypoints, station.x, direction, start=current_index)
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
    lane_waypoints: Sequence[Point],
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
        (exit_x, pose[1]),          # out to the headland in the current lane
        (exit_x, target_lane),      # across, clear of the rows
        (station.x, target_lane),   # down the target lane to the task
    ]
    resume = index_ahead(lane_waypoints, pose[0], direction, start=current_index)
    return Plan(waypoints=waypoints, station=station, resume_index=resume)
 
 
def return_path(station: Station, resume_point: Point, lane_y: float) -> List[Point]:
    """Path from a treatment station back onto the lane path.
 
    When the station and the resume point share a lane the move is purely
    longitudinal. Otherwise the caller is responsible for supplying a headland
    route; this function refuses to produce a lateral crossing.
    """
    if abs(resume_point[1] - station.y) > 1e-9 and abs(station.y - lane_y) > 1e-9:
        raise ValueError("return would cross a crop row; route via the headland")
    return [(resume_point[0], station.y), resume_point]
 
 
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
    legs = [pose] + list(plan.waypoints)
    out = sum(math.dist(legs[i], legs[i + 1]) for i in range(len(legs) - 1))
    back = math.dist((plan.station.x, plan.station.y), resume_point)
    return (out + back) * energy_per_m_j + treat_cost_j
