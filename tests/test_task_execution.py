from __future__ import annotations

import math

import pytest

from agri_swarm_core.pure_pursuit import lane_clearance_m
from agri_swarm_core.task_execution import (
    ExecState,
    Station,
    advance,
    clearance_from_track_m,
    detour_cost_j,
    headland_crossing_x,
    index_ahead,
    lane_centres,
    nearest_lane_centre,
    needs_headland_transit,
    path_length_m,
    plan_in_lane,
    lane_segment_end,
    plan_via_headland,
    resume_after_station,
    return_waypoints,
    select_next_task,
    same_lane,
    travel_direction,
    treatment_station,
    within,
)

# Defaults from generate_field.py and agri_bot.urdf.xacro.
ROW_SPACING = 0.75
CROP_ROW_WIDTH = 0.22
WHEEL_SEP = 0.30
WHEEL_W = 0.04
SPRAY_REACH = 0.45
N_ROWS = 10
HEADLAND = (-1.5, 31.5)
LANES = lane_centres(N_ROWS, ROW_SPACING)


# --------------------------------------------------------------------- geometry


def test_lane_clearance_is_95_mm():
    """The pinned invariant, derived rather than remembered.

    corridor    = 0.75 - 0.22 = 0.530
    outer track = 0.30 + 0.04 = 0.340
    clearance   = (0.530 - 0.340) / 2 = 0.095
    """
    c = clearance_from_track_m(ROW_SPACING, CROP_ROW_WIDTH, WHEEL_SEP, WHEEL_W)
    assert c == pytest.approx(0.095, abs=1e-12)


def test_the_two_clearance_functions_agree():
    """pure_pursuit reduces the robot to a half width; this derives it.

    If they ever disagree, one of them has a stale constant.
    """
    derived = clearance_from_track_m(ROW_SPACING, CROP_ROW_WIDTH, WHEEL_SEP, WHEEL_W)
    reduced = lane_clearance_m(ROW_SPACING, CROP_ROW_WIDTH, WHEEL_SEP / 2.0 + WHEEL_W / 2.0)
    assert derived == pytest.approx(reduced, abs=1e-12)


def test_clearance_goes_negative_on_a_wider_track():
    assert clearance_from_track_m(ROW_SPACING, CROP_ROW_WIDTH, 0.52, WHEEL_W) < 0.0


def test_lane_centres_match_generate_field():
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    assert len(lanes) == N_ROWS + 1 == 11
    assert lanes[0] == pytest.approx(-0.375)
    assert lanes[9] == pytest.approx(6.375)
    assert lanes[-1] == pytest.approx(7.125)


def test_robot_3_first_lane_matches_observed_spawn():
    """Cross-checks the launch spawn pose seen in the first-pose log line."""
    assert lane_centres(N_ROWS, ROW_SPACING)[9] == pytest.approx(6.375)


def test_nearest_lane_centre():
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    assert nearest_lane_centre(0.40, lanes) == pytest.approx(0.375)


def test_nearest_lane_centre_rejects_empty():
    with pytest.raises(ValueError):
        nearest_lane_centre(0.0, [])


def test_same_lane():
    assert same_lane(0.375, 0.40, ROW_SPACING)
    assert not same_lane(0.375, 1.125, ROW_SPACING)


def test_headland_crossing_with_margin():
    assert headland_crossing_x(0.0, 30.0, 1.5) == (pytest.approx(-1.5), pytest.approx(31.5))


def test_headland_crossing_without_margin():
    """Lanes CSV that already includes the headland needs no extra push-out."""
    assert headland_crossing_x(-1.5, 31.5, 0.0) == (pytest.approx(-1.5), pytest.approx(31.5))


def test_headland_crossing_rejects_negative_margin():
    with pytest.raises(ValueError):
        headland_crossing_x(0.0, 30.0, -0.1)


# -------------------------------------------------------------------- stations


def test_intra_row_weed_is_reachable_from_either_adjacent_lane():
    """54 of 79 weeds sit on the crop row, 0.375 m from both neighbouring lanes."""
    assert treatment_station(5.0, 0.0, 0.375, SPRAY_REACH) is not None
    assert treatment_station(5.0, 0.0, -0.375, SPRAY_REACH) is not None


def test_station_stays_in_the_lane():
    s = treatment_station(5.0, 0.0, 0.375, SPRAY_REACH)
    assert s is not None
    assert s.y == pytest.approx(0.375)      # never the weed's y
    assert s.x == pytest.approx(5.0)
    assert s.lateral_offset_m == pytest.approx(0.375)


def test_out_of_reach_returns_none():
    assert treatment_station(5.0, 3.0, 0.375, SPRAY_REACH) is None


def test_spray_reach_must_cover_half_the_row_spacing():
    """Otherwise intra-row weeds are unreachable from any lane."""
    assert SPRAY_REACH >= ROW_SPACING / 2.0


def test_needs_headland_transit():
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    assert not needs_headland_transit(0.375, 0.0, lanes, ROW_SPACING, SPRAY_REACH)
    assert needs_headland_transit(0.375, 2.25, lanes, ROW_SPACING, SPRAY_REACH)


def test_tracking_error_does_not_trigger_a_needless_transit():
    """A robot 9 cm off centreline is still in its lane."""
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    assert not needs_headland_transit(0.470, 0.0, lanes, ROW_SPACING, SPRAY_REACH)


# ------------------------------------------------------- index and direction


def test_index_ahead_forward():
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0, 15.0)]
    assert index_ahead(wps, 7.0, +1) == 2


def test_index_ahead_reverse():
    wps = [(x, 0.375) for x in (15.0, 10.0, 5.0, 0.0)]
    assert index_ahead(wps, 7.0, -1) == 2


def test_index_ahead_past_the_end():
    wps = [(x, 0.375) for x in (0.0, 5.0)]
    assert index_ahead(wps, 99.0, +1) == 2


def test_index_ahead_rejects_bad_direction():
    with pytest.raises(ValueError):
        index_ahead([(0.0, 0.0)], 0.0, 0)


def test_travel_direction_reads_the_path():
    """A serpentine path alternates per lane, so direction cannot be assumed."""
    forward = [(0.0, 0.375), (10.0, 0.375), (20.0, 0.375)]
    backward = [(20.0, 1.125), (10.0, 1.125), (0.0, 1.125)]
    assert travel_direction(forward, 0) == 1
    assert travel_direction(backward, 0) == -1


def test_travel_direction_skips_a_lateral_step():
    """The lane-transition waypoint has no x component."""
    path = [(30.0, 0.375), (30.0, 1.125), (20.0, 1.125)]
    assert travel_direction(path, 0) == -1


def test_travel_direction_on_a_degenerate_path():
    assert travel_direction([(0.0, 0.0)], 0) == 1


# ---------------------------------------------------------------------- plans


def test_in_lane_detour_is_longitudinal_only():
    """The bug class that produced 'lane sweep complete' while crossing crops."""
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0, 15.0, 20.0)]
    plan = plan_in_lane((3.0, 0.375), 8.0, 0.0, LANES, wps, 1, +1, SPRAY_REACH)
    assert plan is not None
    for _, y in plan.waypoints:
        assert y == pytest.approx(0.375)


def test_in_lane_detour_resumes_ahead_of_the_station():
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0, 15.0, 20.0)]
    plan = plan_in_lane((3.0, 0.375), 8.0, 0.0, LANES, wps, 1, +1, SPRAY_REACH)
    assert plan is not None
    assert wps[plan.resume_index][0] > plan.station.x


def test_in_lane_detour_may_reverse():
    """A task behind the robot is still in the lane and still serviceable."""
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0, 15.0)]
    plan = plan_in_lane((12.0, 0.375), 8.0, 0.0, LANES, wps, 2, +1, SPRAY_REACH)
    assert plan is not None
    assert plan.station.x == pytest.approx(8.0)


def test_in_lane_plan_refuses_a_different_lane():
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0)]
    assert plan_in_lane((3.0, 0.375), 8.0, 3.0, LANES, wps, 0, +1, SPRAY_REACH) is None


def test_headland_route_never_crosses_a_row():
    """Lateral movement happens only outside the row extent."""
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    wps = [(x, 0.375) for x in (0.0, 15.0, 30.0)]
    plan = plan_via_headland(
        (10.0, 0.375), 20.0, 3.0, lanes, HEADLAND, wps, 1, +1, SPRAY_REACH)
    assert plan is not None
    lateral = [(a, b) for a, b in zip(plan.waypoints, plan.waypoints[1:])
               if abs(a[1] - b[1]) > 1e-9]
    assert lateral, "expected at least one crossing"
    for a, b in lateral:
        assert a[0] == pytest.approx(b[0])
        assert a[0] <= HEADLAND[0] or a[0] >= HEADLAND[1]


def test_headland_route_exits_by_the_nearer_end():
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    near_start = plan_via_headland(
        (2.0, 0.375), 20.0, 3.0, lanes, HEADLAND, wps, 0, +1, SPRAY_REACH)
    near_end = plan_via_headland(
        (29.0, 0.375), 20.0, 3.0, lanes, HEADLAND, wps, 0, +1, SPRAY_REACH)
    assert near_start is not None and near_end is not None
    assert near_start.waypoints[0][0] == pytest.approx(-1.5)
    assert near_end.waypoints[0][0] == pytest.approx(31.5)


def test_headland_plan_targets_the_task_lane():
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    plan = plan_via_headland(
        (10.0, 0.375), 20.0, 3.0, lanes, HEADLAND, wps, 0, +1, SPRAY_REACH)
    assert plan is not None
    assert plan.station.y == pytest.approx(nearest_lane_centre(3.0, lanes))


def test_return_is_longitudinal_within_a_lane():
    s = Station(x=8.0, y=0.375, lateral_offset_m=0.375)
    for _, y in return_waypoints(s, (10.0, 0.375), HEADLAND):
        assert y == pytest.approx(0.375)


def test_return_from_another_lane_goes_via_the_headland():
    s = Station(x=8.0, y=3.375, lateral_offset_m=0.375)
    path = return_waypoints(s, (10.0, 0.375), HEADLAND)
    lateral = [(a, b) for a, b in zip(path, path[1:]) if abs(a[1] - b[1]) > 1e-9]
    assert lateral
    for a, b in lateral:
        assert a[0] == pytest.approx(b[0])
        assert a[0] <= HEADLAND[0] or a[0] >= HEADLAND[1]


def test_return_never_cuts_diagonally():
    """Every segment is axis-aligned, so no path clips a row corner."""
    s = Station(x=8.0, y=3.375, lateral_offset_m=0.375)
    path = [(8.0, 3.375)] + return_waypoints(s, (10.0, 0.375), HEADLAND)
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) < 1e-9 or abs(a[1] - b[1]) < 1e-9


# --------------------------------------------------------------- state machine


def test_lane_holds_without_a_task():
    assert advance(
        ExecState.LANE, has_task=False, at_station=False, dwell_elapsed_s=0.0,
        treat_duration_s=2.0, at_resume_point=False
    ) is ExecState.LANE


def test_award_interrupts_the_lane_sweep():
    assert advance(
        ExecState.LANE, has_task=True, at_station=False, dwell_elapsed_s=0.0,
        treat_duration_s=2.0, at_resume_point=False
    ) is ExecState.DETOUR


def test_detour_holds_until_the_station_is_reached():
    assert advance(
        ExecState.DETOUR, has_task=True, at_station=False, dwell_elapsed_s=0.0,
        treat_duration_s=2.0, at_resume_point=False
    ) is ExecState.DETOUR


def test_treatment_waits_out_the_dwell():
    kw = dict(has_task=True, at_station=True, treat_duration_s=2.0, at_resume_point=False)
    assert advance(ExecState.TREAT, dwell_elapsed_s=1.9, **kw) is ExecState.TREAT
    assert advance(ExecState.TREAT, dwell_elapsed_s=2.0, **kw) is ExecState.RESUME


def test_queued_task_is_taken_without_returning_to_lane_following():
    assert advance(
        ExecState.RESUME, has_task=True, at_station=False, dwell_elapsed_s=0.0,
        treat_duration_s=2.0, at_resume_point=True
    ) is ExecState.DETOUR


def test_empty_queue_returns_to_the_lane():
    assert advance(
        ExecState.RESUME, has_task=False, at_station=False, dwell_elapsed_s=0.0,
        treat_duration_s=2.0, at_resume_point=True
    ) is ExecState.LANE


def test_failed_robot_freezes():
    """The allocator detects this by absent heartbeats, not by a state change."""
    for s in ExecState:
        assert advance(
            s, has_task=True, at_station=True, dwell_elapsed_s=99.0,
            treat_duration_s=2.0, at_resume_point=True, failed=True
        ) is s


def test_every_state_is_handled():
    for s in ExecState:
        advance(
            s, has_task=False, at_station=False, dwell_elapsed_s=0.0,
            treat_duration_s=1.0, at_resume_point=False
        )


def test_within():
    assert within((0.0, 0.0), (0.05, 0.0), 0.1)
    assert not within((0.0, 0.0), (0.5, 0.0), 0.1)


# --------------------------------------------------------------------- energy


def test_path_length():
    assert path_length_m([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]) == pytest.approx(7.0)


def test_routed_cost_exceeds_the_euclidean_estimate_across_lanes():
    """The allocator bids hypot; the real path goes round. Quantifies the gap."""
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    pose = (15.0, 0.375)
    task = (15.0, 2.25)
    plan = plan_via_headland(
        pose, task[0], task[1], lanes, HEADLAND, wps, 0, +1, SPRAY_REACH)
    assert plan is not None
    euclidean = math.dist(pose, task) * 12.0 + 30.0
    routed = detour_cost_j(pose, plan, (15.0, 0.375), 12.0, 30.0)
    assert routed > euclidean * 5.0


def test_in_lane_cost_is_close_to_euclidean():
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    pose = (10.0, 0.375)
    plan = plan_in_lane(pose, 12.0, 0.0, LANES, wps, 0, +1, SPRAY_REACH)
    assert plan is not None
    routed = detour_cost_j(pose, plan, (12.0, 0.375), 12.0, 30.0)
    assert routed == pytest.approx(2.0 * 12.0 + 30.0)


def test_station_snaps_to_the_lane_centre():
    """Observed live: 'treated ... from station (3.22, -0.47)'.

    Lane centres are -0.375 and +0.375. A station at -0.47 is 9.5 cm off, the
    entire clearance budget, because the plan used the robot's drifted y
    instead of the lane centre.
    """
    wps = [(x, -0.375) for x in (0.0, 10.0, 20.0)]
    plan = plan_in_lane((3.22, -0.47), 3.22, -0.43, LANES, wps, 0, +1, SPRAY_REACH)
    assert plan is not None
    assert plan.station.y == pytest.approx(-0.375)


def test_headland_exit_leg_uses_the_lane_centre():
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    plan = plan_via_headland(
        (10.0, 0.47), 20.0, 3.0, LANES, HEADLAND, wps, 0, +1, SPRAY_REACH)
    assert plan is not None
    assert plan.waypoints[0][1] == pytest.approx(0.375)


# ------------------------------------------------- resume index (regression)

SERPENTINE = (
    [(x, -0.375) for x in (0.0, 10.0, 20.0, 30.0)]
    + [(x, 0.375) for x in (30.0, 20.0, 10.0, 0.0)]
)


def test_lane_segment_end_stops_at_the_turn():
    assert lane_segment_end(SERPENTINE, 0) == 3
    assert lane_segment_end(SERPENTINE, 4) == 7


def test_resume_never_jumps_to_the_end_of_the_sweep():
    """Observed live: robot_0 treated 4 of 95 queued tasks and stopped moving.

    index_ahead returned len(path) for a station behind the robot, which then
    clamped to the final waypoint of the WHOLE sweep. The robot drove to the
    end of the field and resumed there for every subsequent detour.
    """
    i = resume_after_station(SERPENTINE, station_x=5.0, current_index=2, direction=+1)
    assert i < len(SERPENTINE) - 1
    assert i == 2                      # station behind: carry on where we were


def test_resume_stays_inside_the_current_lane():
    """Without the lane bound, a forward search crosses into the next lane."""
    i = resume_after_station(SERPENTINE, station_x=25.0, current_index=1, direction=+1)
    assert SERPENTINE[i][1] == pytest.approx(-0.375)


def test_resume_ahead_of_the_station_when_there_is_room():
    i = resume_after_station(SERPENTINE, station_x=15.0, current_index=1, direction=+1)
    assert SERPENTINE[i][0] > 15.0
    assert SERPENTINE[i][1] == pytest.approx(-0.375)


def test_plan_never_resumes_past_the_lane():
    plan = plan_in_lane((20.0, -0.375), 5.0, -0.1, LANES, SERPENTINE, 2, +1, SPRAY_REACH)
    assert plan is not None
    assert SERPENTINE[plan.resume_index][1] == pytest.approx(-0.375)


# --------------------------------------------------------- task selection

def test_selection_prefers_work_ahead_in_the_same_lane():
    """FIFO sends the robot backwards to an old award, then forwards again."""
    pos = {1: (2.0, -0.375), 2: (12.0, -0.375), 3: (25.0, -0.375)}
    assert select_next_task([1, 2, 3], pos, (10.0, -0.375), LANES, +1) == 2


def test_selection_falls_back_to_work_behind():
    pos = {1: (2.0, -0.375), 2: (4.0, -0.375)}
    assert select_next_task([1, 2], pos, (10.0, -0.375), LANES, +1) == 2


def test_selection_deprioritises_other_lanes():
    pos = {1: (11.0, 3.375), 2: (20.0, -0.375)}
    assert select_next_task([1, 2], pos, (10.0, -0.375), LANES, +1) == 2


def test_selection_ignores_tasks_with_no_position():
    pos = {2: (12.0, -0.375)}
    assert select_next_task([1, 2], pos, (10.0, -0.375), LANES, +1) == 2


def test_selection_on_an_empty_queue():
    assert select_next_task([], {}, (0.0, -0.375), LANES, +1) is None


def test_selection_is_deterministic_on_ties():
    pos = {7: (12.0, -0.375), 3: (12.0, -0.375)}
    assert select_next_task([7, 3], pos, (10.0, -0.375), LANES, +1) == 3


# ============================================================================
# decide(): the executor transition table.
#
# Every bug in the node layer so far has been here, and none of them were
# caught by a test. These cover the four observed failures directly.
# ============================================================================

from agri_swarm_core.task_execution import Trigger, decide, worth_detouring

BASE = dict(
    has_task=False, at_station=False, at_resume=False,
    dwell_elapsed_s=0.0, treat_duration_s=2.0,
    detour_elapsed_s=0.0, detour_timeout_s=20.0,
    resume_elapsed_s=0.0, resume_timeout_s=20.0,
)


def d(state, **kw):
    return decide(state, **{**BASE, **kw})


def test_lane_holds_without_work():
    assert d(ExecState.LANE) == (ExecState.LANE, Trigger.NONE)


def test_lane_takes_a_task():
    assert d(ExecState.LANE, has_task=True) == (ExecState.DETOUR, Trigger.TAKE_TASK)


def test_detour_arrives():
    assert d(ExecState.DETOUR, has_task=True, at_station=True) == (
        ExecState.TREAT, Trigger.ARRIVED)


def test_detour_times_out_to_lane():
    """Observed live: robots sat in DETOUR for minutes with no plan."""
    assert d(ExecState.DETOUR, has_task=True, detour_elapsed_s=21.0) == (
        ExecState.LANE, Trigger.DETOUR_TIMEOUT)


def test_arrival_beats_the_timeout():
    assert d(ExecState.DETOUR, at_station=True, detour_elapsed_s=99.0)[0] is ExecState.TREAT


def test_treat_waits_the_dwell():
    assert d(ExecState.TREAT, dwell_elapsed_s=1.9)[0] is ExecState.TREAT
    assert d(ExecState.TREAT, dwell_elapsed_s=2.0) == (ExecState.RESUME, Trigger.DWELL_DONE)


def test_resume_rejoins_the_lane():
    assert d(ExecState.RESUME, at_resume=True) == (ExecState.LANE, Trigger.REJOINED)


def test_resume_chains_into_the_next_task():
    assert d(ExecState.RESUME, at_resume=True, has_task=True) == (
        ExecState.DETOUR, Trigger.TAKE_TASK)


def test_resume_times_out():
    """RESUME was unbounded: a robot that could not reach its resume point
    drove at it forever and nothing in the log said so."""
    assert d(ExecState.RESUME, resume_elapsed_s=21.0) == (
        ExecState.LANE, Trigger.RESUME_TIMEOUT)


def test_failed_robot_holds_every_state():
    for s in ExecState:
        assert d(s, failed=True, has_task=True, at_station=True,
                 dwell_elapsed_s=99.0, at_resume=True) == (s, Trigger.NONE)


def test_decide_is_total():
    for s in ExecState:
        for has in (False, True):
            for st in (False, True):
                for rs in (False, True):
                    decide(s, **{**BASE, "has_task": has, "at_station": st,
                                 "at_resume": rs})


def test_every_trigger_is_reachable():
    seen = set()
    for s in ExecState:
        for kw in ({}, {"has_task": True}, {"at_station": True},
                   {"at_resume": True}, {"dwell_elapsed_s": 9.0},
                   {"detour_elapsed_s": 99.0}, {"resume_elapsed_s": 99.0},
                   {"at_resume": True, "has_task": True}):
            seen.add(d(s, **kw)[1])
    assert seen == set(Trigger)


# ---------------------------------------------------------------- detour cap

def test_near_work_is_worth_a_detour():
    assert worth_detouring((10.0, 0.375), (12.0, 0.0), 15.0)


def test_distant_work_is_not():
    """Without this the queue never empties, LANE is exited every tick, and
    the sweep never progresses -- 470 m travelled, zero sweeps completed."""
    assert not worth_detouring((10.0, 0.375), (29.0, 0.0), 15.0)


def test_selection_skips_work_beyond_the_cap():
    pos = {1: (28.0, -0.375), 2: (12.0, -0.375)}
    assert select_next_task([1, 2], pos, (10.0, -0.375), LANES, +1, 15.0) == 2


def test_selection_returns_none_when_all_work_is_far():
    pos = {1: (28.0, -0.375)}
    assert select_next_task([1], pos, (10.0, -0.375), LANES, +1, 5.0) is None


def test_selection_without_a_cap_is_unchanged():
    pos = {1: (28.0, -0.375)}
    assert select_next_task([1], pos, (10.0, -0.375), LANES, +1) == 1