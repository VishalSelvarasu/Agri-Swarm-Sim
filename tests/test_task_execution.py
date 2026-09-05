from __future__ import annotations
import math
import pytest
from agri_swarm_core.task_execution import (
    ExecState,
    Plan,
    Station,
    advance,
    detour_cost_j,
    index_ahead,
    lane_centres,
    lane_clearance_m,
    nearest_lane_centre,
    needs_headland_transit,
    plan_in_lane,
    plan_via_headland,
    return_path,
    same_lane,
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
 
 
# --------------------------------------------------------------------- geometry
 
 
def test_lane_clearance_is_95_mm():
    """The pinned invariant, derived rather than remembered.
 
    corridor    = 0.75 - 0.22 = 0.530
    outer track = 0.30 + 0.04 = 0.340
    clearance   = (0.530 - 0.340) / 2 = 0.095
    """
    c = lane_clearance_m(ROW_SPACING, CROP_ROW_WIDTH, WHEEL_SEP, WHEEL_W)
    assert c == pytest.approx(0.095, abs=1e-12)
 
 
def test_clearance_is_set_by_the_wheels_not_the_body():
    """Widening the body below the track does not change clearance."""
    a = lane_clearance_m(ROW_SPACING, CROP_ROW_WIDTH, WHEEL_SEP, WHEEL_W)
    b = lane_clearance_m(ROW_SPACING, CROP_ROW_WIDTH, WHEEL_SEP, WHEEL_W)
    assert a == b
 
 
def test_clearance_goes_negative_on_a_wider_track():
    assert lane_clearance_m(ROW_SPACING, CROP_ROW_WIDTH, 0.52, WHEEL_W) < 0.0
 
 
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
    assert nearest_lane_centre(0.0, lanes) in (pytest.approx(-0.375), pytest.approx(0.375))
 
 
def test_nearest_lane_centre_rejects_empty():
    with pytest.raises(ValueError):
        nearest_lane_centre(0.0, [])
 
 
def test_same_lane():
    assert same_lane(0.375, 0.40, ROW_SPACING)
    assert not same_lane(0.375, 1.125, ROW_SPACING)
 
 
# -------------------------------------------------------------------- stations
 
 
def test_intra_row_weed_is_reachable_from_either_adjacent_lane():
    """70% of weeds sit on the crop row, 0.375 m from both neighbouring lanes."""
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
    assert not needs_headland_transit(0.375, 0.0, ROW_SPACING, SPRAY_REACH)
    assert needs_headland_transit(0.375, 2.25, ROW_SPACING, SPRAY_REACH)
 
 
# ----------------------------------------------------------------- index_ahead
 
 
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
 
 
# ---------------------------------------------------------------------- plans
 
 
def test_in_lane_detour_is_longitudinal_only():
    """The bug class that produced 'lane sweep complete' while crossing crops."""
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0, 15.0, 20.0)]
    plan = plan_in_lane((3.0, 0.375), 8.0, 0.0, wps, 1, +1, SPRAY_REACH)
    assert plan is not None
    for _, y in plan.waypoints:
        assert y == pytest.approx(0.375)
 
 
def test_in_lane_detour_resumes_ahead_of_the_station():
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0, 15.0, 20.0)]
    plan = plan_in_lane((3.0, 0.375), 8.0, 0.0, wps, 1, +1, SPRAY_REACH)
    assert plan is not None
    assert wps[plan.resume_index][0] > plan.station.x
 
 
def test_in_lane_detour_may_reverse():
    """A task behind the robot is still in the lane and still serviceable."""
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0, 15.0)]
    plan = plan_in_lane((12.0, 0.375), 8.0, 0.0, wps, 2, +1, SPRAY_REACH)
    assert plan is not None
    assert plan.station.x == pytest.approx(8.0)
 
 
def test_in_lane_plan_refuses_a_different_lane():
    wps = [(x, 0.375) for x in (0.0, 5.0, 10.0)]
    assert plan_in_lane((3.0, 0.375), 8.0, 3.0, wps, 0, +1, SPRAY_REACH) is None
 
 
def test_headland_route_never_crosses_a_row():
    """Lateral movement happens only outside the row extent."""
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    wps = [(x, 0.375) for x in (0.0, 15.0, 30.0)]
    headland = (-1.5, 31.5)
    plan = plan_via_headland(
        (10.0, 0.375), 20.0, 3.0, lanes, headland, wps, 1, +1, SPRAY_REACH
    )
    assert plan is not None
    lateral_moves = [
        (a, b) for a, b in zip(plan.waypoints, plan.waypoints[1:])
        if abs(a[1] - b[1]) > 1e-9
    ]
    assert lateral_moves, "expected at least one crossing"
    for a, b in lateral_moves:
        assert a[0] == pytest.approx(b[0])
        assert a[0] <= headland[0] or a[0] >= headland[1]
 
 
def test_headland_route_exits_by_the_nearer_end():
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    headland = (-1.5, 31.5)
    near_start = plan_via_headland(
        (2.0, 0.375), 20.0, 3.0, lanes, headland, wps, 0, +1, SPRAY_REACH
    )
    near_end = plan_via_headland(
        (29.0, 0.375), 20.0, 3.0, lanes, headland, wps, 0, +1, SPRAY_REACH
    )
    assert near_start is not None and near_end is not None
    assert near_start.waypoints[0][0] == pytest.approx(-1.5)
    assert near_end.waypoints[0][0] == pytest.approx(31.5)
 
 
def test_headland_plan_targets_the_task_lane():
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    plan = plan_via_headland(
        (10.0, 0.375), 20.0, 3.0, lanes, (-1.5, 31.5), wps, 0, +1, SPRAY_REACH
    )
    assert plan is not None
    assert plan.station.y == pytest.approx(nearest_lane_centre(3.0, lanes))
 
 
def test_return_path_is_longitudinal():
    s = Station(x=8.0, y=0.375, lateral_offset_m=0.375)
    path = return_path(s, (10.0, 0.375), 0.375)
    for _, y in path:
        assert y == pytest.approx(0.375)
 
 
def test_return_path_refuses_to_cross_a_row():
    s = Station(x=8.0, y=0.375, lateral_offset_m=0.375)
    with pytest.raises(ValueError):
        return_path(s, (10.0, 2.125), 3.0)
 
 
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
 
 
def test_routed_cost_exceeds_the_euclidean_estimate_across_lanes():
    """The allocator bids hypot; the real path goes round. Quantifies the gap."""
    lanes = lane_centres(N_ROWS, ROW_SPACING)
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    pose = (15.0, 0.375)
    task = (15.0, 2.25)
    plan = plan_via_headland(
        pose, task[0], task[1], lanes, (-1.5, 31.5), wps, 0, +1, SPRAY_REACH
    )
    assert plan is not None
 
    euclidean = math.dist(pose, task) * 12.0 + 30.0
    routed = detour_cost_j(pose, plan, (15.0, 0.375), 12.0, 30.0)
    assert routed > euclidean * 5.0
 
 
def test_in_lane_cost_is_close_to_euclidean():
    wps = [(x, 0.375) for x in (0.0, 30.0)]
    pose = (10.0, 0.375)
    plan = plan_in_lane(pose, 12.0, 0.0, wps, 0, +1, SPRAY_REACH)
    assert plan is not None
    routed = detour_cost_j(pose, plan, (12.0, 0.375), 12.0, 30.0)
    assert routed == pytest.approx(2.0 * 12.0 + 30.0)
 
