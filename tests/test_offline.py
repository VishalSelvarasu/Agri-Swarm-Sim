"""ROS-free unit tests. These are the only machine-verified parts of the repo.

Run:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "agri_swarm_core"))
sys.path.insert(0, os.path.join(REPO, "analysis"))

from agri_swarm_core.pure_pursuit import (  # noqa: E402
    Lane, Pose2D, PursuitLimits, assign_lanes, find_lookahead, is_finished,
    lane_clearance_m, lane_waypoints, pursuit_command,
)
import score_run  # noqa: E402
from score_run import Treatment, Weed, score_at_threshold, sweep  # noqa: E402


# ---------------------------------------------------------------------------
# Lane assignment must be a PARTITION. A dropped lane is a silent hole in
# coverage that shows up as a permanently low recall you will blame on the
# detector.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_lanes", range(0, 25))
@pytest.mark.parametrize("n_robots", [1, 2, 3, 4, 8, 12])
def test_lane_assignment_is_a_partition(n_lanes, n_robots):
    got = []
    for r in range(n_robots):
        got.extend(assign_lanes(n_lanes, n_robots, r))
    assert sorted(got) == list(range(n_lanes))


@pytest.mark.parametrize("n_robots", [1, 2, 3, 4, 8])
def test_lane_assignment_is_balanced_and_contiguous(n_robots):
    n_lanes = 11
    sizes = []
    for r in range(n_robots):
        block = assign_lanes(n_lanes, n_robots, r)
        sizes.append(len(block))
        if block:
            assert block == list(range(block[0], block[0] + len(block)))
    assert max(sizes) - min(sizes) <= 1


def test_lane_assignment_rejects_bad_input():
    with pytest.raises(ValueError):
        assign_lanes(10, 0, 0)
    with pytest.raises(ValueError):
        assign_lanes(10, 4, 4)


# ---------------------------------------------------------------------------
# The week-1 acceptance criterion, as a test: lane centrelines clear the crop.
# ---------------------------------------------------------------------------

def test_lane_clearance_is_positive_with_shipped_defaults():
    c = lane_clearance_m()
    assert c > 0.0, "robot is wider than the lane; no controller can fix this"
    # 9.5 cm per side. Recorded here so a change to row_spacing, crop_row_width,
    # body_w or wheel_sep that eats the margin fails loudly instead of showing
    # up as robots grinding through crop rows in RViz.
    assert c == pytest.approx(0.095, abs=1e-9)


def test_lane_clearance_goes_negative_for_a_wide_robot():
    assert lane_clearance_m(robot_half_width=0.40) < 0.0


def test_waypoints_stay_on_lane_centrelines():
    lanes = [Lane(i, -0.375 + i * 0.75, 0.0, 30.0) for i in range(11)]
    path = lane_waypoints(lanes, [0, 1, 2], step_m=1.0)
    lane_ys = {ln.y for ln in lanes}
    for _x, y in path:
        assert any(abs(y - ly) < 1e-9 for ly in lane_ys)


def test_serpentine_reverses_alternate_lanes():
    lanes = [Lane(0, 0.0, 0.0, 3.0), Lane(1, 0.75, 0.0, 3.0)]
    path = lane_waypoints(lanes, [0, 1], step_m=1.0, serpentine=True)
    first = [p for p in path if p[1] == 0.0]
    second = [p for p in path if p[1] == 0.75]
    assert first[0][0] < first[-1][0]
    assert second[0][0] > second[-1][0]
    # Transit between lanes is one spacing, not one field length.
    assert math.dist(first[-1], second[0]) == pytest.approx(0.75)


def test_lane_waypoints_rejects_unknown_lane():
    with pytest.raises(KeyError):
        lane_waypoints([Lane(0, 0.0, 0.0, 3.0)], [7])


# ---------------------------------------------------------------------------
# Pure pursuit
# ---------------------------------------------------------------------------

LIM = PursuitLimits()


def test_straight_ahead_drives_straight():
    cmd = pursuit_command(Pose2D(0.0, 0.0, 0.0), (5.0, 0.0), LIM)
    assert cmd.v > 0.0
    assert cmd.omega == pytest.approx(0.0, abs=1e-12)


def test_target_to_the_left_turns_left():
    cmd = pursuit_command(Pose2D(0.0, 0.0, 0.0), (2.0, 1.0), LIM)
    assert cmd.omega > 0.0


def test_target_to_the_right_turns_right():
    cmd = pursuit_command(Pose2D(0.0, 0.0, 0.0), (2.0, -1.0), LIM)
    assert cmd.omega < 0.0


def test_sign_convention_holds_when_the_robot_is_rotated():
    # Robot facing +y; target is to its left (i.e. towards -x).
    cmd = pursuit_command(Pose2D(0.0, 0.0, math.pi / 2), (-1.0, 2.0), LIM)
    assert cmd.omega > 0.0


def test_target_behind_turns_in_place():
    cmd = pursuit_command(Pose2D(0.0, 0.0, 0.0), (-3.0, 0.1), LIM)
    assert cmd.v == 0.0
    assert abs(cmd.omega) == pytest.approx(LIM.omega_max)


def test_omega_never_exceeds_limit_over_a_dense_sample():
    for th in [i * math.pi / 12 for i in range(24)]:
        for tx in [-3.0, -0.5, 0.2, 1.0, 5.0]:
            for ty in [-3.0, -0.4, 0.0, 0.4, 3.0]:
                cmd = pursuit_command(Pose2D(0.0, 0.0, th), (tx, ty), LIM)
                assert abs(cmd.omega) <= LIM.omega_max + 1e-9
                assert 0.0 <= cmd.v <= LIM.v_nom + 1e-9


def test_saturating_omega_slows_down_instead_of_straightening():
    tight = PursuitLimits(omega_max=0.2)
    cmd = pursuit_command(Pose2D(0.0, 0.0, 0.0), (0.4, 0.35), tight)
    assert abs(cmd.omega) == pytest.approx(tight.omega_max)
    assert cmd.v < tight.v_nom


def test_lookahead_index_is_monotone_on_a_serpentine_path():
    lanes = [Lane(0, 0.0, 0.0, 10.0), Lane(1, 0.75, 0.0, 10.0)]
    path = lane_waypoints(lanes, [0, 1], step_m=0.5)
    idx = 0
    for p in path:
        new_idx, _pt = find_lookahead(path, p, 0.7, start_index=idx)
        assert new_idx >= idx, "controller latched back onto a finished lane"
        idx = new_idx


def test_lookahead_returns_last_point_at_the_end():
    path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    i, pt = find_lookahead(path, (2.0, 0.0), 5.0)
    assert i == len(path) - 1 and pt == (2.0, 0.0)


def test_closed_loop_reaches_the_end_without_leaving_the_lane_corridor():
    """Integrate the controller. This is the milestone rehearsed offline."""
    lanes = [Lane(0, 0.0, 0.0, 12.0), Lane(1, 0.75, 0.0, 12.0)]
    path = lane_waypoints(lanes, [0, 1], step_m=0.5)
    pose = Pose2D(0.0, 0.0, 0.0)
    idx = 0
    dt = 0.05
    clearance = lane_clearance_m()
    max_excursion = 0.0

    for _ in range(20000):
        idx, target = find_lookahead(path, (pose.x, pose.y), LIM.lookahead_m, idx)
        if is_finished(pose, path, idx, LIM):
            break
        cmd = pursuit_command(pose, target, LIM)
        pose = Pose2D(
            pose.x + cmd.v * math.cos(pose.theta) * dt,
            pose.y + cmd.v * math.sin(pose.theta) * dt,
            pose.theta + cmd.omega * dt,
        )
        lateral = min(abs(pose.y - ln.y) for ln in lanes)
        max_excursion = max(max_excursion, lateral)
    else:
        pytest.fail("controller did not finish the path")

    # The lane-change manoeuvre itself is a legitimate excursion; what matters
    # is that it stays inside the half-spacing corridor rather than wandering
    # into the neighbouring crop row.
    assert max_excursion < 0.75, (
        f"excursion {max_excursion:.3f} m exceeds row spacing; "
        f"clearance budget is only {clearance:.3f} m per side"
    )


# ---------------------------------------------------------------------------
# Offline scorer
# ---------------------------------------------------------------------------

def _weeds(n=5):
    return [Weed(i, float(i) * 2.0, 0.0, 0.10, 1 if i % 2 == 0 else 0) for i in range(n)]


def _treat(x, y, conf, tid=0):
    return Treatment(t_s=1.0, robot_id=0, task_id=tid, x=x, y=y, confidence=conf)


def test_perfect_log_scores_perfect_recall_and_no_false_positives():
    ws = _weeds()
    ts = [_treat(w.x, w.y, 0.9, w.id) for w in ws]
    s = score_at_threshold(ws, ts, 0.2, 0.25)
    assert s.weeds_treated == len(ws)
    assert s.weeds_treated_frac == pytest.approx(1.0)
    assert s.false_positive_treatments == 0
    assert s.precision == pytest.approx(1.0)


def test_treatments_on_bare_soil_are_false_positives():
    ws = _weeds()
    ts = [_treat(100.0 + i, 100.0, 0.9, i) for i in range(3)]
    s = score_at_threshold(ws, ts, 0.2, 0.25)
    assert s.weeds_treated == 0
    assert s.false_positive_treatments == 3
    assert s.precision == pytest.approx(0.0)


def test_treating_the_same_weed_twice_is_redundant_not_a_false_positive():
    ws = _weeds(1)
    ts = [_treat(0.0, 0.0, 0.9), _treat(0.02, 0.01, 0.9)]
    s = score_at_threshold(ws, ts, 0.2, 0.25)
    assert s.weeds_treated == 1
    assert s.redundant_treatments == 1
    assert s.false_positive_treatments == 0
    assert s.treatments_applied == 2  # herbicide was still dispensed


def test_raising_the_threshold_is_monotone_in_both_directions():
    ws = _weeds(6)
    ts = [_treat(w.x, w.y, 0.2 + 0.1 * w.id, w.id) for w in ws]
    ts += [_treat(50.0, 50.0, 0.35), _treat(60.0, 60.0, 0.75)]
    scores = sweep(ws, ts, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 0.25)

    recalls = [s.weeds_treated_frac for s in scores]
    fps = [s.false_positive_treatments for s in scores]
    applied = [s.treatments_applied for s in scores]

    assert recalls == sorted(recalls, reverse=True), "recall must not rise with threshold"
    assert fps == sorted(fps, reverse=True), "FPs must not rise with threshold"
    assert applied == sorted(applied, reverse=True)
    # The trade actually bites somewhere in the sweep, otherwise the curve is flat
    # and the threshold is not doing anything.
    assert recalls[0] > recalls[-1]


def test_intra_row_fraction_is_tracked_separately():
    ws = _weeds(6)  # ids 0,2,4 are intra-row
    ts = [_treat(w.x, w.y, 0.9, w.id) for w in ws if w.in_row == 1]
    s = score_at_threshold(ws, ts, 0.2, 0.25)
    assert s.intra_row_treated_frac == pytest.approx(1.0)
    assert s.weeds_treated_frac == pytest.approx(0.5)


def test_scorer_cli_end_to_end(tmp_path):
    gt = tmp_path / "ground_truth_0.csv"
    with open(gt, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "radius", "in_row"])
        for i in range(4):
            w.writerow([i, i * 2.0, 0.0, 0.1, i % 2])

    tr = tmp_path / "treatments.csv"
    with open(tr, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "robot_id", "task_id", "x", "y", "confidence"])
        for i in range(4):
            w.writerow([i * 3.0, i % 2, i, i * 2.0, 0.01, 0.25 + 0.15 * i])

    out = tmp_path / "curve.csv"
    rc = score_run.main([
        "--ground-truth", str(gt), "--treatments", str(tr),
        "--thresholds", "0.2", "0.5", "0.8", "--out", str(out),
    ])
    assert rc == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3
    assert int(rows[0]["weeds_treated"]) >= int(rows[-1]["weeds_treated"])


# ---------------------------------------------------------------------------
# The determinism claim in the README, made executable.
# ---------------------------------------------------------------------------

GEN = os.path.join(REPO, "src", "agri_swarm_core", "agri_swarm_core", "generate_field.py")


def _generate(seed, out):
    subprocess.run([sys.executable, GEN, "--seed", str(seed), "--out", out],
                   check=True, capture_output=True)


def test_same_seed_gives_byte_identical_output(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _generate(7, str(a))
    _generate(7, str(b))
    for name in ["agri_field_7.sdf", "ground_truth_7.csv", "lanes_7.csv"]:
        assert open(a / name, "rb").read() == open(b / name, "rb").read(), name


def test_different_seeds_differ(tmp_path):
    _generate(1, str(tmp_path))
    _generate(2, str(tmp_path))
    assert (open(tmp_path / "ground_truth_1.csv").read()
            != open(tmp_path / "ground_truth_2.csv").read())


def test_lanes_never_coincide_with_crop_rows(tmp_path):
    """Structural, not cosmetic: a lane on top of a row is an unwinnable field."""
    _generate(3, str(tmp_path))
    lanes = list(csv.DictReader(open(tmp_path / "lanes_3.csv")))
    row_spacing = 0.75
    row_ys = [i * row_spacing for i in range(10)]
    for ln in lanes:
        y = float(ln["y"])
        assert min(abs(y - ry) for ry in row_ys) > lane_clearance_m()


def test_ground_truth_is_never_read_by_the_allocator():
    """Guard rail. If this fails the experiment is invalid, not merely wrong."""
    alloc_dir = os.path.join(REPO, "src", "agri_swarm_allocation")
    offenders = []
    for root, _dirs, files in os.walk(alloc_dir):
        for fn in files:
            if not fn.endswith((".cpp", ".hpp", ".h", ".py")):
                continue
            text = open(os.path.join(root, fn), errors="ignore").read()
            if "ground_truth" in text or "patch_id" in text:
                offenders.append(os.path.join(root, fn))
    assert not offenders, f"allocator touches oracle data: {offenders}"


def test_ablation_does_not_leak_outside_utility_hpp():
    """bid_mode may be declared and passed around; only utility.hpp may BRANCH on it.

    Declaring the parameter default, or storing the parsed enum, is fine. What
    is not fine is a second place in the codebase that behaves differently
    depending on the mode -- at that point the two arms of the experiment
    differ by more than one function and the comparison is void.
    """
    alloc_dir = os.path.join(REPO, "src", "agri_swarm_allocation")
    allowed = {"utility.hpp", "test_utility.cpp"}
    branch_markers = (
        'bid_mode_ ==', 'bid_mode ==',
        '== "confidence_energy"', '== "distance"',
        'case BidMode::', 'case agri_swarm::BidMode::',
        'switch (bid_mode', 'switch(bid_mode',
    )
    offenders = []
    for root, _dirs, files in os.walk(alloc_dir):
        for fn in files:
            if not fn.endswith((".cpp", ".hpp", ".h")) or fn in allowed:
                continue
            text = open(os.path.join(root, fn), errors="ignore").read()
            hits = [m for m in branch_markers if m in text]
            if hits:
                offenders.append((os.path.join(root, fn), hits))
    assert not offenders, (
        "the ablation has leaked out of utility.hpp; the comparison is no "
        f"longer clean: {offenders}"
    )
