from __future__ import annotations

import csv
import os
import random
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, os.path.join(REPO, "experiments"))

import decision_overlap as do  # noqa: E402
import run_batch  # noqa: E402

CAP = 16000.0
E_PER_M = 12.0
TREAT_J = 30.0


def bid(robot, dist, energy=CAP, feasible=True, task=1, rnd=0, t=0.0,
        utility=None):
    cost = dist * E_PER_M + TREAT_J
    if utility is None:
        utility = -dist
    return do.Bid(t_s=t, task_id=task, round=rnd, robot_id=robot,
                  utility=utility, travel_cost_m=dist, energy_cost_j=cost,
                  remaining_energy_j=energy, feasible=feasible)



def test_distance_picks_the_nearest_feasible_robot():
    bids = [bid("robot_0", 5.0), bid("robot_1", 2.0, feasible=False),
            bid("robot_2", 3.0)]
    assert do.distance_winner(bids) == "robot_2"


def test_ties_go_to_the_lower_robot_id():
    bids = [bid("robot_3", 2.0), bid("robot_1", 2.0)]
    assert do.distance_winner(bids) == "robot_1"
    assert do.energy_winner(bids, CAP) == "robot_1"


def test_nothing_feasible_means_no_winner():
    bids = [bid("robot_0", 1.0, feasible=False)]
    assert do.distance_winner(bids) == ""
    assert do.energy_winner(bids, CAP) == ""


def test_energy_aware_can_prefer_a_farther_fuller_robot():
    # SoC ratio 0.70 / 0.43 = 1.63 beats cost ratio (5*12+31) / (3*12+31) = 1.36.
    bids = [bid("robot_0", 3.0, energy=0.43 * CAP),
            bid("robot_1", 5.0, energy=0.70 * CAP)]
    assert do.distance_winner(bids) == "robot_0"
    assert do.energy_winner(bids, CAP) == "robot_1"


def test_energy_aware_keeps_the_nearer_robot_when_the_gap_is_small():
    bids = [bid("robot_0", 3.0, energy=0.60 * CAP),
            bid("robot_1", 5.0, energy=0.70 * CAP)]
    assert do.energy_winner(bids, CAP) == "robot_0"


@pytest.mark.parametrize("trial", range(200))
def test_equal_charge_means_identical_rankings(trial):
    # The central claim: with every bidder at the same state of charge the
    # energy_aware ranking is a monotone transform of distance.
    rng = random.Random(trial)
    energy = rng.uniform(0.2, 1.0) * CAP
    bids = [bid(f"robot_{i}", rng.uniform(0.0, 40.0), energy=energy,
                feasible=rng.random() > 0.2)
            for i in range(rng.randint(1, 8))]
    assert do.energy_winner(bids, CAP) == do.distance_winner(bids)


def test_unknown_energy_counts_as_a_full_pack():
    assert do.state_of_charge(bid("r", 1.0, energy=do.UNKNOWN_ENERGY_J), CAP) == 1.0


def test_state_of_charge_is_clamped():
    assert do.state_of_charge(bid("r", 1.0, energy=2 * CAP), CAP) == 1.0
    assert do.state_of_charge(bid("r", 1.0, energy=0.0), CAP) == 0.0


def test_logged_winner_uses_the_logged_utility():
    bids = [bid("robot_0", 1.0, utility=0.001), bid("robot_1", 9.0, utility=0.002)]
    assert do.logged_winner(bids) == "robot_1"


def test_old_mode_name_is_accepted():
    assert do.normalise_mode("confidence_energy") == "energy_aware"
    with pytest.raises(ValueError):
        do.normalise_mode("greedy")


def _run():
    b = {
        # Contested, modes agree.
        (1, 0): [bid("robot_0", 2.0, task=1, t=10.0),
                 bid("robot_1", 4.0, task=1, t=10.0)],
        # Contested, modes disagree; energy pick is 2 m farther.
        (2, 0): [bid("robot_0", 3.0, energy=0.43 * CAP, task=2, t=20.0),
                 bid("robot_1", 5.0, energy=0.70 * CAP, task=2, t=20.0)],
        # Single feasible bidder: not contested.
        (3, 0): [bid("robot_0", 1.0, task=3, t=30.0),
                 bid("robot_1", 1.0, feasible=False, task=3, t=30.0)],
        # Nobody feasible: not an auction at all.
        (4, 1): [bid("robot_0", 1.0, feasible=False, task=4, rnd=1, t=40.0)],
    }
    awards = {(1, 0): "robot_0", (2, 0): "robot_0", (3, 0): "robot_0"}
    return do.analyse(b, awards, CAP, "distance", "r")


def test_analyse_counts_only_contested_auctions():
    r = _run()
    assert len(r.auctions) == 3
    assert len(r.contested) == 2
    assert r.n_disagree == 1
    assert r.disagree_rate == pytest.approx(0.5)


def test_extra_metres_are_energy_pick_minus_nearest():
    assert _run().extra_m == pytest.approx(2.0)


def test_validation_rates():
    r = _run()
    assert r.own_mode_match == pytest.approx(1.0)   # distance run, distance winners
    assert r.logged_match == pytest.approx(1.0)
    r.mode = "energy_aware"
    assert r.own_mode_match == pytest.approx(2 / 3)


def test_halves_split_by_time():
    early, late = _run().halves()
    assert (early, late) == (0.0, 1.0)


def test_empty_run_reports_nan_not_an_error():
    r = do.analyse({}, {}, CAP, "distance")
    assert r.disagree_rate != r.disagree_rate
    assert r.own_mode_match != r.own_mode_match



BID_HEADER = ["t_s", "task_id", "round", "robot_id", "utility",
              "travel_cost_m", "energy_cost_j", "remaining_energy_j", "feasible"]
EVENT_HEADER = ["t_s", "event", "observer_id", "task_id", "round", "x", "y",
                "confidence", "winner_a", "winner_b", "n_bids_a", "n_bids_b",
                "detail"]


def _write_run(d, bids, awards):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "bids.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BID_HEADER)
        for b in bids:
            w.writerow([b.t_s, b.task_id, b.round, b.robot_id, b.utility,
                        b.travel_cost_m, b.energy_cost_j,
                        b.remaining_energy_j, int(b.feasible)])
    with open(os.path.join(d, "events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(EVENT_HEADER)
        for (task, rnd), winner in awards.items():
            w.writerow([1.0, "award", winner, task, rnd, "", "", "", winner,
                        "", 2, "", "utility=0"])
        w.writerow([2.0, "reannounce", "robot_0", 9, 0, 1, 1, 0.9, "", "",
                    0, 0, "no_bids"])


def test_load_bids_keeps_one_bid_per_robot(tmp_path):
    d = str(tmp_path / "run")
    first = bid("robot_0", 2.0)
    repeat = bid("robot_0", 9.0)
    _write_run(d, [first, repeat, bid("robot_1", 3.0)], {})
    got = do.load_bids(os.path.join(d, "bids.csv"))
    assert len(got[(1, 0)]) == 2
    assert {b.travel_cost_m for b in got[(1, 0)]} == {2.0, 3.0}


def test_load_awards_ignores_other_events(tmp_path):
    d = str(tmp_path / "run")
    _write_run(d, [], {(1, 0): "robot_1"})
    assert do.load_awards(os.path.join(d, "events.csv")) == {(1, 0): "robot_1"}


def test_missing_bids_file_is_skipped(tmp_path):
    assert do.analyse_dir(str(tmp_path), CAP, "distance") is None


def test_cli_reads_results_and_skips_old_runs(tmp_path):
    new = str(tmp_path / "new")
    old = str(tmp_path / "old")
    os.makedirs(old)
    _write_run(new, [bid("robot_0", 3.0, energy=0.43 * CAP),
                     bid("robot_1", 5.0, energy=0.70 * CAP)],
               {(1, 0): "robot_0"})
    results = tmp_path / "results.csv"
    with open(results, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "status", "out_dir", "bid_mode",
                    "energy_capacity_j", "total_energy_j"])
        w.writerow(["new", "ok", new, "distance", CAP, 40000])
        w.writerow(["old", "ok", old, "confidence_energy", CAP, 40000])
        w.writerow(["bad", "wall_timeout", new, "distance", CAP, ""])
    out = tmp_path / "runs.csv"
    auctions = tmp_path / "auctions.csv"
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "analysis", "decision_overlap.py"),
         "--results", str(results), "--out", str(out),
         "--out-auctions", str(auctions)],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "1 run(s) have no bids.csv" in r.stdout
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [x["run"] for x in rows] == ["new"]
    assert rows[0]["disagree"] == "1"
    with open(auctions, newline="") as f:
        a = list(csv.DictReader(f))
    assert a[0]["by_distance"] == "robot_0" and a[0]["by_energy"] == "robot_1"


def test_cli_direct_dirs_need_mode_and_capacity(tmp_path):
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "analysis", "decision_overlap.py"),
         str(tmp_path)],
        capture_output=True, text=True, check=False)
    assert r.returncode != 0
    assert "--mode" in r.stderr



def test_set_aside_moves_a_populated_dir(tmp_path):
    d = tmp_path / "base_s0_n4_distance_r1"
    d.mkdir()
    (d / "run.log").write_text("hung")
    dest = run_batch.set_aside(str(d), "failed")
    assert dest == f"{d}.failed1"
    assert not d.exists()
    assert (tmp_path / "base_s0_n4_distance_r1.failed1" / "run.log").read_text() == "hung"


def test_set_aside_numbers_repeated_failures(tmp_path):
    d = tmp_path / "run"
    for expected in ("failed1", "failed2", "failed3"):
        d.mkdir()
        (d / "run.log").write_text(expected)
        assert run_batch.set_aside(str(d), "failed").endswith(expected)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "run.failed1", "run.failed2", "run.failed3"]


def test_set_aside_leaves_empty_or_missing_dirs_alone(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_batch.set_aside(str(empty), "prev") is None
    assert empty.exists()
    assert run_batch.set_aside(str(tmp_path / "absent"), "prev") is None