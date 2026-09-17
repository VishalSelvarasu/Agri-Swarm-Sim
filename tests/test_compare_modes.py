from __future__ import annotations
 
import csv
import os
import subprocess
import sys
 
import pytest
 
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "analysis"))
 
import compare_modes as cm  # noqa: E402
 
SLACK = os.path.join(REPO, "results", "slack_22kj.csv")
BINDING = os.path.join(REPO, "results", "binding_16kj.csv")
 
 
# ---------------------------------------------------------------------------
# Student's t
# ---------------------------------------------------------------------------
 
@pytest.mark.parametrize("p,df,expected", [
    (0.975, 1, 12.706205),
    (0.975, 3, 3.182446),
    (0.975, 9, 2.262157),
    (0.975, 19, 2.093024),
    (0.95, 9, 1.833113),
    (0.80, 9, 0.883404),
    (0.80, 19, 0.860951),
    (0.975, 1000, 1.962339),
])
def test_t_ppf_matches_scipy(p, df, expected):
    assert cm.t_ppf(p, df) == pytest.approx(expected, abs=2e-6)
 
 
@pytest.mark.parametrize("df", [1, 2, 5, 9, 19, 50])
@pytest.mark.parametrize("p", [0.6, 0.8, 0.9, 0.95, 0.975, 0.995])
def test_t_cdf_inverts_t_ppf(p, df):
    assert cm.t_cdf(cm.t_ppf(p, df), df) == pytest.approx(p, abs=1e-9)
 
 
def test_t_ppf_is_antisymmetric():
    assert cm.t_ppf(0.025, 9) == pytest.approx(-cm.t_ppf(0.975, 9), abs=1e-12)
    assert cm.t_ppf(0.5, 9) == 0.0
 
 
def test_t_ppf_rejects_out_of_range():
    for p in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            cm.t_ppf(p, 5)
 
 
# ---------------------------------------------------------------------------
# Fisher exact
# ---------------------------------------------------------------------------
 
@pytest.mark.parametrize("table,expected", [
    ((4, 40, 2, 40), 0.6766149565582592),
    ((2, 20, 3, 20), 1.0),
    ((0, 10, 6, 4), 0.010835913312693497),
    ((7, 1, 2, 9), 0.005477494641581329),
    ((1, 30, 9, 22), 0.012450169784506008),
])
def test_fisher_exact_matches_scipy(table, expected):
    assert cm.fisher_exact(*table) == pytest.approx(expected, rel=1e-9)
 
 
# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
 
def test_summarise_on_a_known_vector():
    # Differences 2, -1, 4, 1.
    rows = [(0, 10.0, 12.0), (1, 10.0, 9.0), (2, 10.0, 14.0), (3, 10.0, 11.0)]
    s = cm.summarise(rows, better_is_lower=True)
    assert s.n == 4
    assert s.diff == pytest.approx(1.5)
    assert s.sd == pytest.approx(2.0816659994661326)
    assert s.lo == pytest.approx(-1.812395134417855, abs=1e-5)
    assert s.hi == pytest.approx(4.812395134417855, abs=1e-5)
    assert s.p_value == pytest.approx(0.24519388179494778, abs=1e-6)
    assert s.mde == pytest.approx(4.330821406450704, abs=1e-5)
    assert s.equiv_bound == pytest.approx(3.949458323306894, abs=1e-5)
    assert s.wins == 1          # only the -1 favours the treatment
 
 
def test_summarise_needs_two_seeds():
    assert cm.summarise([(0, 1.0, 2.0)], better_is_lower=True) is None
 
 
def test_summarise_survives_zero_variance():
    s = cm.summarise([(0, 1.0, 2.0), (1, 1.0, 2.0)], better_is_lower=True)
    assert s.sd == 0.0
    assert s.p_value == 0.0
 
 
def test_wins_follow_the_metric_direction():
    rows = [(0, 1.0, 2.0), (1, 1.0, 0.5), (2, 1.0, 3.0)]
    assert cm.summarise(rows, better_is_lower=True).wins == 1
    assert cm.summarise(rows, better_is_lower=False).wins == 2
 
 
# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
 
def _write(path, rows):
    fields = ["run_id", "seed", "bid_mode", "rep", "status",
              "mission_time_s", "total_energy_j"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
 
 
def test_load_aliases_the_old_mode_name_and_keeps_failures(tmp_path):
    p = tmp_path / "r.csv"
    _write(p, [
        dict(run_id="a", seed=0, bid_mode="distance", rep=1, status="ok",
             mission_time_s=100, total_energy_j=10),
        dict(run_id="b", seed=0, bid_mode="confidence_energy", rep=1,
             status="ok", mission_time_s=100, total_energy_j=12),
        dict(run_id="c", seed=0, bid_mode="confidence_energy", rep=2,
             status="wall_timeout", mission_time_s="", total_energy_j=""),
    ])
    rows = cm.load(str(p))
    assert {r["bid_mode"] for r in rows} == {"distance", "energy_aware"}
    assert cm.status_counts(rows) == {"distance": (1, 0), "energy_aware": (1, 1)}
    cells = cm.ok_cells(rows)
    assert len(cells[(0, "energy_aware")]) == 1
    assert cm.paired(cells, "total_energy_j") == [(0, 10.0, 12.0)]
 
 
def test_exclude_seed_drops_every_arm(tmp_path):
    p = tmp_path / "r.csv"
    _write(p, [
        dict(run_id=f"{s}{m}", seed=s, bid_mode=m, rep=1, status="ok",
             mission_time_s=100, total_energy_j=10 + s)
        for s in (0, 1, 2) for m in ("distance", "energy_aware")
    ])
    rows = cm.load(str(p), exclude_seeds=[1])
    assert sorted({int(r["seed"]) for r in rows}) == [0, 2]
 
 
def test_long_runs_are_flagged(tmp_path):
    p = tmp_path / "r.csv"
    _write(p, [
        dict(run_id=f"r{i}", seed=i, bid_mode="distance", rep=1, status="ok",
             mission_time_s=t, total_energy_j=1)
        for i, t in enumerate([100, 110, 90, 105, 400])
    ])
    _med, slow = cm.long_runs(cm.load(str(p)))
    assert [r["run_id"] for r in slow] == ["r4"]
 
 
# ---------------------------------------------------------------------------
# Committed results. If one of these fails, the published numbers changed.
# ---------------------------------------------------------------------------
 
def _summary(path, key, lower=True, exclude=()):
    cells = cm.ok_cells(cm.load(path, exclude))
    return cm.summarise(cm.paired(cells, key), better_is_lower=lower)
 
 
def test_binding_energy_is_pinned():
    s = _summary(BINDING, "total_energy_j")
    assert s.n == 10
    assert s.diff == pytest.approx(1272.95, abs=1e-6)
    assert s.lo == pytest.approx(-1049.4633623186276, abs=0.01)
    assert s.hi == pytest.approx(3595.363362318628, abs=0.01)
    assert s.mde == pytest.approx(3229.34810664017, abs=0.01)
    assert s.p_value == pytest.approx(0.24635783750065787, abs=1e-6)
 
 
def test_slack_energy_is_pinned():
    s = _summary(SLACK, "total_energy_j")
    assert s.n == 20
    assert s.diff == pytest.approx(-798.7, abs=0.05)
    assert s.lo == pytest.approx(-2556, abs=1)
    assert s.hi == pytest.approx(959, abs=1)
    assert s.mde == pytest.approx(2480, abs=1)
 
 
def test_slack_mission_time_sign_depends_on_seed_15():
    # One near-hang run (seed 15) carries the whole positive mission-time
    # difference in the slack condition.
    assert _summary(SLACK, "mission_time_s").diff > 0
    assert _summary(SLACK, "mission_time_s", exclude=[15]).diff < 0
 
 
def test_timeouts_do_not_depend_on_mode():
    for path in (SLACK, BINDING):
        sc = cm.status_counts(cm.load(path))
        b_ok, b_bad = sc["distance"]
        t_ok, t_bad = sc["energy_aware"]
        assert cm.fisher_exact(b_bad, b_ok, t_bad, t_ok) > 0.5
 
 
def test_cli_writes_one_row_per_metric(tmp_path):
    out = tmp_path / "table.csv"
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "analysis", "compare_modes.py"),
         BINDING, "--per-seed", "--out", str(out)],
        capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    with open(out, newline="") as f:
        metrics = [row["metric"] for row in csv.DictReader(f)]
    assert metrics == [m[0] for m in cm.METRICS]
