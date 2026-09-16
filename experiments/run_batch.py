#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import yaml

FIELDS = [
    "run_id", "config", "seed", "n_robots", "bid_mode", "treat_threshold",
    "rep", "energy_capacity_j",
    "status", "returncode", "wall_s", "timeout_s",
    "mission_time_s", "sweeps_complete",
    "total_energy_j", "max_energy_j", "min_energy_j", "energy_spread_j",
    "total_distance_m", "treatments",
    "weeds_treated_frac", "intra_row_treated_frac", "precision",
    "redundant_treatments", "duplicate_task_treatments",
    "split_brain_awards", "reannounce_events", "abandon_events",
    "git_sha", "git_dirty", "out_dir", "command",
]

DEFAULT_COMMAND = (
    "ros2 launch agri_swarm_bringup swarm.launch.py "
    "n_robots:={n_robots} seed:={seed} bid_mode:={bid_mode} "
    "treat_confidence_threshold:={treat_threshold} "
    "energy_capacity_j:={energy_capacity_j} "
    "world_dir:={world_dir} "
    "treatments_csv:={out_dir}/treatments.csv "
    "max_mission_s:={max_mission_s} "
    "headless:=true use_allocator:=true"
)

# A run at a threshold below this treats detector noise: at 0.25 recall was
# 0.101. The executed threshold must also equal min(--thresholds) in
# score_run.py or the recovered curve's low end is fabricated.
MIN_THRESHOLD = 0.5
# Wall-clock floor for the per-run kill. A complete mission takes ~700 s.
MIN_TIMEOUT_S = 1200.0

SCORED = ("weeds_treated_frac", "intra_row_treated_frac", "precision",
          "redundant_treatments", "duplicate_task_treatments")
CONTENDED = ("split_brain_awards", "reannounce_events", "abandon_events")

# "robot_3: 415.8 m travelled, 30 treatments, 5889 J spent of 100000 J"
ENERGY_RE = re.compile(
    r"(robot_\d+): ([\d.]+) m travelled, (\d+) treatments, ([\d.]+) J spent")
COMPLETE_RE = re.compile(r"mission complete at (\d+)s")
TIMEOUT_RE = re.compile(r"mission timeout at (\d+)s")


def git_meta(repo: str) -> Dict[str, str]:
    def run(args: Sequence[str]) -> str:
        try:
            return subprocess.run(args, cwd=repo, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"
    dirty = run(["git", "status", "--porcelain"])
    return {"git_sha": run(["git", "rev-parse", "--short", "HEAD"]),
            "git_dirty": "1" if dirty and dirty != "unknown" else "0"}


# launch signals the ruby wrapper, not the server it spawned, and does not
# reliably reap its own nodes either. Six generations of orphans accumulated in
# one session: four energy monitors and four executors per robot, plus multiple
# ros_gz bridges forwarding the same odometry onto the same topic. It presented
# as bad experimental results, never as an error.
STRAY = "agri_swarm|gz sim|robot_state_publisher|parameter_bridge|ros_gz"


def strays_running() -> bool:
    return subprocess.run(["pgrep", "-f", STRAY],
                          capture_output=True).returncode == 0


def reap_gazebo(grace_s: float = 15.0) -> bool:
    """Kill every process from a previous run and wait until they are gone."""
    if not strays_running():
        return True
    subprocess.run(["pkill", "-f", STRAY], check=False)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not strays_running():
            return True
        time.sleep(0.5)
    subprocess.run(["pkill", "-9", "-f", STRAY], check=False)
    time.sleep(3.0)
    return not strays_running()


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def ensure_field(world_dir: str, seed: int, repo: str) -> None:
    needed = [f"agri_field_{seed}.sdf", f"lanes_{seed}.csv",
              f"ground_truth_{seed}.csv"]
    if all(os.path.isfile(os.path.join(world_dir, n)) for n in needed):
        return
    print(f"    generating field for seed {seed}", flush=True)
    subprocess.run([sys.executable, "-m", "agri_swarm_core.generate_field",
                    "--seed", str(seed), "--out", world_dir],
                   cwd=repo, check=True)


def expand_plan(cfg: dict, args) -> List[dict]:
    seeds = args.seeds or list(range(int(cfg["experiment"]["n_seeds"])))
    robots = args.n_robots or [int(cfg["swarm"]["n_robots"])]
    modes = args.bid_modes or [cfg["auction"]["bid_mode"]]
    thresholds = sorted(
        cfg.get("sweeps", {}).get("treat_confidence_threshold", [0.5]))
    thr = (args.treat_threshold if args.treat_threshold is not None
           else thresholds[0])

    plan = []
    for seed, n, mode, rep in itertools.product(
            seeds, robots, modes, range(1, args.repeats + 1)):
        plan.append({
            "run_id": f"{args.name}_s{seed}_n{n}_{mode}_r{rep}",
            "config": args.name, "seed": seed, "n_robots": n,
            "bid_mode": mode, "treat_threshold": thr, "rep": rep,
            "energy_capacity_j": args.energy_capacity,
        })
    return plan


def build_command(tpl: str, spec: dict, out_dir: str, world_dir: str,
                  max_mission_s: float) -> List[str]:
    return shlex.split(tpl.format(out_dir=out_dir, world_dir=world_dir,
                                  max_mission_s=max_mission_s, **spec))


def inspect_log(path: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        "mission_time_s": "", "sweeps_complete": 0, "total_energy_j": "",
        "max_energy_j": "", "min_energy_j": "", "energy_spread_j": "",
        "total_distance_m": "", "treatments": "", "log_status": "error",
    }
    if not os.path.isfile(path):
        return out

    per_robot: Dict[str, Dict[str, float]] = {}
    sweeps = 0
    with open(path, errors="replace") as f:
        for line in f:
            if "lane sweep complete" in line:
                sweeps += 1
            m = ENERGY_RE.search(line)
            if m:
                # Last line per robot wins: that is the shutdown summary.
                per_robot[m.group(1)] = {"d": float(m.group(2)),
                                         "t": int(m.group(3)),
                                         "e": float(m.group(4))}
            m = COMPLETE_RE.search(line)
            if m:
                out["mission_time_s"] = int(m.group(1))
                out["log_status"] = "ok"
            m = TIMEOUT_RE.search(line)
            if m:
                out["mission_time_s"] = int(m.group(1))
                out["log_status"] = "sim_timeout"

    out["sweeps_complete"] = sweeps
    if per_robot:
        e = [r["e"] for r in per_robot.values()]
        out["total_energy_j"] = round(sum(e), 1)
        out["max_energy_j"] = round(max(e), 1)
        out["min_energy_j"] = round(min(e), 1)
        out["energy_spread_j"] = round(max(e) - min(e), 1)
        out["total_distance_m"] = round(
            sum(r["d"] for r in per_robot.values()), 1)
        out["treatments"] = sum(r["t"] for r in per_robot.values())
    return out


def score_run(repo: str, world_dir: str, seed: int, out_dir: str,
              threshold: float) -> Dict[str, object]:
    """Score in-line, so a broken scorer surfaces on run 1 not run 120."""
    row = {k: "" for k in SCORED + CONTENDED}
    treatments = os.path.join(out_dir, "treatments.csv")
    if not os.path.isfile(treatments):
        return row

    curve = os.path.join(out_dir, "curve.csv")
    contention = os.path.join(out_dir, "contention.csv")
    with open(os.path.join(out_dir, "score.log"), "w") as log:
        subprocess.run(
            [sys.executable, "analysis/score_run.py",
             "--ground-truth",
             os.path.join(world_dir, f"ground_truth_{seed}.csv"),
             "--treatments", treatments,
             "--thresholds", str(threshold),
             "--out", curve, "--out-contention", contention],
            cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=False)

    for path, keys in ((curve, SCORED), (contention, CONTENDED)):
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            for k in keys:
                row[k] = rows[0].get(k, "")
    return row


def run_one(cmd: Sequence[str], timeout_s: float,
            log_path: str) -> Dict[str, object]:
    """Run one launch, killing it gracefully if it overruns.

    SIGINT first, not SIGKILL. launch turns SIGINT into an orderly shutdown
    and every node prints its summary — including which robots never reached
    idle, which is the only record of why a run hung. A SIGKILL discards all
    of it and leaves the energy columns blank.
    """
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        except FileNotFoundError as e:
            log.write(f"launcher not found: {e}\n")
            return {"killed": False, "returncode": -2, "wall_s": 0.0}

        killed = False
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            killed = True
            proc.send_signal(signal.SIGINT)
            try:
                rc = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
    return {"killed": killed, "returncode": rc,
            "wall_s": round(time.monotonic() - t0, 2)}


def completed(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        return {r["run_id"] for r in csv.DictReader(f)
                if r.get("status") == "ok"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/configs/base.yaml")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=os.path.expanduser("~/agri-runs/batch"))
    p.add_argument("--world-dir", dest="world_dir",
                   default=os.path.expanduser("~/agri-worlds"))
    p.add_argument("--results", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--n-robots", type=int, nargs="+", dest="n_robots",
                   default=None)
    p.add_argument("--bid-modes", nargs="+", dest="bid_modes", default=None,
                   choices=["distance", "energy_aware", "confidence_energy"])
    p.add_argument("--treat-threshold", type=float, default=None,
                   dest="treat_threshold")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--energy-capacity", type=float, default=16000.0,
                   dest="energy_capacity")
    p.add_argument("--max-mission-s", type=float, default=20000.0,
                   dest="max_mission_s")
    p.add_argument("--timeout", type=float, default=None,
                   help="Wall-clock kill per run, seconds.")
    p.add_argument("--min-free-gb", type=float, default=5.0,
                   dest="min_free_gb")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--command", default=DEFAULT_COMMAND)
    a = p.parse_args(argv)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    a.name = a.name or cfg["experiment"]["name"]
    timeout_s = (a.timeout if a.timeout is not None
                 else float(cfg["experiment"]["timeout_s"]))

    # A run takes ~11 min of wall clock. A shorter kill truncates every
    # mission and records it as wall_timeout.
    if timeout_s < MIN_TIMEOUT_S:
        print(f"ERROR: --timeout {timeout_s:.0f}s is below {MIN_TIMEOUT_S:.0f}s. "
              f"A full mission needs ~700s; anything tighter kills every run.",
              file=sys.stderr)
        return 2

    if a.out.startswith("/tmp"):
        print("ERROR: --out is under /tmp, which is tmpfs and clears on "
              "reboot.", file=sys.stderr)
        return 2

    # Fail once, loudly, rather than forty times in half a second each. A
    # fresh shell that has not sourced the overlay resolves the launch file
    # against ~/ros2_ws and every run dies before Gazebo starts.
    try:
        found = subprocess.run(["ros2", "pkg", "prefix", "agri_swarm_bringup"],
                               capture_output=True).returncode == 0
    except FileNotFoundError:
        found = False
    if not found:
        print("ERROR: agri_swarm_bringup is not on AMENT_PREFIX_PATH. Run "
              "`source /opt/ros/jazzy/setup.bash && source install/setup.bash` "
              "from the workspace root, in this shell, before starting the "
              "batch.", file=sys.stderr)
        return 2

    plan = expand_plan(cfg, a)
    if plan and float(plan[0]["treat_threshold"]) < MIN_THRESHOLD:
        print(f"ERROR: executed threshold {plan[0]['treat_threshold']} is below "
              f"{MIN_THRESHOLD}. The swarm would treat detector noise. Pass "
              f"--treat-threshold {MIN_THRESHOLD}, or fix the first entry of "
              f"sweeps.treat_confidence_threshold in {a.config}.",
              file=sys.stderr)
        return 2
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta = git_meta(repo)

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.world_dir, exist_ok=True)
    results_path = a.results or os.path.join(a.out, "results.csv")
    done = set() if a.force else completed(results_path)
    todo = [s for s in plan if s["run_id"] not in done]

    print(f"{len(plan)} runs planned, {len(done)} already ok, {len(todo)} to "
          f"run (~{len(todo) * 420 / 3600.0:.1f} h at ~7 min each)")
    if meta["git_dirty"] == "1":
        print("WARNING: working tree is dirty; these results will not be "
              "reproducible from the recorded SHA.", file=sys.stderr)

    if a.dry_run:
        for spec in todo:
            print(" ", " ".join(build_command(
                a.command, spec, os.path.join(a.out, spec["run_id"]),
                a.world_dir, a.max_mission_s)))
        return 0

    if free_gb(a.out) < a.min_free_gb:
        print(f"ERROR: {free_gb(a.out):.1f} GB free at {a.out}, need "
              f"{a.min_free_gb:.1f}. Clear ~/.cache and ~/.ros/log.",
              file=sys.stderr)
        return 2

    fresh = not os.path.isfile(results_path)
    n_bad = 0

    with open(results_path, "a", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
            rf.flush()

        for i, spec in enumerate(todo, 1):
            if free_gb(a.out) < a.min_free_gb:
                print(f"ABORT: free space below {a.min_free_gb} GB at run {i}. "
                      f"Re-run this command to resume.", file=sys.stderr)
                break

            out_dir = os.path.join(a.out, spec["run_id"])
            os.makedirs(out_dir, exist_ok=True)
            ensure_field(a.world_dir, spec["seed"], repo)

            # Before, not only after: a server orphaned by an earlier crash
            # would otherwise poison this run.
            if not reap_gazebo():
                print("ABORT: a gz server survived SIGKILL. Every later run "
                      "would attach to its world.", file=sys.stderr)
                return 3

            cmd = build_command(a.command, spec, out_dir, a.world_dir,
                                a.max_mission_s)
            print(f"[{i}/{len(todo)}] {spec['run_id']}", flush=True)
            proc = run_one(cmd, timeout_s, os.path.join(out_dir, "run.log"))
            reap_gazebo()

            parsed = inspect_log(os.path.join(out_dir, "run.log"))
            status = parsed.pop("log_status")
            if proc["killed"]:
                status = "wall_timeout"
            scored = score_run(repo, a.world_dir, spec["seed"], out_dir,
                               spec["treat_threshold"])

            writer.writerow({**spec, **parsed, **scored, **meta,
                             "status": status,
                             "returncode": proc["returncode"],
                             "wall_s": proc["wall_s"],
                             "timeout_s": timeout_s,
                             "out_dir": out_dir,
                             "command": " ".join(cmd)})
            rf.flush()
            os.fsync(rf.fileno())

            if status != "ok":
                n_bad += 1
            print(f"    {status} in {proc['wall_s']}s, "
                  f"{parsed['sweeps_complete']}/{spec['n_robots']} sweeps, "
                  f"energy {parsed['total_energy_j']} J, "
                  f"recall {scored['weeds_treated_frac']}", flush=True)

    print(f"\nwrote {results_path}")
    if n_bad:
        print(f"{n_bad} runs did not reach 'mission complete'.",
              file=sys.stderr)
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())