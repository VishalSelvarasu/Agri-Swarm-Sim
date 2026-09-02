#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

FIELDS = [
    "run_id", "config", "seed", "n_robots", "bid_mode", "treat_threshold",
    "status", "returncode", "wall_s", "timeout_s", "git_sha", "git_dirty",
    "out_dir", "command",
]


def git_meta(repo: str) -> Dict[str, str]:
    def run(args: Sequence[str]) -> str:
        try:
            return subprocess.run(args, cwd=repo, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    sha = run(["git", "rev-parse", "--short", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    return {"git_sha": sha, "git_dirty": "1" if dirty and dirty != "unknown" else "0"}


def expand_plan(cfg: dict, args) -> List[dict]:
    """Cartesian product of the axes that actually need simulating."""
    seeds = args.seeds if args.seeds else list(range(int(cfg["experiment"]["n_seeds"])))
    robots = args.n_robots if args.n_robots else [int(cfg["swarm"]["n_robots"])]
    modes = args.bid_modes if args.bid_modes else [cfg["auction"]["bid_mode"]]

    # Deliberately NOT a sweep axis. Run at the lowest threshold once and
    # recover the curve offline; see the module docstring.
    thresholds = sorted(cfg.get("sweeps", {}).get("treat_confidence_threshold", [0.2]))
    run_threshold = args.treat_threshold if args.treat_threshold is not None else thresholds[0]

    plan = []
    for i, (seed, n, mode) in enumerate(itertools.product(seeds, robots, modes)):
        plan.append({
            "run_id": f"{args.name}_s{seed}_n{n}_{mode}",
            "config": args.name,
            "seed": seed,
            "n_robots": n,
            "bid_mode": mode,
            "treat_threshold": run_threshold,
        })
    return plan


def build_command(tpl: str, spec: dict, out_dir: str) -> List[str]:
    return shlex.split(tpl.format(out_dir=out_dir, **spec))


def run_one(cmd: Sequence[str], timeout_s: float, log_path: str) -> Dict[str, object]:
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout_s)
            status = "ok" if proc.returncode == 0 else "error"
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            status, rc = "timeout", -1
        except FileNotFoundError as e:
            log.write(f"launcher not found: {e}\n")
            status, rc = "error", -2
    return {"status": status, "returncode": rc, "wall_s": round(time.monotonic() - t0, 2)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="experiments/configs/base.yaml")
    p.add_argument("--name", default=None, help="Overrides experiment.name.")
    p.add_argument("--out", default="runs", help="Root directory for run outputs.")
    p.add_argument("--results", default=None, help="Results CSV (default: <out>/results.csv).")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--n-robots", type=int, nargs="+", dest="n_robots", default=None)
    p.add_argument("--bid-modes", nargs="+", dest="bid_modes", default=None,
                   choices=["distance", "confidence_energy"])
    p.add_argument("--treat-threshold", type=float, default=None, dest="treat_threshold")
    p.add_argument("--timeout", type=float, default=None, help="Overrides experiment.timeout_s.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--command",
        default=(
            "ros2 launch agri_swarm_bringup swarm.launch.py "
            "n_robots:={n_robots} seed:={seed} bid_mode:={bid_mode} "
            "treat_confidence_threshold:={treat_threshold} "
            "world_dir:={out_dir} log_dir:={out_dir} headless:=true"
        ),
        help="Command template. {seed} {n_robots} {bid_mode} {treat_threshold} {out_dir}.",
    )
    a = p.parse_args(argv)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    a.name = a.name or cfg["experiment"]["name"]
    timeout_s = a.timeout if a.timeout is not None else float(cfg["experiment"]["timeout_s"])

    plan = expand_plan(cfg, a)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta = git_meta(repo)

    est_h = len(plan) * timeout_s / 3600.0
    print(f"{len(plan)} runs, timeout {timeout_s:.0f}s each "
          f"(worst case {est_h:.1f} h wall clock)")
    if meta["git_dirty"] == "1":
        print("WARNING: working tree is dirty; these results will not be reproducible "
              "from the recorded SHA.", file=sys.stderr)

    if a.dry_run:
        for spec in plan:
            out_dir = os.path.join(a.out, spec["run_id"])
            print(" ", " ".join(build_command(a.command, spec, out_dir)))
        return 0

    os.makedirs(a.out, exist_ok=True)
    results_path = a.results or os.path.join(a.out, "results.csv")
    fresh = not os.path.exists(results_path)

    with open(results_path, "a", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
            rf.flush()

        for i, spec in enumerate(plan, 1):
            out_dir = os.path.join(a.out, spec["run_id"])
            os.makedirs(out_dir, exist_ok=True)
            cmd = build_command(a.command, spec, out_dir)

            print(f"[{i}/{len(plan)}] {spec['run_id']}", flush=True)
            result = run_one(cmd, timeout_s, os.path.join(out_dir, "run.log"))

            row = {**spec, **result, **meta,
                   "timeout_s": timeout_s, "out_dir": out_dir,
                   "command": " ".join(cmd)}
            writer.writerow(row)
            rf.flush()   # survive a crash on run 37
            os.fsync(rf.fileno())

            print(f"    {result['status']} in {result['wall_s']}s", flush=True)

    print(f"\nwrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
