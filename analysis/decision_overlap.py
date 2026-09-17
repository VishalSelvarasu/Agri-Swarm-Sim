#!/usr/bin/env python3
from __future__ import annotations
 
import argparse
import csv
import os
import statistics as st
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
 
MODES = ("distance", "energy_aware")
MODE_ALIASES = {"confidence_energy": "energy_aware"}
 
# allocator_node.cpp initialises energy_j_ to 1.0 and bids with it until the
# first RobotState arrives; utility.hpp then treats the pack as full.
UNKNOWN_ENERGY_J = 1.0
 
Key = Tuple[int, int]  # (task_id, round)
 
 
@dataclass(frozen=True)
class Bid:
    t_s: float
    task_id: int
    round: int
    robot_id: str
    utility: float
    travel_cost_m: float
    energy_cost_j: float
    remaining_energy_j: float
    feasible: bool
 
 
@dataclass
class Auction:
    task_id: int
    round: int
    t_s: float
    n_bids: int
    n_feasible: int
    actual: str
    logged: str
    by_distance: str
    by_energy: str
    soc_spread: float
    extra_m: float
 
    @property
    def contested(self) -> bool:
        return self.n_feasible >= 2
 
    @property
    def disagree(self) -> bool:
        return self.by_distance != self.by_energy
 
 
@dataclass
class RunOverlap:
    label: str
    mode: str
    capacity_j: float
    auctions: List[Auction] = field(default_factory=list)
    unknown_energy_bids: int = 0
 
    @property
    def contested(self) -> List[Auction]:
        return [a for a in self.auctions if a.contested]
 
    @property
    def n_disagree(self) -> int:
        return sum(1 for a in self.contested if a.disagree)
 
    @property
    def disagree_rate(self) -> float:
        c = self.contested
        return self.n_disagree / len(c) if c else float("nan")
 
    def _match(self, attr: str) -> float:
        awarded = [a for a in self.auctions if a.actual]
        if not awarded:
            return float("nan")
        return sum(1 for a in awarded if a.actual == getattr(a, attr)) / len(awarded)
 
    @property
    def own_mode_match(self) -> float:
        return self._match("by_distance" if self.mode == "distance" else "by_energy")
 
    @property
    def logged_match(self) -> float:
        return self._match("logged")
 
    @property
    def extra_m(self) -> float:
        """Metres the energy_aware pick adds over the nearest robot, summed."""
        return sum(a.extra_m for a in self.contested if a.disagree)
 
    def halves(self) -> Tuple[float, float]:
        """Disagreement rate in the first and second half of the auctions."""
        c = sorted(self.contested, key=lambda a: a.t_s)
        mid = len(c) // 2
 
        def rate(xs: Sequence[Auction]) -> float:
            return sum(1 for a in xs if a.disagree) / len(xs) if xs else float("nan")
        return rate(c[:mid]), rate(c[mid:])
 
 
# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
 
def normalise_mode(m: str) -> str:
    m = MODE_ALIASES.get(m, m)
    if m not in MODES:
        raise ValueError(f"unknown bid mode {m!r}")
    return m
 
 
def load_bids(path: str) -> Dict[Key, List[Bid]]:
    """Bids grouped by auction. A robot counts once per auction: first bid wins."""
    out: Dict[Key, Dict[str, Bid]] = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            b = Bid(
                t_s=float(r["t_s"]),
                task_id=int(r["task_id"]),
                round=int(r["round"]),
                robot_id=r["robot_id"],
                utility=float(r["utility"]),
                travel_cost_m=float(r["travel_cost_m"]),
                energy_cost_j=float(r["energy_cost_j"]),
                remaining_energy_j=float(r["remaining_energy_j"]),
                feasible=r["feasible"].strip() in ("1", "True", "true"),
            )
            out[(b.task_id, b.round)].setdefault(b.robot_id, b)
    return {k: list(v.values()) for k, v in out.items()}
 
 
def load_awards(path: str) -> Dict[Key, str]:
    """Winner per awarded (task_id, round), from events.csv."""
    out: Dict[Key, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["event"] != "award":
                continue
            out.setdefault((int(r["task_id"]), int(r["round"])), r["winner_a"])
    return out
 
 
# --------------------------------------------------------------------------
# The two utilities
# --------------------------------------------------------------------------
 
def state_of_charge(b: Bid, capacity_j: float) -> float:
    if b.remaining_energy_j == UNKNOWN_ENERGY_J or capacity_j <= 0.0:
        return 1.0
    return min(1.0, max(0.0, b.remaining_energy_j / capacity_j))
 
 
def _pick(bids: Sequence[Bid], score: Callable[[Bid], float]) -> str:
    """Highest score among feasible bids; ties go to the lower robot_id,
    as in bid_beats(). Empty string when nothing is feasible."""
    best: Optional[Bid] = None
    best_s = 0.0
    for b in bids:
        if not b.feasible:
            continue
        s = score(b)
        if best is None or s > best_s or (s == best_s and b.robot_id < best.robot_id):
            best, best_s = b, s
    return best.robot_id if best else ""
 
 
def distance_winner(bids: Sequence[Bid]) -> str:
    return _pick(bids, lambda b: -b.travel_cost_m)
 
 
def energy_winner(bids: Sequence[Bid], capacity_j: float) -> str:
    return _pick(bids, lambda b: state_of_charge(b, capacity_j)
                 / (b.energy_cost_j + 1.0))
 
 
def logged_winner(bids: Sequence[Bid]) -> str:
    return _pick(bids, lambda b: b.utility)
 
 
# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
 
def analyse(bids: Dict[Key, List[Bid]], awards: Dict[Key, str],
            capacity_j: float, mode: str, label: str = "") -> RunOverlap:
    run = RunOverlap(label=label, mode=normalise_mode(mode), capacity_j=capacity_j)
    for (task_id, rnd), group in sorted(bids.items(), key=lambda kv: min(b.t_s for b in kv[1])):
        run.unknown_energy_bids += sum(
            1 for b in group if b.remaining_energy_j == UNKNOWN_ENERGY_J)
        feasible = [b for b in group if b.feasible]
        if not feasible:
            continue
        by_d = distance_winner(group)
        by_e = energy_winner(group, capacity_j)
        travel = {b.robot_id: b.travel_cost_m for b in feasible}
        socs = [state_of_charge(b, capacity_j) for b in feasible]
        run.auctions.append(Auction(
            task_id=task_id, round=rnd,
            t_s=min(b.t_s for b in group),
            n_bids=len(group), n_feasible=len(feasible),
            actual=awards.get((task_id, rnd), ""),
            logged=logged_winner(group),
            by_distance=by_d, by_energy=by_e,
            soc_spread=max(socs) - min(socs),
            extra_m=travel[by_e] - travel[by_d],
        ))
    return run
 
 
def analyse_dir(run_dir: str, capacity_j: float, mode: str,
                label: str = "") -> Optional[RunOverlap]:
    path = os.path.join(run_dir, "bids.csv")
    if not os.path.isfile(path):
        return None
    return analyse(load_bids(path), load_awards(os.path.join(run_dir, "events.csv")),
                   capacity_j, mode, label or os.path.basename(run_dir.rstrip("/")))
 
 
# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
 
def _pct(x: float) -> str:
    return "   n/a" if x != x else f"{100.0 * x:5.1f}%"
 
 
def print_runs(runs: Sequence[RunOverlap]) -> None:
    print(f"\n{'run':<38}{'contested':>10}{'disagree':>10}{'early':>8}"
          f"{'late':>8}{'extra m':>9}{'own':>8}{'logged':>8}")
    for r in runs:
        early, late = r.halves()
        print(f"{r.label:<38}{len(r.contested):>10}{_pct(r.disagree_rate):>10}"
              f"{_pct(early):>8}{_pct(late):>8}{r.extra_m:>9.1f}"
              f"{_pct(r.own_mode_match):>8}{_pct(r.logged_match):>8}")
 
 
def print_groups(runs: Sequence[RunOverlap], energy_per_m: float,
                 totals: Dict[str, float]) -> None:
    groups: Dict[Tuple[float, str], List[RunOverlap]] = defaultdict(list)
    for r in runs:
        groups[(r.capacity_j, r.mode)].append(r)
 
    print(f"\n{'capacity':>9} {'mode':<13}{'runs':>5}{'contested':>10}"
          f"{'pooled':>9}{'per-run range':>18}{'immediate ΔE':>14}")
    for (cap, mode), rs in sorted(groups.items()):
        n_c = sum(len(r.contested) for r in rs)
        n_d = sum(r.n_disagree for r in rs)
        rates = [r.disagree_rate for r in rs if r.contested]
        span = (f"{100 * min(rates):.1f}–{100 * max(rates):.1f}%"
                if rates else "n/a")
        shares = [energy_per_m * r.extra_m / totals[r.label]
                  for r in rs if totals.get(r.label)]
        share = f"{100 * st.mean(shares):+.2f}%" if shares else "n/a"
        pooled = _pct(n_d / n_c) if n_c else "   n/a"
        print(f"{cap:>9.0f} {mode:<13}{len(rs):>5}{n_c:>10}{pooled:>9}"
              f"{span:>18}{share:>14}")
    print("\nimmediate ΔE: metres the energy_aware choice adds over the nearest "
          "robot, times energy_per_m, as a share of the run's total energy. "
          "First-order only; queueing effects downstream are not included.")
 
 
def write_runs(path: str, runs: Sequence[RunOverlap]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["run", "mode", "capacity_j", "auctions", "contested",
                    "disagree", "disagree_rate", "early_rate", "late_rate",
                    "extra_m", "own_mode_match", "logged_match",
                    "unknown_energy_bids"])
        for r in runs:
            early, late = r.halves()
            w.writerow([r.label, r.mode, r.capacity_j, len(r.auctions),
                        len(r.contested), r.n_disagree,
                        f"{r.disagree_rate:.4f}", f"{early:.4f}", f"{late:.4f}",
                        f"{r.extra_m:.2f}", f"{r.own_mode_match:.4f}",
                        f"{r.logged_match:.4f}", r.unknown_energy_bids])
 
 
def write_auctions(path: str, runs: Sequence[RunOverlap]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["run", "mode", "t_s", "task_id", "round", "n_bids",
                    "n_feasible", "actual", "logged", "by_distance",
                    "by_energy", "disagree", "soc_spread", "extra_m"])
        for r in runs:
            for a in r.auctions:
                w.writerow([r.label, r.mode, f"{a.t_s:.3f}", a.task_id, a.round,
                            a.n_bids, a.n_feasible, a.actual, a.logged,
                            a.by_distance, a.by_energy, int(a.disagree),
                            f"{a.soc_spread:.4f}", f"{a.extra_m:.3f}"])
 
 
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", help="run directories holding bids.csv")
    p.add_argument("--results", help="results.csv from run_batch.py; its ok "
                   "rows supply out_dir, bid_mode and energy_capacity_j")
    p.add_argument("--mode", choices=list(MODES) + list(MODE_ALIASES),
                   help="bid mode of the run directories given directly")
    p.add_argument("--energy-capacity", type=float, dest="energy_capacity",
                   help="pack capacity of the run directories given directly")
    p.add_argument("--energy-per-m", type=float, default=12.0,
                   dest="energy_per_m")
    p.add_argument("--out", help="write one row per run to this CSV")
    p.add_argument("--out-auctions", dest="out_auctions",
                   help="write one row per auction to this CSV")
    a = p.parse_args(argv)
 
    jobs: List[Tuple[str, float, str, str]] = []
    totals: Dict[str, float] = {}
    if a.results:
        with open(a.results, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                jobs.append((r["out_dir"], float(r["energy_capacity_j"]),
                             r["bid_mode"], r["run_id"]))
                if r.get("total_energy_j"):
                    totals[r["run_id"]] = float(r["total_energy_j"])
    if a.run_dirs:
        if not a.mode or a.energy_capacity is None:
            p.error("run directories need --mode and --energy-capacity")
        for d in a.run_dirs:
            jobs.append((d, a.energy_capacity, a.mode, ""))
    if not jobs:
        p.error("give --results or at least one run directory")
 
    runs: List[RunOverlap] = []
    missing = []
    for run_dir, cap, mode, label in jobs:
        r = analyse_dir(run_dir, cap, mode, label)
        if r is None:
            missing.append(run_dir)
        else:
            runs.append(r)
 
    if missing:
        print(f"{len(missing)} run(s) have no bids.csv and were skipped; "
              f"they predate bid logging:")
        for d in missing[:5]:
            print(f"  {d}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    if not runs:
        print("nothing to analyse")
        return 1
 
    print_runs(runs)
    print_groups(runs, a.energy_per_m, totals)
 
    low = [r.label for r in runs
           if r.own_mode_match == r.own_mode_match and r.own_mode_match < 0.95]
    if low:
        print(f"\nWARNING: own-mode match below 95% in {len(low)} run(s): "
              f"{', '.join(low[:5])}. The recomputed utilities do not "
              "reproduce the allocator's decisions there.")
 
    if a.out:
        write_runs(a.out, runs)
        print(f"\nwrote {a.out}")
    if a.out_auctions:
        write_auctions(a.out_auctions, runs)
        print(f"wrote {a.out_auctions}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
