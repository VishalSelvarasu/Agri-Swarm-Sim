#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import math
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Weed:
    id: int
    x: float
    y: float
    radius: float
    in_row: int


@dataclass(frozen=True)
class Treatment:
    t_s: float
    robot_id: int
    task_id: int
    x: float
    y: float
    confidence: float


@dataclass
class Score:
    threshold: float
    weeds_total: int
    weeds_treated: int
    weeds_missed: int
    weeds_treated_frac: float
    treatments_applied: int          # herbicide units dispensed
    true_positive_treatments: int    # first hit on a real weed
    redundant_treatments: int        # same weed treated again -> wasted
    false_positive_treatments: int   # herbicide on bare soil
    precision: float
    intra_row_treated_frac: float    # the hard subset; the premise of the project


def load_ground_truth(path: str) -> List[Weed]:
    with open(path, newline="") as f:
        return [
            Weed(int(r["id"]), float(r["x"]), float(r["y"]),
                 float(r["radius"]), int(r["in_row"]))
            for r in csv.DictReader(f)
        ]


def load_treatments(path: str) -> List[Treatment]:
    with open(path, newline="") as f:
        return [
            Treatment(float(r["t_s"]), int(r["robot_id"]), int(r["task_id"]),
                      float(r["x"]), float(r["y"]), float(r["confidence"]))
            for r in csv.DictReader(f)
        ]


def match_treatments(
    weeds: Sequence[Weed],
    treatments: Sequence[Treatment],
    match_radius_m: float,
) -> Dict[int, Optional[int]]:
    """Map treatment list-index -> weed id, or None for a false positive.

    Greedy over ascending distance. A weed may be claimed once; later
    treatments landing on an already-claimed weed are redundant, not true
    positives, so wasted herbicide is counted rather than hidden.

    Greedy is not globally optimal, but the alternative (Hungarian) would need
    scipy and would flatter the result. Greedy under-counts matches slightly,
    which is the safe direction to be wrong in.
    """
    pairs = []
    candidates_by_treatment: Dict[int, List[Weed]] = {}
    for ti, t in enumerate(treatments):
        for w in weeds:
            d = math.dist((t.x, t.y), (w.x, w.y))
            if d <= match_radius_m + w.radius:
                pairs.append((d, ti, w.id))
                candidates_by_treatment.setdefault(ti, []).append(w)

    # A scorer must never silently let one treatment bridge two crop rows.
    # With the synthetic field geometry, weeds from one row occupy a narrow
    # lateral band; candidates spanning more than two match radii indicate that
    # the tolerance is wide enough to make row identity ambiguous.
    for ti, candidates in candidates_by_treatment.items():
        if len(candidates) < 2:
            continue
        y_span = max(w.y for w in candidates) - min(w.y for w in candidates)
        assert y_span <= 2.0 * match_radius_m + 1e-9, (
            f"treatment {ti} at ({treatments[ti].x:.3f}, {treatments[ti].y:.3f}) "
            f"can match weeds spanning {y_span:.3f} m laterally; "
            f"match_radius_m={match_radius_m:.3f} can bridge crop rows"
        )

    pairs.sort(key=lambda p: (p[0], p[1], p[2]))

    assignment: Dict[int, Optional[int]] = {ti: None for ti in range(len(treatments))}
    claimed_weeds = set()
    assigned_treatments = set()

    for _d, ti, wid in pairs:
        if ti in assigned_treatments or wid in claimed_weeds:
            continue
        assignment[ti] = wid
        claimed_weeds.add(wid)
        assigned_treatments.add(ti)

    return assignment


def score_at_threshold(
    weeds: Sequence[Weed],
    treatments: Sequence[Treatment],
    threshold: float,
    match_radius_m: float,
) -> Score:
    kept = [t for t in treatments if t.confidence >= threshold]
    assignment = match_treatments(weeds, kept, match_radius_m)

    treated_ids = {wid for wid in assignment.values() if wid is not None}
    tp = len(treated_ids)
    fp = sum(1 for wid in assignment.values() if wid is None)

    # A treatment landing near an already-claimed weed is redundant: it did
    # dispense herbicide, but it did not treat anything new.
    redundant = 0
    for ti, wid in assignment.items():
        if wid is None:
            near = any(
                math.dist((kept[ti].x, kept[ti].y), (w.x, w.y)) <= match_radius_m + w.radius
                for w in weeds if w.id in treated_ids
            )
            if near:
                redundant += 1
    fp -= redundant

    intra = [w for w in weeds if w.in_row == 1]
    intra_treated = sum(1 for w in intra if w.id in treated_ids)

    n = len(weeds)
    applied = len(kept)
    return Score(
        threshold=threshold,
        weeds_total=n,
        weeds_treated=tp,
        weeds_missed=n - tp,
        weeds_treated_frac=(tp / n) if n else 0.0,
        treatments_applied=applied,
        true_positive_treatments=tp,
        redundant_treatments=redundant,
        false_positive_treatments=fp,
        precision=(tp / applied) if applied else 0.0,
        intra_row_treated_frac=(intra_treated / len(intra)) if intra else 0.0,
    )


# NOTE: there is deliberately no "herbicide_reduction_vs_blanket" field.
# With weed coverage around 8% of field area, any pipeline that works at all
# reports ~90% reduction, including a hardcoded waypoint list with no swarm.
# The defensible output is the curve below: weeds_treated_frac against
# false_positive_treatments, swept over threshold.


def sweep(
    weeds: Sequence[Weed],
    treatments: Sequence[Treatment],
    thresholds: Sequence[float],
    match_radius_m: float,
) -> List[Score]:
    return [score_at_threshold(weeds, treatments, t, match_radius_m)
            for t in sorted(thresholds)]


def write_curve(scores: Sequence[Score], path: str) -> None:
    if not scores:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(scores[0]).keys()))
        w.writeheader()
        for s in scores:
            w.writerow(asdict(s))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--treatments", required=True)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    p.add_argument("--match-radius", type=float, default=0.20,
                   help="Metres, added to each weed's own radius. Should exceed "
                        "detector position_sigma by a healthy margin.")
    p.add_argument("--out", default=None, help="Write the curve CSV here.")
    a = p.parse_args(argv)

    weeds = load_ground_truth(a.ground_truth)
    treatments = load_treatments(a.treatments)

    run_min_conf = min((t.confidence for t in treatments), default=1.0)
    lowest = min(a.thresholds)
    if run_min_conf > lowest + 1e-9:
        print(
            f"WARNING: lowest treatment confidence in the log is {run_min_conf:.3f} "
            f"but you are sweeping down to {lowest:.3f}. The run was executed at a "
            f"higher threshold, so the low end of this curve is fabricated. "
            f"Re-run with treat_confidence_threshold={lowest}.",
            file=sys.stderr,
        )

    scores = sweep(weeds, treatments, a.thresholds, a.match_radius)

    print(f"{'thresh':>7} {'treated':>8} {'missed':>7} {'recall':>7} "
          f"{'applied':>8} {'FP':>5} {'redun':>6} {'prec':>6} {'intra':>6}")
    for s in scores:
        print(f"{s.threshold:7.2f} {s.weeds_treated:8d} {s.weeds_missed:7d} "
              f"{s.weeds_treated_frac:7.3f} {s.treatments_applied:8d} "
              f"{s.false_positive_treatments:5d} {s.redundant_treatments:6d} "
              f"{s.precision:6.3f} {s.intra_row_treated_frac:6.3f}")

    if a.out:
        write_curve(scores, a.out)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
