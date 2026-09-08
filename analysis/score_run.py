from __future__ import annotations
 
import argparse
import csv
import math
import sys
from collections import Counter
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
    robot_id: str        # "robot_0" etc; never arithmetic, so not an int
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
    duplicate_task_treatments: int   # same task_id treated twice -> double award
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
            Treatment(float(r["t_s"]), r["robot_id"], int(r["task_id"]),
                      float(r["x"]), float(r["y"]), float(r["confidence"]))
            for r in csv.DictReader(f)
        ]
 
 
def max_safe_match_radius_m(weeds: Sequence[Weed], row_spacing_m: float) -> float:
    """Largest match radius that cannot bridge two crop rows.
 
    A treatment matches a weed within match_radius + that weed's own radius. If
    the widest such reach exceeds half the row spacing, a treatment aimed at one
    row can be scored against a weed in the next, inflating recall and hiding
    the redundant treatment it actually was.
    """
    widest = max((w.radius for w in weeds), default=0.0)
    return row_spacing_m / 2.0 - widest
 
 
def ambiguous_treatments(
    weeds: Sequence[Weed],
    treatments: Sequence[Treatment],
    match_radius_m: float,
    row_spacing_m: float,
) -> int:
    """Count treatments whose candidate weeds straddle more than one row band.
 
    Reported rather than raised: a handful is tolerable and greedy matching
    resolves them by distance, but a large count means match_radius is too
    permissive for this field and the recall figure cannot be trusted.
    """
    n = 0
    for t in treatments:
        ys = [w.y for w in weeds
              if math.dist((t.x, t.y), (w.x, w.y)) <= match_radius_m + w.radius]
        if len(ys) >= 2 and (max(ys) - min(ys)) > row_spacing_m / 2.0:
            n += 1
    return n
 
 
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
    for ti, t in enumerate(treatments):
        for w in weeds:
            d = math.dist((t.x, t.y), (w.x, w.y))
            if d <= match_radius_m + w.radius:
                pairs.append((d, ti, w.id))
 
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
 
 
def count_duplicate_tasks(treatments: Sequence[Treatment]) -> int:
    """Treatments beyond the first for a given task_id.
 
    Distinct from redundant_treatments, which is geometric. This one is
    protocol-level: the same task was awarded to, and serviced by, more than
    one robot. Non-zero means the auction failed to reach a single winner,
    which is the failure mode the ablation is meant to separate.
    """
    counts = Counter(t.task_id for t in treatments)
    return sum(c - 1 for c in counts.values() if c > 1)
 
 
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
        duplicate_task_treatments=count_duplicate_tasks(kept),
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
                   default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    p.add_argument("--match-radius", type=float, default=0.20,
                   help="Metres, added to each weed's own radius. Should exceed "
                        "detector position_sigma by a healthy margin, and stay "
                        "below row_spacing/2 minus the largest weed radius.")
    p.add_argument("--row-spacing", type=float, default=0.75,
                   help="Crop row pitch, used to bound --match-radius.")
    p.add_argument("--out", default=None, help="Write the curve CSV here.")
    a = p.parse_args(argv)
 
    weeds = load_ground_truth(a.ground_truth)
    treatments = load_treatments(a.treatments)
 
    if not weeds:
        print("ERROR: ground truth is empty", file=sys.stderr)
        return 2
 
    # Hard bound. Above this, a treatment aimed at one crop row can be matched
    # to a weed in the next and scored as a true positive on the wrong weed,
    # which inflates recall and hides a redundant treatment at the same time.
    ceiling = max_safe_match_radius_m(weeds, a.row_spacing)
    if a.match_radius >= ceiling:
        print(
            f"ERROR: --match-radius {a.match_radius:.3f} can bridge crop rows. "
            f"With row spacing {a.row_spacing:.2f} m and a largest weed radius "
            f"of {max(w.radius for w in weeds):.3f} m the ceiling is "
            f"{ceiling:.3f} m.",
            file=sys.stderr,
        )
        return 2
 
    if not treatments:
        print("ERROR: no treatments; the run produced no rows", file=sys.stderr)
        return 2
 
    # A task position outside the field is a frame error upstream, not a weed.
    pad = 2.0
    x_lo, x_hi = min(w.x for w in weeds) - pad, max(w.x for w in weeds) + pad
    y_lo, y_hi = min(w.y for w in weeds) - pad, max(w.y for w in weeds) + pad
    stray = [t for t in treatments
             if not (x_lo <= t.x <= x_hi and y_lo <= t.y <= y_hi)]
    if stray:
        print(
            f"WARNING: {len(stray)} of {len(treatments)} treatments lie outside "
            f"the field x[{x_lo:.1f}, {x_hi:.1f}] y[{y_lo:.1f}, {y_hi:.1f}]; "
            f"first at ({stray[0].x:.2f}, {stray[0].y:.2f}). Check that every "
            f"publisher of task positions is in world frame.",
            file=sys.stderr,
        )
 
    run_min_conf = min(t.confidence for t in treatments)
    lowest = min(a.thresholds)
    if run_min_conf > lowest + 1e-9:
        print(
            f"WARNING: lowest treatment confidence in the log is {run_min_conf:.3f} "
            f"but you are sweeping down to {lowest:.3f}. The run was executed at a "
            f"higher threshold, so the low end of this curve is fabricated. "
            f"Re-run with treat_confidence_threshold={lowest}.",
            file=sys.stderr,
        )
 
    ambiguous = ambiguous_treatments(weeds, treatments, a.match_radius, a.row_spacing)
    if ambiguous:
        print(
            f"WARNING: {ambiguous} of {len(treatments)} treatments have candidate "
            f"weeds spanning more than half the row spacing. Greedy matching "
            f"resolves them by distance, but a large count means recall is "
            f"sensitive to --match-radius.",
            file=sys.stderr,
        )
 
    scores = sweep(weeds, treatments, a.thresholds, a.match_radius)
 
    print(f"{'thresh':>7} {'treated':>8} {'missed':>7} {'recall':>7} "
          f"{'applied':>8} {'FP':>5} {'redun':>6} {'dupe':>5} {'prec':>6} "
          f"{'intra':>6}")
    for s in scores:
        print(f"{s.threshold:7.2f} {s.weeds_treated:8d} {s.weeds_missed:7d} "
              f"{s.weeds_treated_frac:7.3f} {s.treatments_applied:8d} "
              f"{s.false_positive_treatments:5d} {s.redundant_treatments:6d} "
              f"{s.duplicate_task_treatments:5d} "
              f"{s.precision:6.3f} {s.intra_row_treated_frac:6.3f}")
 
    # Load balance across robots. A bid mode that strands one robot shows up
    # here before it shows up in recall.
    per_robot = Counter(t.robot_id for t in treatments)
    span = max(t.t_s for t in treatments) - min(t.t_s for t in treatments)
    print(f"\n{len(treatments)} treatments over {span:.1f} s of run time")
    for rid in sorted(per_robot):
        print(f"  {rid}: {per_robot[rid]}")
 
    if a.out:
        write_curve(scores, a.out)
        print(f"\nwrote {a.out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
