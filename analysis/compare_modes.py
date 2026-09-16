#!/usr/bin/env python3

from __future__ import annotations
 
import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from typing import Dict, List, Tuple
 
BASELINE = "distance"
TREATMENT = "energy_aware"
TREATMENT_ALIASES = {"energy_aware", "confidence_energy"}
 
# 80% power, two-sided alpha = 0.05.
Z_ALPHA = 1.959964
Z_BETA = 0.841621
 
 
def normalise_mode(m: str) -> str:
    return TREATMENT if m in TREATMENT_ALIASES else m
 
 
def load(path: str) -> Dict[Tuple[int, str], List[dict]]:
    cells: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            cells[(int(r["seed"]), normalise_mode(r["bid_mode"]))].append(r)
    return cells
 
 
def cell_mean(rows: List[dict], key: str) -> float:
    return st.mean(float(r[key]) for r in rows if r[key] != "")
 
 
def paired(cells, key: str) -> List[Tuple[int, float, float]]:
    """(seed, baseline mean, treatment mean) for seeds present in both arms."""
    out = []
    seeds = sorted({s for s, _ in cells})
    for s in seeds:
        b = cells.get((s, BASELINE))
        t = cells.get((s, TREATMENT))
        if not b or not t:
            continue
        out.append((s, cell_mean(b, key), cell_mean(t, key)))
    return out
 
 
def summarise(rows, label: str, unit: str, fmt: str, better_is_lower: bool):
    diffs = [t - b for _s, b, t in rows]
    n = len(diffs)
    if n < 2:
        print(f"{label}: only {n} paired seed(s); nothing to summarise")
        return
 
    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    lo, hi = mean - Z_ALPHA * se, mean + Z_ALPHA * se
    wins = sum(1 for d in diffs if (d < 0) == better_is_lower and d != 0)
 
    print(f"\n{label}  (paired over {n} seeds, {unit})")
    print(f"  mean Δ          {mean:+{fmt}}")
    print(f"  95% CI          [{lo:+{fmt}}, {hi:+{fmt}}]")
    print(f"  sd of Δ         {sd:{fmt}}")
    print(f"  SE              {se:{fmt}}")
    print(f"  seeds favouring {TREATMENT}: {wins}/{n}")
 
    if mean != 0.0:
        need = (Z_ALPHA + Z_BETA) ** 2 * (sd / abs(mean)) ** 2
        print(f"  an effect this size would need ~{math.ceil(need)} paired "
              f"seeds for 80% power")
 
    crosses = lo < 0 < hi
    print(f"  interval {'straddles' if crosses else 'excludes'} zero — "
          f"{'no detectable difference' if crosses else 'DIFFERENCE DETECTED'}")
 
 
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="a results.csv written by run_batch.py")
    p.add_argument("--per-seed", action="store_true",
                   help="print the per-seed table as well as the summary")
    a = p.parse_args()
 
    cells = load(a.results)
    if not cells:
        raise SystemExit(f"{a.results}: no rows with status == ok")
 
    energy = paired(cells, "total_energy_j")
    recall = paired(cells, "weeds_treated_frac")
    if not energy:
        raise SystemExit("no seed has both arms completed; nothing to pair")
 
    n_ok = sum(len(v) for v in cells.values())
    reps = st.mean(len(v) for v in cells.values())
    print(f"{a.results}")
    print(f"  {n_ok} completed runs, {len(energy)} paired seeds, "
          f"{reps:.1f} repeats per cell on average")
    print(f"  Δ is {TREATMENT} minus {BASELINE}")
 
    if a.per_seed:
        print(f"\n{'seed':>5} {'energy ' + BASELINE:>16} "
              f"{'energy ' + TREATMENT:>16} {'Δ':>9}   "
              f"{'recall b':>9} {'recall t':>9} {'Δ':>8}")
        rec = {s: (b, t) for s, b, t in recall}
        for s, b, t in energy:
            rb, rt = rec.get(s, (float("nan"), float("nan")))
            print(f"{s:5d} {b:16.0f} {t:16.0f} {t - b:+9.0f}   "
                  f"{rb:9.3f} {rt:9.3f} {rt - rb:+8.3f}")
 
    summarise(energy, "total_energy_j", "joules", ".0f", better_is_lower=True)
    summarise(recall, "weeds_treated_frac", "fraction of weeds", ".4f",
              better_is_lower=False)
 
 
if __name__ == "__main__":
    main()
