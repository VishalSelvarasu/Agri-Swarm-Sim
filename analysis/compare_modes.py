#!/usr/bin/env python3

from __future__ import annotations
 
import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
 
BASELINE = "distance"
TREATMENT = "energy_aware"
TREATMENT_ALIASES = {"energy_aware", "confidence_energy"}
 
ALPHA = 0.05
POWER = 0.80
 
# (column, label, number format, lower is better)
METRICS = [
    ("total_energy_j", "total energy (J)", ".0f", True),
    ("weeds_treated_frac", "recall", ".4f", False),
    ("mission_time_s", "mission time (s)", ".0f", True),
    ("max_energy_j", "max robot energy (J)", ".0f", True),
    ("energy_spread_j", "energy spread (J)", ".0f", True),
    ("total_distance_m", "distance (m)", ".1f", True),
]
 
# A completed run this much longer than the median is reported, because it
# is usually a near-hang that finished just inside the wall-clock limit.
LONG_RUN_FACTOR = 2.0
 
 
# --------------------------------------------------------------------------
# Student's t, without scipy
# --------------------------------------------------------------------------
 
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            return h
    raise ArithmeticError("incomplete beta did not converge")
 
 
def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b
 
 
def t_cdf(t: float, df: float) -> float:
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail
 
 
def t_ppf(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)
 
 
def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for the table [[a, b], [c, d]]."""
    row1, row2, col1 = a + b, c + d, a + c
    lo, hi = max(0, col1 - row2), min(row1, col1)
 
    def weight(k: int) -> int:
        return math.comb(row1, k) * math.comb(row2, col1 - k)
 
    observed = weight(a)
    extreme = sum(w for w in (weight(k) for k in range(lo, hi + 1))
                  if w <= observed)
    return min(1.0, extreme / math.comb(row1 + row2, col1))
 
 
# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
 
def normalise_mode(m: str) -> str:
    return TREATMENT if m in TREATMENT_ALIASES else m
 
 
def load(path: str, exclude_seeds: Sequence[int] = ()) -> List[dict]:
    """Every row, completed or not, with bid_mode normalised."""
    skip = set(exclude_seeds)
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["seed"]) in skip:
                continue
            r["bid_mode"] = normalise_mode(r["bid_mode"])
            rows.append(r)
    return rows
 
 
def ok_cells(rows: Sequence[dict]) -> Dict[Tuple[int, str], List[dict]]:
    cells: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok":
            cells[(int(r["seed"]), r["bid_mode"])].append(r)
    return cells
 
 
def cell_mean(rows: Sequence[dict], key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows if r.get(key, "") != ""]
    return st.mean(vals) if vals else None
 
 
def paired(cells, key: str) -> List[Tuple[int, float, float]]:
    """(seed, baseline mean, treatment mean) for seeds present in both arms."""
    out = []
    for s in sorted({s for s, _ in cells}):
        b = cells.get((s, BASELINE))
        t = cells.get((s, TREATMENT))
        if not b or not t:
            continue
        mb, mt = cell_mean(b, key), cell_mean(t, key)
        if mb is None or mt is None:
            continue
        out.append((s, mb, mt))
    return out
 
 
def status_counts(rows: Sequence[dict]) -> Dict[str, Tuple[int, int]]:
    """mode -> (completed, not completed)."""
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        counts[r["bid_mode"]][0 if r.get("status") == "ok" else 1] += 1
    return {m: (c[0], c[1]) for m, c in counts.items()}
 
 
def long_runs(rows: Sequence[dict]) -> Tuple[float, List[dict]]:
    times = [float(r["mission_time_s"]) for r in rows
             if r.get("status") == "ok" and r.get("mission_time_s", "") != ""]
    if not times:
        return 0.0, []
    med = st.median(times)
    slow = [r for r in rows
            if r.get("status") == "ok" and r.get("mission_time_s", "") != ""
            and float(r["mission_time_s"]) > LONG_RUN_FACTOR * med]
    return med, slow
 
 
# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
 
@dataclass
class Summary:
    n: int
    base_mean: float
    treat_mean: float
    diff: float
    sd: float
    se: float
    t_crit: float
    lo: float
    hi: float
    p_value: float
    mde: float
    equiv_bound: float
    wins: int
 
    def pct(self, value: float) -> float:
        return 100.0 * value / self.base_mean if self.base_mean else float("nan")
 
 
def summarise(rows: Sequence[Tuple[int, float, float]],
              better_is_lower: bool) -> Optional[Summary]:
    diffs = [t - b for _s, b, t in rows]
    n = len(diffs)
    if n < 2:
        return None
    df = n - 1
    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
 
    t_crit = t_ppf(1.0 - ALPHA / 2.0, df)
    t_90 = t_ppf(1.0 - ALPHA, df)
    t_beta = t_ppf(POWER, df)
 
    if se > 0.0:
        p_value = 2.0 * (1.0 - t_cdf(abs(mean) / se, df))
    else:
        p_value = 1.0 if mean == 0.0 else 0.0
 
    wins = sum(1 for d in diffs if d != 0 and (d < 0) == better_is_lower)
    return Summary(
        n=n,
        base_mean=st.mean(b for _s, b, _t in rows),
        treat_mean=st.mean(t for _s, _b, t in rows),
        diff=mean, sd=sd, se=se, t_crit=t_crit,
        lo=mean - t_crit * se, hi=mean + t_crit * se,
        p_value=p_value,
        mde=(t_crit + t_beta) * se,
        equiv_bound=max(abs(mean - t_90 * se), abs(mean + t_90 * se)),
        wins=wins,
    )
 
 
# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
 
def print_header(path: str, rows: Sequence[dict], cells, n_paired: int,
                 excluded: Sequence[int]) -> None:
    shas = sorted({r.get("git_sha", "") for r in rows})
    dirty = sum(1 for r in rows if r.get("git_dirty", "0") not in ("0", ""))
    n_ok = sum(len(v) for v in cells.values())
    reps = st.mean(len(v) for v in cells.values())
 
    print(path)
    print(f"  commit {', '.join(shas)}"
          + (f"   WARNING: {dirty} rows from a dirty tree" if dirty else ""))
    if len(shas) > 1:
        print("  WARNING: rows come from more than one commit")
    print(f"  {n_ok} completed runs, {n_paired} paired seeds, "
          f"{reps:.1f} repeats per cell on average")
    if excluded:
        print(f"  excluded seeds: {', '.join(str(s) for s in sorted(excluded))}")
 
    sc = status_counts(rows)
    b_ok, b_bad = sc.get(BASELINE, (0, 0))
    t_ok, t_bad = sc.get(TREATMENT, (0, 0))
    p = fisher_exact(b_bad, b_ok, t_bad, t_ok)
    print(f"  not completed: {BASELINE} {b_bad}/{b_ok + b_bad}, "
          f"{TREATMENT} {t_bad}/{t_ok + t_bad}  (Fisher exact p = {p:.2f})")
 
    med, slow = long_runs(rows)
    for r in slow:
        print(f"  long run: {r['run_id']} took {float(r['mission_time_s']):.0f} s "
              f"against a median of {med:.0f} s; check it with --exclude-seed "
              f"{r['seed']}")
    print(f"  Δ is {TREATMENT} minus {BASELINE}; intervals are 95% (t), "
          f"MDE at {POWER:.0%} power")
 
 
def print_table(results: Sequence[Tuple[str, str, str, Summary]]) -> None:
    head = (f"\n{'metric':<22}{'n':>3}{BASELINE:>11}{'Δ':>10}{'Δ%':>8}"
            f"   {'95% CI':<21}{'p':>6}{'MDE':>9}{'MDE%':>7}"
            f"{'equiv±%':>9}{'wins':>7}")
    print(head)
    print("-" * (len(head) - 1))
    for _key, label, fmt, s in results:
        ci = f"[{s.lo:+{fmt}}, {s.hi:+{fmt}}]"
        print(f"{label:<22}{s.n:>3}{s.base_mean:>11{fmt}}{s.diff:>+10{fmt}}"
              f"{s.pct(s.diff):>+7.1f}%   {ci:<21}{s.p_value:>6.2f}"
              f"{s.mde:>9{fmt}}{s.pct(s.mde):>6.1f}%"
              f"{s.pct(s.equiv_bound):>8.1f}%{s.wins:>4}/{s.n:<2}")
 
 
def print_per_seed(cells) -> None:
    keys = [("total_energy_j", ".0f", 9), ("weeds_treated_frac", ".3f", 7),
            ("mission_time_s", ".0f", 7)]
    cols = {k: {s: (b, t) for s, b, t in paired(cells, k)} for k, _f, _w in keys}
    seeds = sorted(cols["total_energy_j"])
    print(f"\n{'seed':>5}  {'energy d':>9} {'energy e':>9} {'Δ':>7}"
          f"   {'recall d':>8} {'recall e':>8} {'Δ':>7}"
          f"   {'time d':>7} {'time e':>7} {'Δ':>6}")
    for s in seeds:
        parts = [f"{s:5d}"]
        for k, fmt, w in keys:
            b, t = cols[k].get(s, (float("nan"), float("nan")))
            parts.append(f"{b:>{w}{fmt}} {t:>{w}{fmt}} {t - b:>+7{fmt}}")
        print("  ".join(parts[:1]) + "  " + "   ".join(parts[1:]))
 
 
def write_csv(path: str, source: str, excluded: Sequence[int],
              results: Sequence[Tuple[str, str, str, Summary]]) -> None:
    fields = ["source", "excluded_seeds", "metric", "n", "baseline_mean",
              "treatment_mean", "diff", "diff_pct", "sd", "se", "t_crit",
              "ci_lo", "ci_hi", "p_value", "mde", "mde_pct",
              "equiv_bound", "equiv_bound_pct", "wins"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for key, _label, _fmt, s in results:
            w.writerow({
                "source": source,
                "excluded_seeds": " ".join(str(x) for x in sorted(excluded)),
                "metric": key, "n": s.n,
                "baseline_mean": f"{s.base_mean:.6g}",
                "treatment_mean": f"{s.treat_mean:.6g}",
                "diff": f"{s.diff:.6g}", "diff_pct": f"{s.pct(s.diff):.3f}",
                "sd": f"{s.sd:.6g}", "se": f"{s.se:.6g}",
                "t_crit": f"{s.t_crit:.6f}",
                "ci_lo": f"{s.lo:.6g}", "ci_hi": f"{s.hi:.6g}",
                "p_value": f"{s.p_value:.4f}",
                "mde": f"{s.mde:.6g}", "mde_pct": f"{s.pct(s.mde):.3f}",
                "equiv_bound": f"{s.equiv_bound:.6g}",
                "equiv_bound_pct": f"{s.pct(s.equiv_bound):.3f}",
                "wins": s.wins,
            })
 
 
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="a results.csv written by run_batch.py")
    p.add_argument("--per-seed", action="store_true",
                   help="print the per-seed table as well as the summary")
    p.add_argument("--exclude-seed", type=int, action="append", default=[],
                   metavar="SEED", help="drop a seed from every arm (repeatable)")
    p.add_argument("--out", help="write the summary table to this CSV")
    a = p.parse_args(argv)
 
    rows = load(a.results, a.exclude_seed)
    cells = ok_cells(rows)
    if not cells:
        raise SystemExit(f"{a.results}: no rows with status == ok")
 
    results = []
    for key, label, fmt, lower in METRICS:
        s = summarise(paired(cells, key), better_is_lower=lower)
        if s is not None:
            results.append((key, label, fmt, s))
    if not results:
        raise SystemExit("fewer than two seeds have both arms completed")
 
    print_header(a.results, rows, cells, results[0][3].n, a.exclude_seed)
    print_table(results)
    if a.per_seed:
        print_per_seed(cells)
    if a.out:
        write_csv(a.out, a.results, a.exclude_seed, results)
        print(f"\nwrote {a.out}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
