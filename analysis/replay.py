#!/usr/bin/env python3
from __future__ import annotations
 
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
 
# Colour-blind-safe, and distinct in greyscale print.
ROBOT_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                 "#F0E442", "#56B4E9", "#E69F00", "#000000"]
UNTREATED = "#8a8a8a"
TREATED = "#1a9850"
CROP = "#4d8c3f"
SOIL = "#c8b49a"
 
 
def load_lanes(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return [{"i": int(r["lane_index"]), "y": float(r["y"]),
                 "x_min": float(r["x_min"]), "x_max": float(r["x_max"])}
                for r in csv.DictReader(f)]
 
 
def load_weeds(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return [{"id": int(r["id"]), "x": float(r["x"]), "y": float(r["y"]),
                 "in_row": int(r["in_row"])} for r in csv.DictReader(f)]
 
 
def load_treatments(path: str) -> List[dict]:
    with open(path, newline="") as f:
        rows = [{"t": float(r["t_s"]), "robot": r["robot_id"],
                 "task": int(r["task_id"]), "x": float(r["x"]),
                 "y": float(r["y"]), "conf": float(r["confidence"])}
                for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: r["t"])
 
 
def load_awards(path: str) -> List[dict]:
    """Award rows only. They carry no position, so the caller joins by task."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return [{"t": float(r["t_s"]), "task": int(r["task_id"]),
                 "winner": r["winner_a"]}
                for r in csv.DictReader(f) if r["event"] == "award"]
 
 
def load_poses(path: str) -> Dict[str, np.ndarray]:
    """(t, x, y) per robot from poses.csv, if the run logged them."""
    if not os.path.isfile(path):
        return {}
    by: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by[r["robot_id"]].append(
                (float(r["t_s"]), float(r["x"]), float(r["y"])))
    return {k: np.array(sorted(v)) for k, v in by.items() if v}


def robot_tracks(treatments: List[dict], spawn: Dict[str, Tuple[float, float]],
                 t_end: float) -> Dict[str, np.ndarray]:
    """(t, x, y) per robot: spawn, then every treatment position in order.
 
    Approximate between those points — see the module docstring.
    """
    by_robot: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for rid, (x0, y0) in spawn.items():
        by_robot[rid].append((0.0, x0, y0))
    for r in treatments:
        by_robot[r["robot"]].append((r["t"], r["x"], r["y"]))
    for rid, pts in by_robot.items():
        pts.append((t_end, pts[-1][1], pts[-1][2]))
    return {rid: np.array(sorted(pts)) for rid, pts in by_robot.items()}
 
 
def pose_at(track: np.ndarray, t: float) -> Tuple[float, float]:
    x = np.interp(t, track[:, 0], track[:, 1])
    y = np.interp(t, track[:, 0], track[:, 2])
    return float(x), float(y)
 
 
def build(args) -> None:
    lanes = load_lanes(args.lanes)
    weeds = load_weeds(args.ground_truth)
    treatments = load_treatments(args.treatments)
    events_path = args.events or os.path.join(
        os.path.dirname(args.treatments) or ".", "events.csv")
    awards = load_awards(events_path)
 
    if not treatments:
        raise SystemExit("no treatments in the log; nothing to replay")
 
    # Award rows have no position. Join to the treatment that serviced the
    # task; awards for tasks never treated are dropped rather than guessed.
    task_xy = {r["task"]: (r["x"], r["y"]) for r in treatments}
    awards = [a for a in awards if a["task"] in task_xy]
 
    robots = sorted({r["robot"] for r in treatments})
    lane_ys = sorted(ln["y"] for ln in lanes)
    spacing = (lane_ys[1] - lane_ys[0]) if len(lane_ys) > 1 else 0.75
    # Crop rows sit between lanes.
    row_ys = [y + spacing / 2.0 for y in lane_ys[:-1]]
 
    # Spawn: each robot starts at the head of its first assigned lane. Lanes
    # are handed out in contiguous blocks, so robot i owns the i-th block.
    # assign_lanes() hands out contiguous blocks, remainder to the first
    # robots. Match that rule so the spawn positions here are the ones the
    # run actually logged.
    n = len(robots)
    base, extra = divmod(len(lanes), n)
    spawn, start = {}, 0
    for i, rid in enumerate(robots):
        spawn[rid] = (lanes[0]["x_min"] - 1.2, lane_ys[min(start, len(lane_ys) - 1)])
        start += base + (1 if i < extra else 0)
 
    t_end = treatments[-1]["t"] + 5.0
    poses_path = args.poses or os.path.join(
        os.path.dirname(args.treatments) or ".", "poses.csv")
    tracks = load_poses(poses_path)
    interpolated = not tracks
    if interpolated:
        tracks = robot_tracks(treatments, spawn, t_end)
    else:
        t_end = max(t_end, max(tr[-1, 0] for tr in tracks.values()))
        robots = sorted(set(robots) | set(tracks))
 
    x_min = min(ln["x_min"] for ln in lanes) - 2.0
    x_max = max(ln["x_max"] for ln in lanes) + 2.0
    y_min = min(lane_ys) - 1.0
    y_max = max(lane_ys) + 1.0
 
    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SOIL)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
 
    for y in row_ys:
        ax.plot([lanes[0]["x_min"], lanes[0]["x_max"]], [y, y],
                color=CROP, lw=4, solid_capstyle="butt", zorder=1)
    for y in lane_ys:
        ax.plot([lanes[0]["x_min"], lanes[0]["x_max"]], [y, y],
                color="white", lw=0.6, ls=":", alpha=0.6, zorder=2)
 
    wx = np.array([w["x"] for w in weeds])
    wy = np.array([w["y"] for w in weeds])
    weed_scatter = ax.scatter(wx, wy, s=26, c=[UNTREATED] * len(weeds),
                              edgecolors="none", zorder=3)
    weed_colours = [UNTREATED] * len(weeds)
 
    # A treatment is credited to the nearest true weed within match_radius;
    # the same rule the scorer uses, so the animation and the numbers agree.
    treat_target = []
    for r in treatments:
        d = np.hypot(wx - r["x"], wy - r["y"])
        j = int(np.argmin(d))
        treat_target.append(j if d[j] <= args.match_radius + 0.12 else -1)
 
    trails = {}
    markers = {}
    for i, rid in enumerate(robots):
        c = ROBOT_COLOURS[i % len(ROBOT_COLOURS)]
        trails[rid], = ax.plot([], [], color=c, lw=1.0, alpha=0.45, zorder=4)
        markers[rid], = ax.plot([], [], marker="s", ms=8, color=c,
                                mec="black", mew=0.6, zorder=6, label=rid)
    award_lines = [ax.plot([], [], color="black", lw=0.8, alpha=0.0,
                           zorder=5)[0] for _ in range(args.max_award_lines)]
 
    ax.legend(loc="upper right", ncol=len(robots), fontsize=8,
              framealpha=0.9)
    title = ax.set_title("")
    note = ax.text(
        0.01, 0.02,
        "robot tracks interpolated between treatments" if interpolated
        else "robot tracks from logged pose",
        transform=ax.transAxes, fontsize=7, color="#333333")
 
    n_frames = int(args.fps * t_end / args.speed) + 1
    hist = {rid: ([], []) for rid in robots}
 
    def frame(k):
        t = k * args.speed / args.fps
 
        for idx, r in enumerate(treatments):
            if r["t"] <= t and treat_target[idx] >= 0:
                weed_colours[treat_target[idx]] = TREATED
        weed_scatter.set_color(weed_colours)
 
        for rid in robots:
            x, y = pose_at(tracks[rid], t)
            hx, hy = hist[rid]
            hx.append(x); hy.append(y)
            # Keep only a recent window: a full mission of track turns the
            # plot into spaghetti and hides the motion it is meant to show.
            keep = max(1, int(args.trail_s * args.fps / args.speed))
            del hx[:-keep]; del hy[:-keep]
            trails[rid].set_data(hx, hy)
            markers[rid].set_data([x], [y])
 
        recent = [a for a in awards if 0 <= t - a["t"] <= args.award_hold]
        for li, line in enumerate(award_lines):
            if li < len(recent):
                a = recent[li]
                ax_, ay_ = task_xy[a["task"]]
                rx, ry = pose_at(tracks.get(
                    a["winner"], tracks[robots[0]]), t)
                line.set_data([rx, ax_], [ry, ay_])
                line.set_alpha(0.55 * (1.0 - (t - a["t"]) / args.award_hold))
            else:
                line.set_alpha(0.0)
 
        done = sum(1 for c in weed_colours if c == TREATED)
        title.set_text(f"t = {t:6.1f} s    treated {done}/{len(weeds)}"
                       f"    recall {done/len(weeds):.3f}")
        return ([weed_scatter, title, note]
                + list(trails.values()) + list(markers.values())
                + award_lines)
 
    anim = FuncAnimation(fig, frame, frames=n_frames, blit=False,
                         interval=1000 / args.fps)
 
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.endswith(".mp4"):
        from matplotlib.animation import FFMpegWriter
        anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    else:
        anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"wrote {args.out}  ({n_frames} frames, {t_end:.0f}s of run at "
          f"{args.speed:g}x)")
 
 
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lanes", required=True)
    p.add_argument("--ground-truth", required=True, dest="ground_truth")
    p.add_argument("--treatments", required=True)
    p.add_argument("--poses", default=None,
                   help="poses.csv. Defaults to poses.csv beside "
                        "--treatments; falls back to interpolation if absent.")
    p.add_argument("--events", default=None,
                   help="Defaults to events.csv beside --treatments.")
    p.add_argument("--out", default="replay.gif",
                   help=".gif or .mp4 (mp4 needs ffmpeg).")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--trail-s", type=float, default=60.0, dest="trail_s",
                   help="Simulated seconds of track to keep visible.")
    p.add_argument("--speed", type=float, default=20.0,
                   help="Simulated seconds per wall second.")
    p.add_argument("--award-hold", type=float, default=2.0, dest="award_hold",
                   help="Simulated seconds an award line stays visible.")
    p.add_argument("--max-award-lines", type=int, default=8,
                   dest="max_award_lines")
    p.add_argument("--match-radius", type=float, default=0.20,
                   dest="match_radius")
    p.add_argument("--width", type=float, default=16.0)
    p.add_argument("--height", type=float, default=5.0)
    p.add_argument("--dpi", type=int, default=110)
    build(p.parse_args())
 
 
if __name__ == "__main__":
    main()
