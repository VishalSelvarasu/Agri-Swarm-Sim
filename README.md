# AgriSwarm-Sim

Four autonomous robots weed a crop field. Each owns a block of lanes — the gaps
between crop rows — and sweeps them end to end. When a detector reports a weed,
it goes to auction: every robot bids, one wins, and the winner interrupts its
sweep to treat it, then resumes where it left off. Robots cannot drive onto a
crop row, so they stop alongside and spray laterally. Cross-lane work routes
via the headland.

The study compares two bid functions — nearest-robot versus confidence-weighted
energy cost — over 120 simulated missions.

ROS 2 Jazzy, Gazebo Harmonic, Ubuntu 24.04. Simulation only.

![Four robots in the staged Gazebo world](docs/gazebo.png)

*Green: crop rows. Yellow: weeds. Blue: robots. A robot cannot enter a row, so
it stops in the adjacent lane and sprays sideways.*

![Replay of a distance-bid mission](docs/sweep_distance.gif)

*The same kind of run replayed from its own logs. Weeds turn green as they are
treated, robot tracks come from logged pose, and the title carries live recall.
Gazebo shows the physics; this shows the allocation.*

---

## Result

**No detectable difference between the two bid functions**, under either a
slack or a binding energy constraint. Paired by seed, completed runs only.

| condition | seeds × reps | `total_energy_j` Δ | SE | recall Δ | SE |
|---|---|---|---|---|---|
| A — 22 kJ pack, reserve gate never fires | 20 × 2 | −799 J | 840 | −0.0022 | 0.0108 |
| B — 16 kJ pack, gate fires near end of mission | 10 × 2 | +1273 J | 1027 | +0.0044 | 0.0142 |

Δ is `confidence_energy` minus `distance`. Both sit inside one standard error,
the signs disagree between the two conditions, and seed-level win counts
(14/20 and 4/10 on energy) are what coin flips look like. Run-to-run variance
at fixed seed is sd 3755 J against a mean total of ~50 kJ, so resolving the
observed 1.6% difference would need roughly 350 seeds.

Recall ran 0.56–0.84 across the batch, typically ~0.72. Every completed run
finished all four lane sweeps.

This is reported as a null because it is one. An effect found by tuning until
one appears is worth less than an honest negative with the power calculation
attached.

### Three things that came out of getting there

**Contention is zero, and that is evidence rather than an assumption.** Over a
representative mission: 160 awards across 160 distinct (task, round) pairs,
mean 4.03 bids per award, minimum 4. Zero conflicting award rounds, zero
re-announcements, zero abandonments. On single-machine loopback DDS with
reliable QoS, every robot bids on every task and exactly one wins, every time.
Split-brain is measured from the award stream — two awards for one (task,
round) naming different winners — so it does not depend on any allocator
noticing the conflict. `analysis/score_run.py` reports incidence and detection
coverage separately for exactly that reason.

**Mission energy was estimated at 1.6 kJ per robot and measured at 11–19 kJ.**
The estimate had never been probed. About 90% of distance travelled is
discretionary detour rather than lane sweeping: in one run a robot covered
1406 m against a lane length of roughly 100 m. That is the real cost of
interrupt execution, and it is also what gives the bid function something to
act on.

**Robots never crossed a crop row.** Across two runs and over 16,000 logged
poses, zero in-field poses fall between lanes. Every off-lane pose is at the
headland (x < 0 or x > 30), where lateral movement is legal. This was the one
safety-shaped claim in the design, and it now rests on measurement rather than
on the geometric argument alone.

---

## Design decisions on record

| Decision | Why |
|---|---|
| **ROS 2 Jazzy + Gazebo Harmonic (Ubuntu 24.04)** | Gazebo Classic reached EOL in January 2025, and Jazzy does not run on 22.04. |
| **No cameras. Detection is a seeded noise model.** | Rendering dominates gz-sim cost and is what would make a larger swarm infeasible. A colour threshold applied to markers you placed yourself is a lookup table with extra steps. `gz-sim-sensors-system` is not loaded at all. |
| **Custom 90-line diff-drive robot, not TurtleBot3** | TB3's value was its sensor suite. With the camera gone that value is gone, and TB3 on Jazzy/Harmonic was the stack's largest dependency risk. |
| **Interfaces before nodes** | The auction protocol was one vague line in the charter. Writing the messages first forced bid deadlines, rounds, and tie-breaking to be decided up front instead of surfacing later as race conditions. |
| **World is generated, not authored** | `generate_field.py --seed N` is a pure function: same seed, byte-identical SDF and ground truth. This is what makes a 20-seed harness possible rather than a retrofit. |
| **Allocator in C++** | The ablation lives in one ROS-free header with 832 assertions behind it, and it is the part of the system most worth writing in the language the domain uses. |
| **Interrupt execution, not two-pass** | Two-pass makes allocation a static assignment solved with complete information. Under that, both bid modes converge and the ablation shows nothing by construction. |
| **Robots never drive to a weed** | 54 of 79 weeds sit on a crop row. The outer wheel track (0.34 m) cannot enter a 0.22 m row from a lane 0.375 m away. Treating means driving along the lane to the weed's x and spraying sideways. |
| **Run status comes from the log, not the exit code** | Nodes lose races against context teardown and exit 1 on runs that completed perfectly; a run truncated by the mission timeout exits 0. `mission complete` in the log is the only reliable signal. |

## Two things the original charter got wrong

**"≥40% reduction in simulated herbicide vs. blanket spraying" is arithmetic,
not a result.** If weed patches cover fraction *p* of the field, targeted
treatment uses *p* plus false positives. At realistic densities that is an
80–95% reduction the moment anything works at all — reachable with a hardcoded
waypoint list and no swarm. The defensible output is a curve: weeds treated
against false positives, swept over the treatment confidence threshold.

**"Faster or comparable time-to-95%-coverage vs. distance-only auction" is a
claim you should expect to lose.** Distance-only bidding *is* the
distance-optimal assignment; weighting by confidence deliberately deviates from
it. If confidence-weighted had also won on time, the right response would be to
suspect a rigged baseline before believing it. As measured, neither wins.

---

## Layout

```
src/agri_swarm_msgs/          6 interfaces: the auction protocol, pinned down
src/agri_swarm_core/          detector, field generator, executors, run logger
src/agri_swarm_allocation/    C++ decentralized auction — the core contribution
src/agri_swarm_description/   diff-drive robot, no camera
src/agri_swarm_bringup/       namespaced N-robot launch, with mission shutdown
analysis/score_run.py         offline scorer, threshold sweep, contention
analysis/replay.py            top-down animation of a run, from its own logs
scripts/stage_field.py        dressed world for screenshots and video
experiments/run_batch.py      (seed × bid_mode × repeat) runner, resumable
experiments/configs/          every tunable number, and nothing else
```

## Run it

```bash
source /opt/ros/jazzy/setup.bash        # build shell: this only
colcon build
source install/setup.bash

python3 -m agri_swarm_core.generate_field --seed 0 --out $HOME/agri-worlds

ros2 launch agri_swarm_bringup swarm.launch.py \
    n_robots:=4 seed:=0 world_dir:=$HOME/agri-worlds \
    energy_capacity_j:=16000 \
    treatments_csv:=$HOME/agri-runs/base/treatments.csv \
    use_allocator:=true headless:=true
```

Missions end themselves: every executor publishes `mission_idle`, and the run
logger shuts the graph down once all of them have held idle for
`quiet_period_s`. A full mission is about 2000 simulated seconds, roughly
7 minutes of wall clock headless.

Swap `bid_mode:=distance` for the ablation baseline. That is the only change
required between the two arms — if it ever isn't, the ablation has leaked out
of `Allocator::utility()` and the comparison is no longer clean.

The whole grid, resumable and unattended:

```bash
python3 experiments/run_batch.py \
    --bid-modes distance confidence_energy --repeats 2 \
    --treat-threshold 0.5 --timeout 2400 --energy-capacity 16000 \
    --out ~/agri-runs/batch
```

It reaps stray processes between runs, generates missing fields, scores each
run inline, judges success from the log, and skips cells already recorded `ok`
when re-run. Interrupt it and start it again; it picks up where it stopped.

## Visualising a run

Gazebo shows the physics. It does not show the allocation, and from far enough
away to see four robots on a 30 m field they are specks. `analysis/replay.py`
animates a finished run top-down from its own CSVs — weeds turning grey to
green as they are treated, robot tracks from logged pose, a line from each
award to its winner, and a live recall counter.

```bash
python3 analysis/replay.py \
    --lanes $HOME/agri-worlds/lanes_0.csv \
    --ground-truth $HOME/agri-worlds/ground_truth_0.csv \
    --treatments $HOME/agri-runs/base/treatments.csv \
    --out docs/sweep_distance.gif --speed 50 --fps 15 --trail-s 45
```

For screenshots, `scripts/stage_field.py` writes a second world with the same
lane and weed geometry but throttled to real time, with shadows, a trimmed
ground plane, and crop rows drawn as plant clumps. Never run experiments
against it — at 1× a mission takes hours.

```bash
python3 scripts/stage_field.py --seed 0 --demo-seed 99
ros2 launch agri_swarm_bringup swarm.launch.py \
    n_robots:=4 seed:=99 world_dir:=$HOME/agri-worlds headless:=false
```

The second arm of the ablation, same seed and capacity:

![Replay of a confidence-energy mission](docs/sweep_confidence_energy.gif)

## Tests

259 pytest cases and 832 C++ assertions, all runnable on a machine with **no
ROS 2 installed**. Everything verifiable without a simulator is verified
without one, so the untested surface is exactly the ROS plumbing.

```bash
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
    -Isrc/agri_swarm_allocation/include \
    src/agri_swarm_allocation/test/test_utility.cpp -o /tmp/test_utility && /tmp/test_utility

python3 -m pytest tests/ -q
```

Two of these are guard rails rather than unit tests, and they matter most:

- the allocator package must contain no reference to `ground_truth` or
  `patch_id` — if the bidding path ever reads the oracle, the experiment is
  invalid rather than merely wrong;
- nothing outside `utility.hpp` may branch on `bid_mode` — if the ablation
  leaks into a second file, the two arms differ by more than one function and
  the comparison is void.

CI (`.github/workflows/ci.yml`) runs both plus a syntax and manifest check. It
deliberately does not run `colcon build`: a green badge means the maths is
right, not that the system runs.

## Scoring and run budget

`analysis/score_run.py` joins a treatment log against `ground_truth_<seed>.csv`
and sweeps the treatment confidence threshold **offline**. Robots drive the
generated lanes regardless of the threshold, so one simulated run per cell at
the lowest threshold yields the whole curve — the difference between ~560 runs
and 40.

Threshold is therefore not a sweep axis. Repeats are: run-to-run variance is
large enough that a single run per cell cannot separate a bid-mode effect from
noise.

```bash
python3 analysis/score_run.py \
    --ground-truth $HOME/agri-worlds/ground_truth_0.csv \
    --treatments $HOME/agri-runs/base/treatments.csv \
    --out-contention $HOME/agri-runs/base/contention.csv
```

Caveat, stated because it is load-bearing: `mission_time_s` and
`total_energy_j` do depend on the threshold and are valid only at the one
actually simulated. Treatment counts and derived precision/recall are valid
across the sweep. Contention figures are valid only at the executed threshold
and cannot be recovered offline.

## Limitations

- **Contention is structurally absent.** Loopback DDS with reliable QoS loses
  nothing, so split-brain and re-announcement rates are zero by construction
  rather than by protocol quality. Producing non-zero rates would need induced
  faults — `fail_robot`, or a lossy QoS profile — reported as a separate
  fault-condition study.
- **`redundant_treatments` is noise-dominated.** At `task_cell_size` 0.30,
  21.3% of paired sightings of one weed hash to different task IDs, so the
  metric measures grid fragmentation more than allocation quality. The floor is
  reported rather than tuned away.
- **`travel_cost_m` is Euclidean.** `submitBid` uses `std::hypot`, which for a
  task two lanes over reports a straight line through crop rows. Both bid modes
  are wrong identically, so the comparison survives, but cross-lane awards are
  costed optimistically.
- **Idle draw has a units error.** The standby term integrates against
  simulated seconds; observed draw is ~1.07 W against a documented 0.2 W. It
  reaches ~2.5 kJ on a long mission, which biases slower runs.
- **Field bounds derive from weed extent rather than the lanes file**, in both
  the executor and the scorer, so a few legitimate edge detections are
  discarded and robots occasionally overshoot the headland margin chasing a
  task that should have been rejected at award time.
- **Sim-only, and the simulator is frictionless.**
  `gz::sim::systems::DiffDrive` has no slip model, so its odometry is ground
  truth. Measured lateral error (3.4e-5 m over 1178 m) characterises the
  simulator, not the controller. The 9.5 cm per-side lane clearance is a
  geometric bound, verified against logged pose but not against pose noise.