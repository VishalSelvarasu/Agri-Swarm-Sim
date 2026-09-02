# AgriSwarm-Sim

Decentralized multi-robot intra-row weeding in ROS 2 Jazzy + Gazebo Harmonic.
Simulation only.

---

## Decisions on record

Written down because each one is cheap now and expensive in week 5.

| Decision | Why |
|---|---|
| **ROS 2 Jazzy + Gazebo Harmonic (Ubuntu 24.04)** | Gazebo Classic went EOL in January 2025, and Jazzy does not run on 22.04. This also matches the stack already used on VISNAV. |
| **No cameras. Detection is a seeded noise model.** | Rendering dominates gz-sim cost and is what makes 12 robots infeasible. A colour threshold applied to markers you placed yourself is a lookup table with extra steps. `gz-sim-sensors-system` is not loaded at all. |
| **Custom 90-line diff-drive robot, not TurtleBot3** | TB3's value was its sensor suite. With the camera gone that value is gone, and TB3 support on Jazzy/Harmonic is the stack's largest dependency risk. |
| **Interfaces defined before nodes** | The auction protocol was one vague line in the charter. Writing the messages first forces bid deadlines, rounds, and tie-breaking to be decided now instead of surfacing as race conditions later. |
| **World is generated, not authored** | `generate_field.py --seed N` is a pure function. Same seed, byte-identical SDF and ground truth. This is what makes an N=20 harness possible rather than a retrofit. |
| **Allocator in C++** | Most robotics roles in the DE market list C++ as a hard requirement, and there is currently no ROS 2 C++ artifact in the portfolio. |

## Two things the original charter got wrong

**"≥40% reduction in simulated herbicide vs. blanket spraying" is arithmetic,
not a result.** If weed patches cover fraction *p* of the field, targeted
treatment uses *p* + false positives. At realistic densities that is an 80–95%
reduction the moment anything works at all — reachable with a hardcoded
waypoint list and no swarm. Anyone competent does this arithmetic in their head
and discounts the rest of the post. The defensible result is a **curve**:
herbicide saved vs. weeds missed, swept over the treatment confidence
threshold (`sweeps.treat_confidence_threshold` in `experiments/configs/base.yaml`).

**"Faster or comparable time-to-95%-coverage vs. distance-only auction" is a
claim you should expect to lose.** Distance-only bidding *is* the
distance-optimal assignment; weighting by confidence deliberately deviates from
it. If confidence-weighted also wins on time, suspect a rigged baseline or a
bug before believing it. State the trade honestly — precision bought with time
— and set the criterion as "time overhead < X%".

---

## Layout

```
src/agri_swarm_msgs/          5 interfaces: the auction protocol, pinned down
src/agri_swarm_core/          detector noise model, field generator
src/agri_swarm_allocation/    C++ decentralized auction (the core contribution)
src/agri_swarm_description/   diff-drive robot, no camera
src/agri_swarm_bringup/       namespaced N-robot launch
experiments/configs/          every tunable number, and nothing else
```

## Run it

```bash
# 1. build
colcon build --symlink-install && source install/setup.bash

# 2. generate a field (deterministic in --seed)
python3 -m agri_swarm_core.generate_field --seed 0 --out /tmp/worlds

# 3. launch
ros2 launch agri_swarm_bringup swarm.launch.py \
    n_robots:=4 seed:=0 world_dir:=/tmp/worlds \
    bid_mode:=confidence_energy headless:=false
```

Swap `bid_mode:=distance` for the ablation baseline. That is the only change
required between the two arms — if it ever isn't, the ablation has leaked out
of `Allocator::utility()` and the comparison is no longer clean.

## Tests

Two suites, both runnable on a machine with **no ROS 2 installed**. That is the
point: everything that can be verified without a simulator is verified without
one, so the untested surface is exactly the ROS plumbing and nothing else.

```bash
# C++: the bid ablation (824 assertions)
g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
    -Isrc/agri_swarm_allocation/include \
    src/agri_swarm_allocation/test/test_utility.cpp -o /tmp/test_utility && /tmp/test_utility

# Python: lane geometry, pure pursuit, offline scorer, field determinism
python3 -m pytest tests/ -q
```

Two of these are guard rails rather than unit tests, and they are the ones that
matter most:

- the allocator package must contain no reference to `ground_truth` or
  `patch_id` — if the bidding path ever reads the oracle, the experiment is
  invalid rather than merely wrong;
- nothing outside `utility.hpp` may branch on `bid_mode` — if the ablation
  leaks into a second file, the two arms differ by more than one function and
  the comparison is void.

CI (`.github/workflows/ci.yml`) runs both plus a syntax/manifest check. It
deliberately does **not** run `colcon build`. A green badge here means the
maths is right; it does not mean the system runs.

## Scoring and run budget

`analysis/score_run.py` joins a treatment log against `ground_truth_<seed>.csv`
and sweeps the treatment confidence threshold **offline**. Robots drive
generated lanes regardless of the threshold, so one simulated run per seed (at
the lowest threshold) yields the whole curve. That is the difference between
~560 runs and ~80.

Caveat, stated because it is load-bearing: `mission_time_s` and
`total_energy_j` do depend on the threshold, and are only valid at the
threshold actually simulated.

```bash
python3 experiments/run_batch.py --dry-run          # see the plan before queueing it
python3 analysis/score_run.py --ground-truth worlds/ground_truth_0.csv \
                              --treatments runs/base_s0_n4_distance/treatments.csv
```


---

## Week 1 milestone — one thing only

**Four robots drive their lanes end to end without touching a crop row, and
`ros2 topic echo /weed_detections` shows plausible noisy detections.**

No auction, no energy model, no metrics. If this is not running by day 7 the
ten-week plan is already gone and the honest move is to cut scope rather than
compress the tail.

Order of work:

1. `colcon build` clean. The msgs package must build before anything else.
2. Spawn **one** robot. Confirm `/robot_0/odom` moves under
   `ros2 topic pub /robot_0/cmd_vel`. Do not go to N until N=1 is solid.
3. Spawn four. Confirm four separate TF trees in RViz — `robot_0/base_link`,
   `robot_1/base_link`, etc. If you see a single shared `base_link`, stop and
   fix the prefix before writing any other code.
4. Write the lane-follower (pure pursuit along `lanes_<seed>.csv`). Nav2 is not
   needed for a field whose geometry you generated, and twelve Nav2 stacks is
   a load you have no reason to carry yet.
5. Check the detector fires. Then stop.

## Known unbuilt

- Energy monitor node (`RobotState` is published by nobody yet — the allocator
  will bid with a default energy of 1.0 J until it exists).
- Path executor: the allocator commits task IDs to `committed_` and nothing
  consumes them.
- Metrics logger and the offline scorer that joins detections against
  `ground_truth_<seed>.csv`.
- Fault injector (stop publishing `RobotState`; the allocator's silent-winner
  path already handles the rest).

## Caveat

The ROS 2 side has never been compiled or run against a live Jazzy
installation — it was written without access to one. Treat build errors on the
first `colcon build` as expected, not as evidence the design is wrong.

What *is* machine-verified: the bid utility and its ablation invariant, the
lane-assignment and pure-pursuit maths, the offline scorer, and field-generator
determinism. See **Tests** above for exactly where the verified/unverified line
falls. Nothing in this README claims a simulation result, because there isn't
one yet.
