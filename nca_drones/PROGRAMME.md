# Wavefront programme: state, rules and first steps

This file is the hand-off for anyone, person or assistant session, who continues the work. It is kept current:
when a decision is taken or a phase ends, this file changes in the same commit.

Status on 27 November 2026: kickoff done, design version 2 written. The simulator, a first map generator and
their tests are written in `wf/` and have never been executed. The H100 cluster is not connected yet, so nothing
has been run.

## 1. The rule that overrides everything

**No project code runs anywhere but on the H100 cluster and its CPUs.** That includes training, evaluation, unit
tests, shape checks, benchmarks, analysis scripts and the drawing of figures, however small. The PI's laptop holds
the git working copy and runs `git` and `ssh`, nothing else. An assistant session that edits this repository from
elsewhere may read files and results, write code and text, and use git; it must not execute project code there.

## 2. Target

At least 80% capture rate in the hardest scenario, with the wavefront architecture, learning only from reward and
a simulator, by search and learning alone.

* Grid 24x24 or larger, obstacles and a road graph with loops and dead ends.
* Ten drones: 5 capturers, 3 rechargers, 2 observers.
* An evasive target, hidden until a sensor finds it.
* Training: targets respawn, never more than one at a time. Evaluation: one target, in an episode that requires
  at least one recharge and tests flying around obstacles.
* Strict bitter-lesson adherence. Sole exception: each drone's battery level may be an input.

Standing constraints from the 8x8 phase: no solver, no dynamic programming, no value-iteration network in disguise,
no demonstrations, no shaped reward, no distance features; tubes in a (t, x, y) lattice and never a field of
arrows; not a mere forecaster; the steering command comes from the voxels.

## 3. Decisions taken by the PI at kickoff

| Topic | Decision |
|---|---|
| Execution | Cluster only, including unit tests and shape checks |
| Team | 5 capturers, 3 rechargers, 2 observers; capturers and observers have limited batteries, rechargers do not |
| Movement | Observers fly over obstacles. Capturers and the target cannot pass obstacles. Rechargers drive on roads only. The target moves on every cell that is not an obstacle |
| Target | As fast as a drone; flees from any drone it can see; wanders otherwise |
| Capture | Exact: a capturer lands on the target's cell or crosses it. A capture radius stays a setting, at zero |
| Information | One shared lattice (free, instant communication); the map is known in advance; the target is known only inside a sensor footprint |
| Cluster access | The project lives in git on the Mac; the cluster is reached over ssh from there. The Mac folder is to be linked to the assistant's session through the desktop app (Remote Control) |
| Empty battery | A drone with an empty battery cannot fly, and flies again once a recharger on its cell has refilled it |
| Search and training | Accepted: search from saved simulator states (cuts, trunks continuing from kept clones, replays with a drone at random or with its sensing withheld); exploration noise held over several steps; legs scored by 0.97 to the power of steps to capture; a training mixture of grid sides 12 / 16 / 24 and team sizes |

## 4. Waiting for the PI

The full list with options, defaults and reasons is section 15 of the design page. The ones that block work:

1. The Mac folder added in the Claude desktop app (Settings, Claude Code, Remote Control), and the cluster
   facts: ssh alias, scheduler, number of H100s and for how long, working directory, whether the cluster
   reaches GitHub and PyPI. `cluster/probe.sh` answers most of this in one run.
2. Benchmark-side tools (connectivity check, shortest-path detour measure, two scripted reference teams), never
   imported by the learner: allowed?
3. What an empty drone can still do: proposal, it cannot sense or capture and the target ignores it.
4. The bar: lower 95% bound at 80% over five training seeds, or a point estimate?
5. Recharge certificate: target farther than twice each capturer's initial charge (proposal)?
6. The target's flee rule in detail, and whether it sees through obstacles (proposal: yes).
7. Dead ends removed in evaluation maps only; an uncaptured training target moves after 48 steps (proposals).

Nothing that depends on one of these is built before it is decided. Benchmark rows are needed before phase 1,
search and training rows before phase 2 (last column of the table in section 15 of the design page).

## 5. Where things are

| What | Where |
|---|---|
| Design version 2 (the plan of record) | `docs/wavefront_v2_design.html`; sources `docs/build_design_page_v2.py`, `docs/design_v2_part1.py`, `docs/design_v2_part2.py`. Published copy, private until the PI shares it: https://claude.ai/artifact/Mx6WjPg1nanaZ45zVerBCe |
| Design version 1 (superseded) | `docs/unified_wavefront_design.html`; published copy https://claude.ai/artifact/FECZfdW6MjZgEE5TNPsQoz |
| Measured 8x8 baseline | `README.md`, `EVAL_PROTOCOL.md`, `results/` |
| Cluster kit, not yet executed anywhere | `cluster/probe.sh`, `cluster/bench_rule.py` |
| Simulator, map generator and their tests, not yet executed anywhere | `wf/sim.py`, `wf/maps.py`, `wf/test_sim.py` (run with `cd nca_drones && python3 -m wf.test_sim`). Reviewed by reading only: every assertion was traced by hand |
| Exploration noise and the first measurement, not yet executed anywhere | `wf/explore.py`, `wf/yield_table.py` (run with `cd nca_drones && python3 -m wf.yield_table`); card `experiments/2026-11-27-p1-untrained-yield.md` |
| Experiment cards | `experiments/`, one file per experiment, from `experiments/TEMPLATE.md` |

## 6. First session on the cluster (phase 0)

Run on the cluster, in this order. Each step is one command issued over ssh. Nothing here trains anything.

1. Clone the repository and check out `claude/nca-drone-coordination-d2k5kw`.
2. `bash nca_drones/cluster/probe.sh > probe_$(hostname).txt 2>&1` on the login node, and once more inside a GPU
   allocation if the login node shows no GPUs. Read the file back over ssh.
3. Create a Python environment with a CUDA build of PyTorch; record the versions that `probe.sh` prints.
4. `python3 nca_drones/cluster/bench_rule.py --out nca_drones/results/bench/$(hostname).json` inside a GPU
   allocation. This replaces the throughput estimate (100 to 400 million voxel-iterations per second per H100)
   on which every budget in the design rests.
5. `cd nca_drones && python3 test_sim.py` on a cluster CPU: the existing consistency tests.
6. `python3 analyze.py results/v10/best_prereg.pt --prereg` on a cluster CPU. Gate: it reproduces 164 of 200 on
   the test set and 1,507 of 2,000 on the never-posed tasks (closed loop).
7. Redo the cost model of design section 11 with the measured throughput and commit the result.

Phase 0 passes when steps 4 and 6 are recorded. Then phase 1 (the benchmark, still without a learner) starts.

## 7. Roadmap in one table

| Phase | Goal | Gate |
|---|---|---|
| 0 Arrival | Replace every estimate | Throughput measured; stored checkpoint reproduced |
| 1 Benchmark | Frozen benchmark with certified properties, no learner | Reference team at least 90%; one capturer 0; rechargers removed 0; observers' sensing worth 10 points; hashes committed |
| 2 Core on 8x8 | New lattice loses nothing on the measured task | Old recipe reproduced on the cluster (75.4% with shared noise, 65.5% without); new core, without shared noise, at least 75%; iteration gate and strand gate passed |
| 3 Pursuit | Visible fleeing target on 12x12, 2 to 5 capturers | At least 60% with three capturers; one capturer alone exactly 0; ghost drone credited in at most 2% of legs |
| 4 Hidden target | Observers and scouting on 16x16 | Hidden reaches three quarters of visible; withholding observers' sensing costs 10 points |
| 5 Full scenario | 24x24, ten drones, from scratch | At least 80% on validation; no drop with distance; frozen rechargers at most 5% |
| 6 Certification | Pre-registered final number | The agreed bar on the test set, all controls behaving |

## 8. How every experiment is run

1. Write the card first (`experiments/TEMPLATE.md`): one hypothesis, what supports it, what refutes it.
2. Verify on the cluster's CPUs before any GPU time: tests, shapes, a replay that reproduces a recorded episode.
3. Commit, push, check out that exact commit on the cluster, submit to the scheduler. Jobs write a status file
   every minute and resume from their latest checkpoint.
4. Read results over ssh. Look at trajectories of failures, not only at the number.
5. Fill in the card: result, what was learned, the next step. One change per run.
6. A failed gate is reported. Nothing is added to the learner to get past it.
