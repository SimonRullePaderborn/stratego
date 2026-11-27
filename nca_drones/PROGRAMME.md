# Wavefront programme: the complete start plan

Written on 27 November 2026 as the hand-off of the kickoff session. It is meant to be enough to start the work
in a new environment without any other source: where everything is, what is decided, what is open, how to
connect the cluster, and what to run in which order, with the expected output of every step. It is kept
current: when a decision is taken or a phase ends, this file changes in the same commit.

**State in three sentences.** A first system (8x8 grid, two drones) is measured: 82.0% on its 200-task test set
and 75.4% on 2,000 tasks that were never posed in training. For the new target (24x24, ten drones of three
kinds, a hidden evasive target) there is a design, a roadmap, a simulator with tests, a map generator and the
scripts of the first two measurements. None of the new code has ever been executed, because the H100 cluster is
not connected yet and nothing may run anywhere else.

**How to use this file.**

1. The PI goes through the checklist of section 8.1. Most of it is cluster access.
2. The PI starts a new assistant session as described in section 8.2 and pastes the message of section 8.4.
3. The session works through sections 10 to 13 in order. Section 9 says how every experiment is run.

Where this file and the design page differ, this file is newer and wins. The differences are listed in
section 18.

## Contents

1. [Addresses](#1-addresses)
2. [Terms](#2-terms)
3. [Status: what exists](#3-status-what-exists)
4. [Rules that override everything](#4-rules-that-override-everything)
5. [The goal and the benchmark](#5-the-goal-and-the-benchmark)
6. [Decisions: taken and open](#6-decisions-taken-and-open)
7. [The method in brief](#7-the-method-in-brief)
8. [Starting in a new environment](#8-starting-in-a-new-environment)
9. [How work is done on the cluster](#9-how-work-is-done-on-the-cluster)
10. [Phase 0: arrival](#10-phase-0-arrival)
11. [Phase 1: the benchmark](#11-phase-1-the-benchmark)
12. [Phase 2: the core on 8x8](#12-phase-2-the-core-on-8x8)
13. [Phases 3 to 6](#13-phases-3-to-6)
14. [Gates, controls and audits](#14-gates-controls-and-audits)
15. [Cost model and budget](#15-cost-model-and-budget)
16. [Predicted failures](#16-predicted-failures)
17. [Proposed but not decided](#17-proposed-but-not-decided)
18. [Known gaps, and where this file departs from the design page](#18-known-gaps-and-where-this-file-departs-from-the-design-page)
19. [History: the 8x8 result](#19-history-the-8x8-result)

## 1. Addresses

| What | Where |
|---|---|
| Repository | https://github.com/SimonRullePaderborn/stratego |
| Branch | `claude/nca-drone-coordination-d2k5kw`. All work is on this branch. There is no pull request and nothing is merged into the default branch |
| Project folder | `nca_drones/`. Everything else in the repository belongs to another project and is not used |
| This file | `nca_drones/PROGRAMME.md`, https://github.com/SimonRullePaderborn/stratego/blob/claude/nca-drone-coordination-d2k5kw/nca_drones/PROGRAMME.md |
| The folder in a browser | https://github.com/SimonRullePaderborn/stratego/tree/claude/nca-drone-coordination-d2k5kw/nca_drones |
| Design version 2, the plan of record for the mechanism | `docs/wavefront_v2_design.html` (open in a browser). Its text is in `docs/design_v2_part1.py` (sections 1 to 6) and `docs/design_v2_part2.py` (sections 7 to 15). Published copy, private until the PI shares it: https://claude.ai/artifact/Mx6WjPg1nanaZ45zVerBCe |
| Design version 1, superseded | `docs/unified_wavefront_design.html`; published copy https://claude.ai/artifact/FECZfdW6MjZgEE5TNPsQoz |
| The measured 8x8 work | `README.md`, `EVAL_PROTOCOL.md`, `results/` |
| Side conversation about the 8x8 result | https://claude.ai/code/session_01K5Zd8SsCGMPuWWnYveByQA (works on commit `12ae323`, the last one before any wavefront design) |
| Kickoff session, in which design version 2 and this file were written | https://claude.ai/code/session_01K5Zd8UZXC6gYHJGArqK5aM |

To get the code, on the Mac (the cluster never clones; it receives files from the Mac, section 9.2):

```
git clone --branch claude/nca-drone-coordination-d2k5kw https://github.com/SimonRullePaderborn/stratego.git
cd stratego/nca_drones
```

Commits worth knowing: `12ae323` end of the 8x8 work; `b85acec` design version 1; `6934912` kickoff (design
version 2, roadmap, cluster kit); `210b98f` the PI's decisions of 27 November and the first two experiment
cards; `ee63fd2` simulator, map generator, exploration noise, first measurement script, tests; `774d7ae` fixes
from the second review; the commit after it holds this plan.

## 2. Terms

| Term | Meaning |
|---|---|
| PI | The principal investigator: the person who owns the project and takes its decisions |
| Lattice | The grid of voxels with the axes time, x and y: one slice per time step, one voxel per map cell |
| Rule | The one local update that is applied to every voxel at every iteration |
| Tube | For one drone, the marks in future slices that say where it will be |
| Front, dream | The present slice; the slices ahead of it, which the rule fills in |
| Clamp | To overwrite the input channels of a slice with what is known. State and output are never written from outside |
| Code | A one-hot vector of length 16 that a drone receives for one episode. It selects that drone's output channel |
| Greedy rule | The rule acting without exploration noise |
| Strategy | The greedy rule plus one stream of exploration noise |
| World | A seed for everything the team cannot know: where the target appears and how it breaks ties |
| Clone | One strategy run in one world from a saved state |
| Trunk, cut, leg | A training episode; a saved state just before a target is drawn; the stretch from a cut to the capture of that target or to the leg limit |
| Screen, confirm, harvest | The three stages of search at a cut (section 7.4) |
| Ghost drone | A drone with a code, a cell and a tube that the simulator ignores: it cannot sense, capture or be seen. It is used to audit credit |
| Free cell | Any cell that is not an obstacle. Road cells are free cells |
| Gate | A check that a phase must pass and every later phase must keep passing |
| Control | The same model on the same seeds with one thing taken away |
| Arm | A run that differs from the main run in one switch, to measure what that switch is worth |
| Dial | A setting that buys more search or more learning with more compute |
| Card | The file in `experiments/` that states an experiment's hypothesis before it runs and its result afterwards |
| Voxel-iteration | One application of the rule to one voxel. The unit of the cost model |
| Clone-step | One environment step of one clone, including the rule iterations it needs |

## 3. Status: what exists

| Part | Files | State |
|---|---|---|
| The 8x8 system | `env.py`, `nca.py`, `train.py`, `credit.py`, `analyze.py`, `test_sim.py`, `results/` | Measured. It has no GPU code path and runs on CPUs. The stored numbers were produced with PyTorch 2.15.0 on Python 3.13 |
| Design version 2 | `docs/` | Written. It went through a panel of five independent designs with adversarial reviews and a judge, and a second adversarial review of the text. Nothing in it has been run |
| Simulator | `wf/sim.py` | Written, never executed. Reviewed twice by reading |
| Map generator and team layouts | `wf/maps.py` | Written, never executed. One generator family of the four that are planned |
| Exploration noise | `wf/explore.py` | Written, never executed |
| Tests of the three files above | `wf/test_sim.py`, 17 tests | Never executed. Every expected value was worked out by hand from the rules. The tests added in the last review round were read once, not twice |
| First measurement | `wf/yield_table.py` | Written, never executed |
| Cluster kit | `cluster/probe.sh`, `cluster/bench_rule.py`, `cluster/check_manifest.py` | The probe ran once on 27 November (phase 0, step 3); the other two are written and not yet executed |
| Experiment cards | `experiments/` | Two cards are filled in up to "before the run" |
| The rest of the new system: lattice, search, credit, learner, evaluator, benchmark-side tools | none yet | Not written. Specified in sections 11 and 12 |
| The cluster | none | Connected on 27 November: one DGX node with 8 H100, no scheduler, jobs as Docker containers; the facts are in the phase 0 card. Phase 0 is in progress |

Code that has never run will contain errors that reading did not find. The first run of each file is therefore
a test of the file, and section 11.1 says what to do when a test fails.

## 4. Rules that override everything

### 4.1 Where code runs

The PI's rule, in the PI's words: "NO experiments, training runs, tests, or ablations may be executed on this
Mac." "Every single execution—regardless of size or overhead—must run exclusively on the H100 cluster and its
CPUs, even if remote execution takes longer." The reason given is a potential hardware issue, suspected debris
in the Mac's fan. On 27 November the PI confirmed that unit tests and shape checks fall under the rule too:
cluster only.

What follows from it:

* On the Mac: `git` and `ssh`, nothing else. No Python, no package installer, no test runner, no notebook, no
  figure script, no document builder.
* On the cluster: everything else. Its CPUs for tests, checks and analysis; its GPUs for search and learning.
* An assistant session that runs in a cloud container reads and edits files and uses git. It executes no
  project code there. During the kickoff the container also built the design page, which is a text-to-HTML
  script without project logic; from now on the page is rebuilt on the cluster (section 9.7).
* A new session confirms this rule in its first reply, before it proposes or runs anything. The PI asked for
  that acknowledgment explicitly.

### 4.2 What the learner may contain

This is the bitter-lesson contract of design section 4. The code has three parts. The environment is the
simulator and the map generator (`wf/sim.py`, `wf/maps.py`). The learner is every file that takes part in
acting, search, credit and learning (the other files of `wf/`). The benchmark side (`bench/`, not written yet)
holds everything that knows something about solutions; it may import the other two and is never imported by
them.

1. No domain concept in the learner. Distance, meeting, escort, pincer, sweep, low battery, sighting and role
   are not named anywhere in it. This is checked by a text search of the learner's source. The one exception,
   granted by the PI: each drone's battery level is an input.
2. Every ability comes from search or learning and has a compute dial.
3. More compute never hurts: more iterations, a longer window or more clones must not lower success.
4. No shaped reward, no demonstration, no staged curriculum, no solution heuristic. Tasks are drawn from one
   declared distribution. Selection uses the return and nothing else. Two things are admitted and written
   down as priors, both accepted by the PI: exploration noise that is held over several steps, and a training
   mixture over grid sizes and team sizes.
5. Every prior that remains is written down with the test that could remove it (design section 12).
6. The benchmark may know what the learner may not (still needs the PI's approval, decision O2 in section 6):
   shortest paths, scripted reference teams and certificates exist only in the generator of the evaluation
   sets and in the evaluator. Nothing they compute reaches the learner as an input, as a filter on training
   tasks or as a training signal, and the learner's code imports none of theirs. One check does touch
   training and is part of O2: every generated map, training maps included, is tested for connected free
   space and connected roads, and a map that fails is drawn again. That check is a property of a valid map,
   belongs to the environment and lives with the map generator.

Standing constraints from the 8x8 phase, all still in force: only reward and a simulator; only learning and
search; no solver, no dynamic programming, no value-iteration network in disguise; tubes in a (t, x, y)
lattice and never a field of arrows; not a mere forecaster; the steering command comes from the voxels.

What the learner is told about a drone: its cell, its kind as one of three fixed labels, its battery level and
a code that changes every episode. What a kind can do is learned.

What the learner takes from the simulator. For acting: the next observation, the reward, and the cell an
action leads to. For search and credit, in training only: saving and restoring a state, drawing what is hidden
afresh, replaying a leg with some drones acting at random, and running a leg with one drone's sensing withheld.

### 4.3 Repository and reporting

* Work only in `nca_drones/`, only on the branch `claude/nca-drone-coordination-d2k5kw`. No pull request unless
  the PI asks for one.
* All code and all git go through the Mac (the PI's rule of 27 November). Git runs on the Mac and nowhere
  else, between the Mac and GitHub. The cluster receives the files of a commit from the Mac over ssh (section
  9.2); it runs no git command and never contacts GitHub. A run uses a committed state, and its card names the
  commit. (The clone `~/wf/stratego`, made on the cluster on 27 November before this rule was given, is kept by
  the PI's decision and is not used.)
* The repository's ignore lists hide `runs/`, any `run.sh`, and every file or folder whose name starts with
  `data`, `local`, `checkpoints` or `exps`. Files that must be committed get other names.
* No personal data in file names, job names, logs that are committed, or requests to outside services. Raw
  cluster probe output holds account and host names and is not committed; its facts go into the card.
* Every number is reported with how it was obtained: measured (on what, how many episodes, which interval),
  estimated, or not yet tested. A failed gate is reported as failed. Nothing is added to the learner to get
  past a gate.
* The benchmark is frozen at the end of phase 1. After that a change of environment values needs the PI's
  approval and becomes a new version on which earlier results are reported again (proposal, decision O10).

## 5. The goal and the benchmark

### 5.1 The goal

At least 80% capture rate in the hardest scenario, with the wavefront architecture, by search and learning
alone.

* Grid of 24x24 or larger with obstacles and a road graph that has loops and dead ends.
* Ten drones of three kinds: 5 capturers, 3 rechargers, 2 observers.
* An evasive target that is hidden until a sensor finds it.
* Training: targets respawn and there is never more than one. Evaluation: one target, in an episode that
  requires at least one recharge and tests flying around obstacles.

"Capture rate" is the share of evaluation episodes that end in a capture within the episode's step limit, with
the greedy readout: one lattice, no noise, no search.

### 5.2 Benchmark version 0

Version 0 is what the simulator implements today. Values marked "estimate" are frozen at the end of phase 1
from runs that involve no learner. "Decided" was fixed by the PI. "Open" refers to a decision of section 6.

| Kind | Number in the hardest scenario | May enter | Battery | Senses | Can |
|---|---|---|---|---|---|
| Capturer | 5 (decided) | Free cells (decided) | Limited (decided); 12 steps (estimate) | 3 cells (estimate) | Capture |
| Observer | 2 (decided) | Every cell, obstacles included (decided) | Limited (decided); 24 steps (estimate) | 5 cells (estimate) | Nothing else |
| Recharger | 3 (decided) | Road cells only (decided) | Unlimited (decided) | 2 cells (estimate) | Refill every drone on its cell |
| Target | 1 at a time | Free cells (decided) | None | Drones within 2 cells (estimate) | Flee |

Rules:

* **Moves.** Stay, north, south, west, east. Everyone moves one cell per step (decided). A move into a cell
  the kind may not enter becomes a stay.
* **Order within a step.** (1) Drones and target choose from the same state. (2) All move at once. (3) Capture
  is tested. (4) Every battery loses one unit, hovering included; a drone on a recharger's cell is then
  refilled; a drone at zero is grounded (decided): it stays where it is and cannot fly until a recharger comes
  to its cell and refills it. While it is empty it also cannot sense or capture, and the target ignores it
  (open, O3). A capture on the last unit counts, and a drone that arrives on a recharger's cell with its last
  unit keeps flying. (5) Sensing. (6) In training, a new target is drawn if there is none.
* **Capture.** A capturer and the target are on the same cell after the move, or they exchanged cells during
  it (decided). A capture radius exists as a setting and is zero (decided: a larger radius would only make the
  grid smaller). An evaluation episode ends at the capture.
* **Target.** It is as fast as a drone, flees from any drone it can see and wanders otherwise (decided).
  Details (open, O4): it sees every drone, of any kind, within two cells, through obstacles. If it sees none,
  it stays or steps to a free neighbour, at random. Otherwise it takes the option, staying included, that
  maximises its distance to the nearest drone it sees, measured to where the drones stood before the move,
  with ties broken at random.
* **Information.** All drones feed one lattice, so communication is free and instant. The map is known in
  advance. The target is known only while it is inside some drone's sensing radius (all decided). A sensed
  cell reports "target here" or "sensed, empty". All distances are Manhattan distances, and no drone's sensing
  is blocked by obstacles (open, O4).
* **Reward.** One per capture. Nothing else.

The sensing radii are chosen so that a capturer sees the target one cell before the target sees it, and an
observer can shadow it from three to five cells away without making it flee. That is what gives scouting a
value.

**Training.** After a capture the next target appears on a random free cell with no drone on it. Battery
levels, empty drones included, carry over until the episode is reset after 96 steps (estimate). A target that
is not captured within 48 steps moves to a new random cell (open, O7). Maps, grid sides of 12, 16 and 24, and
teams of 1 to 5 capturers, 0 to 3 rechargers and 0 to 2 observers are drawn uniformly, without stages
(decided). Drones start on random cells their kind may enter, with battery levels drawn uniformly between 2
units and full.

**Headline evaluation.** A 24x24 map never seen in training, with the full team of ten. Half of the episodes
start with the team gathered within two cells of one road cell and with full batteries; half start as in
training, spread out and with random battery levels. One target stands farther from every capturer than twice
that capturer's initial charge (open, O5), and for the capturer nearest to it the way around the obstacles is
at least four cells longer than the straight distance. The episode lasts 128 steps (estimate). In evaluation
maps a free cell with fewer than two free neighbours is turned into an obstacle, repeatedly until none is
left, so that a dead end cannot do the capturing; where such a cell is a road cell, a neighbouring obstacle is
opened instead (open, O6).

**Maps.** Roads are laid first: junctions on a jittered coarse lattice, joined by one-cell-wide segments; a
spanning tree is kept and a share of the remaining segments is added back to make loops; dead-end spurs are
attached. Obstacles fill part of the space between roads. Three generator families are planned for training
(city blocks with alleys, scattered walls, braided mazes). A fourth family and twenty hand-drawn maps are never
trained on. Every map has connected free space and connected roads.

### 5.3 What the benchmark must guarantee

| Property | Certificate, computed outside the learner | Control, run on every reported model |
|---|---|---|
| A recharge is required | For every capturer, the target starts more than twice that capturer's initial charge away. A capturer that is not refilled acts for as many steps as its charge, and the gap closes by at most two cells per step | Rechargers removed, or left in the team with refills switched off: exactly 0. Rechargers frozen where they start: at most 5% |
| Obstacles must be flown around | For the capturer nearest to the target, the obstacle-respecting distance exceeds the Manhattan distance by 4 or more. Needs a shortest-path computation in the generator (open, O2) | Obstacle channel zeroed in the rule's input: at most half the headline rate |
| Coordination is required | With a sight of two cells, a lone drone next to the target always leaves it a safe option, so no single drone can capture where every free cell has two free neighbours. Hand proof, to be confirmed by enumeration on small grids | One capturer alone: exactly 0 |
| Capture is possible | A scripted team that always sees the target captures in at least 1 of 16 attempts on the instance, and in at least 90% of attempts over the whole set (open, O2) | The script's rate is published as a reference. Our rate is also reported on the unfiltered set |
| Scouting has value | A second scripted team, which acts on what was sensed instead of on the true position, loses at least 10 points when the observers' sensing is withheld. Otherwise the radii are recalibrated before any training (open, O2) | Observers removed; observers flying with their sensing withheld |

**Pre-registration.** Before the first training run on a benchmark version, the following go into git: the
parameter file, the generator version, the seed blocks, the control thresholds, and hashes of the validation
and test sets. Test: 500 new maps with 4 episodes each. Validation: 200 maps with 2 episodes each. One
checkpoint per training seed, chosen on validation, is evaluated once on the test set. Intervals come from a
bootstrap that resamples training seeds first and maps second.

**The bar** (open, O9). Proposed: the lower end of the 95% interval is at least 80% for the mean over five
training seeds, each trained from scratch, on the 2,000 test episodes from new maps of the training families,
with the greedy readout. The held-out map family and the hand-drawn maps are reported beside it against a
secondary bar of 60%. Phase 5 uses the point estimate on validation as a development gate; only phase 6 tests
the bar.

### 5.4 The benchmark in code

`wf/sim.py` holds every rule above. Kinds are `CAPTURER = 0`, `RECHARGER = 1`, `OBSERVER = 2`. Actions 0 to 4
are stay, north, south, west, east. A position is (row, column). Every field of the state is a tensor with the
batch first, so thousands of clones step at once.

| Setting in `sim.Cfg` | Default | Meaning |
|---|---|---|
| `H`, `W` | 24, 24 | Grid size |
| `capacity` | (12, 0, 24) | Battery in steps for capturer, recharger, observer. 0 means unlimited |
| `sense` | (3, 2, 5) | Sensing radius per kind |
| `target_sight` | 2 | The target sees drones within this distance |
| `capture_radius` | 0 | Decided: exact capture |
| `swap_captures` | True | Decided: exchanging cells with the target is a capture |
| `empty_battery` | "grounded" | Decided. "removed" takes an empty drone out for good; the 8x8 task of phase 2 uses it |
| `empty_senses`, `empty_captures`, `empty_is_seen` | False, False, False | The proposal of O3 |
| `p_move` | 1.0 | Probability that the target acts in a step. 0 gives a target that never moves |
| `relocate_after` | 0 | An uncaptured target moves to a new cell after this many steps. 0 switches it off |
| `T`, `max_spawns` | 96, 32 | Length of a world's tables of random numbers: steps and targets per episode |

Functions the learner may use: `reset`, `draw_target`, `step`, `peek`, `observe`; for search and credit also
`select`, `branch`, `new_world` and the state fields `sense_on` and `ghost`. `step` raises an error instead of
continuing silently when an episode outruns its tables.

Not implemented yet: sight blocked by obstacles (it is not a setting; sight always passes), removal of dead
ends, the evaluation-set generator with its certificates, the other map families, a text rendering of a state.

## 6. Decisions: taken and open

### 6.1 Taken by the PI

| Topic | Decision |
|---|---|
| Execution | Cluster only, including unit tests and shape checks |
| Team | 5 capturers, 3 rechargers, 2 observers. Capturers and observers have limited batteries, rechargers do not |
| Movement | Observers fly over obstacles. Capturers and the target cannot pass obstacles. Rechargers drive on roads only. The target moves on every cell that is not an obstacle |
| Target | As fast as a drone; flees from any drone it can see; wanders otherwise |
| Capture | Exact: a capturer lands on the target's cell or crosses it. The capture radius stays a setting at zero |
| Information | One shared lattice, so communication is free and instant; the map is known in advance; the target is known only inside a sensor footprint |
| Empty battery | A drone with an empty battery cannot fly, and flies again once a recharger on its cell has refilled it |
| Cluster access | The project lives in git on the Mac; the cluster is reached over ssh from there. All code and all git go through the Mac: the cluster runs no git and never contacts GitHub, it receives the files of a commit from the Mac, and results come back to the Mac, where they are committed and pushed to GitHub (27 November) |
| Search and training | Accepted as proposed: search from saved simulator states (cuts, trunks that continue from kept clones, replays with a drone acting at random or with its sensing withheld); exploration noise held over several steps; legs scored by 0.97 to the power of steps to capture; a training mixture of grid sides 12, 16, 24 and team sizes |
| Battery input | Each drone's battery level is an input to the learner. This is the sole exception to the contract |

### 6.2 Open

Each row changes what the result means or departs from "general mechanisms only". The proposal is the
recommended choice. Nothing that depends on a row is built before the PI has decided it. Answering "use the
proposals" accepts all of them.

| No. | Decision | Proposal and reason | Needed before |
|---|---|---|---|
| O1 | Cluster facts: ssh alias, scheduler, number of H100s and for how long, working directory, whether the cluster reaches PyPI | `cluster/probe.sh` answers most of it in one run | Phase 0 |
| O2 | Tools on the benchmark side, never imported by the learner | Allowed: shortest-path detour measure; a scripted team that always sees the target, for admitting instances and as a reference; a second one that acts on what was sensed, for calibrating radii; enumeration of small games; a readout of the target's cell fitted afterwards on saved states. Also allowed, in the map generator: a connectivity check on every generated map, training maps included | Phase 1, step 5 |
| O3 | What a drone with an empty battery can still do | It cannot sense or capture, and the target ignores it. An empty capturer that could still capture would void the recharge certificate | Phase 1, step 5 |
| O4 | Flee and wander rule, sight, metric | As in section 5.2: one step ahead, greatest distance to the nearest drone seen, staying allowed, two cells of sight through obstacles, Manhattan distances. Blocked sight would let one capturer ambush and would void the coordination proof | Phase 1, step 5 |
| O5 | How "requires a recharge" is certified | Target farther than twice each capturer's initial charge. It is arithmetic, it forces the recharge through pursuit instead of waiting, and it covers both start layouts | Phase 1, step 8 |
| O6 | What counts as coordination; dead ends | Any two drones in the target's sight count, and results are reported split by who was in sight. Dead ends are removed in evaluation maps only | Phase 1, step 8 |
| O7 | A training target that is not captured | It moves to a new random cell after 48 steps. Otherwise one unreachable target blocks an episode | Phase 1, step 10 |
| O8 | Starting values: sight and sensing 2 / 3 / 5 / 2 cells, batteries 12 and 24, horizon up to 128, at most 16 drones | Calibrated in phase 1 without a learner, then frozen | End of phase 1 |
| O9 | The bar | Lower 95% bound at 80% over five seeds on new maps of the training families; held-out family and hand-drawn maps beside it at 60%. For comparison: the 8x8 run met its 75% bar as a point estimate and missed it as a lower bound | End of phase 1 |
| O10 | Changing the benchmark after training has started | Only with approval, as a new version on which earlier results are reported again | Phase 2 |
| O11 | Imitation weights under a hidden target | Mean return over the worlds the team cannot tell apart, failures included. Without it a way of searching earns nothing where it found nothing | Phase 4 |
| O12 | The strand gate (new, see section 18) | A common target that stays after a capture; success means every seeker has stood on it. This is the probe that measured 34% on 8x8. A target per seeker would need an input that says whose target a cell is, which the design does not have. The gate is measured on one extra run that is trained with one to five seekers (section 12.4) | Phase 2 |
| O13 | Baselines (proposed in discussion, never decided) | Recurrent MAPPO and QMIX on the phase 3 and phase 5 benchmarks, at equal GPU-hours and with the same pooled information as the lattice. They cost about as much again as the phase's own runs | Phase 3 |
| O14 | Levers held back: a target that sometimes fails to move or sees less far in training; a noise field the lattice can see, or noise shared between drones; coarser copies of the lattice; a forecast of the target, or state that does not slide with the window | None is in the design. Each would be asked for separately, with the measurement that makes the case | When a gate fails |

## 7. The method in brief

This section is a summary of design version 2 with every number that the implementation needs. The design
page gives the reasons.

### 7.1 The lattice

The lattice has time as one of its three axes. It holds a window of 20 slices around the present: 3 behind it,
the present, and 16 ahead. On a 24x24 map that is 11,520 voxels. Slices behind hold facts. Slices ahead hold
the dream, in which each drone has a tube that marks where it will be.

| Channel group | Count | Content | Non-zero in |
|---|---|---|---|
| Map (input) | 2 | Obstacle, road | Every slice |
| Sensed (input) | 2 | This cell was inside a sensing radius; the target was found here | Past and present |
| Kind and code (input) | 3 x 16 | For each kind, the sum of the codes of the drones of that kind on this cell | Past and present |
| Battery (input) | 16 | The sum of battery level times code over the drones on this cell | Past and present |
| State | 32 | Free for the rule to use; kept from step to step | Everywhere |
| Tube (output) | 16 | A vector per voxel. Drone i's tube is its inner product with code i | Read in future slices |

That is 68 input, 32 state and 16 output channels. Clamping writes the input channels of the present slice
after every step. When the window slides, that slice becomes a past slice and keeps what was written. Future
slices hold the map and nothing else. A battery level is given in steps divided by 24, so the largest capacity
reads 1; a recharger always reads 1; an empty drone stays in the input with level zero. The lattice is not told
that a capture happened or that a new target exists; it sees only what the sensors report.

Codes: at the start of an episode every drone receives its own one-hot vector of length 16, drawn at random
without repetition. Because the assignment is redrawn every episode, no channel can come to mean a fixed drone
or role.

The rule, with X the input channels, S the state and Y the tubes:

```
c  = conv 3x3x3 ( X )                     once per step: the inputs do not change between iterations
h  = relu( conv 3x3x3 ( S ) + c )         zero padding; the only place neighbours are seen
S  = bound( S + W3 . relu( W2 . h ) )     bound: divide by max(1, root-mean-square over the voxel's channels)
Y  = Wout . S                             the 16 tube channels, one voxel at a time
```

All weights are shared across voxels, slices and iterations. There are no coordinate channels, no pooling, no
attention and no iteration counter. With 128 hidden units the rule has 366,896 weights (about 370,000). The
state is kept in full precision; only the convolutions run in a reduced numeric format.

### 7.2 Acting

```
reset
    S = 0;  X = map in every slice
    give every drone a code
    clamp the first observation into the present slice
    run the rule K0 times                                   K0 = 48

every step t
    for every drone i, all at once
        for each action a:  u[a] = sim.peek(i, a)           the cell the action leads to
        score[a] = < Y[present + 1][u[a]], code_i >
        a_i = argmax score                                  when searching: a sampled action (7.3)
    observation, reward = sim.step(a_1 ... a_n)
    slide S and X one slice towards the past                the new last slice: S = 0, X = map
    clamp the observation into the present slice
    run the rule k_s times                                  k_s = 4
```

The pseudo-code names the simulator loosely. The real calls are `sim.peek(cfg, st)`, which returns the cells
for all drones and all actions at once as [B, N, 5, 2]; `sim.step(cfg, st, actions)`, which returns the next
state and, per episode, whether the target was captured; and `sim.observe(cfg, st)` for the observation.

There is no order among drones. One step costs k_s iterations whatever the size of the team. The lattice keeps
its state from step to step. When evaluating there is one lattice, no noise and no search. Commit groups (a
random half of the drones chooses and clamps its next cell, the rule runs once, then the other half chooses)
are a dial that starts at one group and is switched on only if two drones of one kind keep ending up on one
strand.

### 7.3 Exploration noise

Implemented in `wf/explore.py`. Each strategy draws a temperature between 0.25 and 2 (log-uniform), an
excursion rate between 0 and 0.3, and a longest hold between 1 and 4. At every step a drone either starts an
excursion with that rate, which is one random action repeated for up to the hold length, or samples its action
from its scores, standardised over its five options and divided by the temperature. Noise is independent
between drones, and the rule never sees it.

### 7.4 Search

```
round
    draw 32 trunks from the training mixture                map, team, start cells, batteries
    for each trunk, until 96 steps have passed
        cut: save simulator and lattice just before a target is drawn
        screen     M = 32 strategies in W1 = 4 worlds, closed loop, the same worlds for all
                   V[strategy] = mean over worlds of 0.97 ^ (steps to capture), 0 without a capture
        confirm    the best E = 4 strategies and the greedy one in W2 = 16 fresh worlds
                   keep the best of them if it passes the test below, else store nothing
        harvest    the kept strategy in W3 = 8 further fresh worlds; only these give weights and credit
        continue the trunk from the end state of one clone
    learn from the harvested legs
```

* A cut is taken at reset, after a capture and after a relocation: moments at which nothing hidden has been
  decided yet, so every world is equally possible from the team's point of view.
* A clone runs until it captures, until 48 steps have passed (the leg limit), or until the trunk's 96th step.
* The confirmation test: for each of the W2 worlds, d is the value of the candidate minus the value of the
  greedy rule in that world. The candidate is kept only if the mean of d exceeds 2.2 times its standard error
  (the standard deviation of d divided by the square root of W2).
* Where the trunk continues: if a strategy was kept, from its end state in one harvest world drawn at random,
  and half of the time from the greedy clone in one confirmation world instead. If nothing was kept, from the
  greedy clone in one confirmation world.
* Only improvements are stored. The greedy rule's own captures are not imitated. Storing them is an arm.
* One cut costs 32 x 4 + 5 x 16 + 8 = 216 clones of at most 48 steps.
* Dropped for good: recombining recorded action lists once anything is hidden, search trees, value estimates,
  backups, and any search at evaluation time.

### 7.5 Credit

Two tests for the kept strategy, in each harvest world. Both count captures within the leg limit.

```
what the drone did                                    simulator only, no lattice
    for each of R = 2 orders of the drones            an order and its reverse
        for j = 0 ... n:  replay the leg K = 8 times; the first j drones repeat their
                          recorded actions, the others act at random, the target reacts
                          v[j] = share of the K replays that capture
        for j = 1 ... n:  share of the j-th drone = v[j] - v[j - 1]

what the drone sensed                                 with the lattice, same noise, same world
    run the leg again unchanged, and once per drone with that drone's sensing withheld
    share of drone i = capture unchanged - capture without its sensing       each 1 or 0

credit[i] = the larger of its two mean shares over orders and harvest worlds,
            if that mean clears the threshold; otherwise 0
```

In a harvest world where the kept strategy did not capture, every share is zero, and such worlds stay in the
mean. The threshold is a multiple of the standard error over the harvest worlds; the multiple is set by the
ghost audit (a ghost drone must clear it in at most 2% of legs). The second test is the only way an observer
can earn credit; it becomes a weight in phase 4, once it has passed the ghost audit there.

### 7.6 Learning

```
for each batch of 64 harvested legs, with iteration counts redrawn
    rebuild each lattice by replaying its trunk up to the cut             no gradient
    for every step t of the leg                                           recorded observations, no noise
        for every drone i and every j = 1 ... 16 with t + j inside the leg
            loss += w(world, t) . credit[i] . cross-entropy( softmax over the cells of < Y[present + j], code_i > ,
                                                             the cell of drone i at time t + j )
        after every second step: back-propagate through those two steps, update, detach
```

* w(world, t) is the mean return of the kept strategy over the harvest worlds that the team cannot tell apart
  from this world at step t, that is, worlds in which its observations have been identical up to t (open, O11).
  With everything visible every world stands alone and the weight is the leg's own return.
* Iteration counts are redrawn for every replay: K0 from 32 to 128, k_s from 2 to 6. Gaussian noise of
  standard deviation 0.02 is added to every state update during replay.
* Optimiser Adam, learning rate from 0.001 falling to 0.0002. Legs are used in the round that found them.
* There is no forecast loss, and no gradient or selection signal ever comes from the hidden position of the
  target.

### 7.7 The dials and their starting values

| Dial | Start |
|---|---|
| Trunks and cuts per round | 32 trunks, about 64 cuts |
| Strategies per cut, M | 32 |
| Worlds: screen W1, confirm W2, harvest W3; candidates E | 4, 16, 8; 4 |
| Leg limit; preference for speed | 48 steps; 0.97 |
| Iterations at reset K0, per step k_s | 48, 4 |
| Window: past, future | 3, 16 |
| State channels, hidden units | 32, 128 |
| Credit: replays K, orders R | 8, 2 |
| Block redraws of the best strategies | Off |
| Commit groups | 1 |
| Back-propagation span | 2 steps |
| Batch; learning rate | 64 legs; 0.001 falling to 0.0002 |
| Rounds per run | 900, carried over from the 8x8 runs; to be set again after a 25-round take-off test |

The take-off test is 25 rounds of a run's full configuration before the run itself. It measures the seconds
per round and the peak memory, and it passes when at least one cut has a confirmed strategy in the last ten
rounds and the training loss falls. If it fails, the full run is not started: that is the first failure of
section 16. The measured time per round then decides how many rounds the run can afford.

## 8. Starting in a new environment

### 8.1 What the PI does once

1. **A working copy on the Mac.** If there is none yet, clone it with the two commands of section 1. Git is
   allowed on the Mac.
2. **Login to the cluster from the Mac without typing anything.** Add an entry like this to `~/.ssh/config`.
   The three `Control` lines make later commands reuse the first connection, so a password or second factor is
   asked once and not at every command.

   ```
   Host h100
       HostName <address of the login node>
       User <your user name on the cluster>
       IdentityFile ~/.ssh/id_ed25519
       ControlMaster auto
       ControlPath ~/.ssh/cm-%C
       ControlPersist 8h
       ServerAliveInterval 30
       ServerAliveCountMax 4
   ```

   Check: `ssh h100 hostname` prints the name of the login node and asks nothing. This file uses the alias
   `h100` and the directory `~/wf` on the cluster in all commands; replace both if you choose other names.
3. **The cluster facts**, as far as you know them: the scheduler (Slurm or another), the names of the GPU and
   CPU partitions and of the account if one is needed, how many H100s you may use and for how long, a
   directory with at least 50 GB of space, whether the cluster reaches pypi.org, and any site rule such as
   "no computing on login nodes". What you do not know, `cluster/probe.sh` finds out.
4. **How the cluster gets the code** (section 9.2): the Mac sends it over ssh. The cluster needs no GitHub
   access, no key and no git identity.
5. **The open decisions** O2 to O7 of section 6.2, which phase 1 needs. "Use the proposals" is a valid answer.
6. Optionally the safeguard of section 8.3.
7. Start the session (8.2) and paste the message (8.4).

### 8.2 Where the assistant session runs

**Route A, recommended: a session that runs on the Mac, in the repository folder.** According to the Claude
Code documentation read on 27 November 2026, the Claude desktop app can start a local session in a folder you
choose (the project folder control next to the environment selector when starting a session), and running
`claude remote-control` in a terminal in the folder makes a session that shows up in the Claude Code app. The
session's shell is then the Mac's shell. That is why the rule of section 4.1 must be the first thing it
reads: every command it issues there must be `git` or `ssh`.

**Route B: a cloud session that is attached to the Mac folder through Remote Control.** The kickoff session
tried this. The list of folders offered to it stayed empty at every check, so this route is unconfirmed. The
folder would be added in the Claude desktop app under Settings, Claude Code, Remote Control; that page may not
offer a way to add folders. If the folder appears, the cloud session asks for it and you approve on the Mac.

A cloud session without the Mac cannot reach the cluster: it holds no key for it and its network passes
through a filtered proxy. In both routes the Mac has to be awake and online while commands are issued. Jobs
on the cluster do not depend on it.

### 8.3 Optional safeguard on the Mac

A permission file can make the session refuse the most likely slip, a direct `python3` call on the Mac. Create
`stratego/.claude/settings.local.json`, at the root of the repository whichever folder you open, with this
content:

```json
{
  "permissions": {
    "allow": ["Bash(git *)", "Bash(ssh *)"],
    "deny": ["Bash(python *)", "Bash(python3 *)", "Bash(pip *)", "Bash(pip3 *)",
             "Bash(pytest *)", "Bash(conda *)", "Bash(uv *)"]
  }
}
```

Then add the line `.claude/settings.local.json` to `.git/info/exclude`, so that git ignores the file without
any change to the repository. The repository's own ignore list does not cover it.

What this is and is not, from the same documentation: deny rules win over allow rules, and they also apply
inside compound commands. They match the command text only, so they do not catch `/usr/bin/python3`,
`bash -c "python3 ..."`, `python3.11`, a script that starts Python, or a git hook. They are a guard against
slips and not a wall. They do not block `ssh h100 'python3 ...'`, which is exactly what is wanted; that last
point is inferred from the documentation and not stated in it, so check it once with
`ssh h100 'python3 --version'`, which must run. In a terminal session `/status` lists the settings files that
were loaded; for the desktop app this is not confirmed.

### 8.4 The first message for the new session

Fill in the angle brackets and paste:

```
You are continuing a research programme. Before doing anything else, read nca_drones/PROGRAMME.md in this
repository completely. It is the plan. Then read nca_drones/docs/design_v2_part1.py and design_v2_part2.py,
which hold the text of the design.

Hard rule, no exceptions: no project code runs on this Mac. No Python, no tests, no benchmarks, no figure
scripts, no package installs. On this Mac you may only use git and ssh. Everything is executed on the H100
cluster, with commands of the form  ssh <alias> '...'.  All code and all git go through this Mac: the cluster
runs no git command and never contacts GitHub. Confirm these rules in your first reply, before you propose or
run anything.

Work only in nca_drones/, on the branch claude/nca-drone-coordination-d2k5kw. Do not open a pull request.

The cluster: ssh alias <alias>. Scheduler: <Slurm / other / unknown>. GPU partition: <name>. CPU partition:
<name>. Account: <name or none>. H100s available: <number, for how long>. Working directory: <path>.
The cluster reaches PyPI: <yes / no / unknown>.

My decisions on the open points of section 6.2: <"use the proposals", or a list such as O2 yes, O4 ...>.

Start with phase 0 (section 10). After each step report what ran, where it ran and the numbers. Write the
experiment card before each experiment. If a gate fails, report it as failed and do not add anything to the
learner to get past it. Ask me before anything that section 6.2 lists as open.
```

## 9. How work is done on the cluster

### 9.1 Layout on the cluster

```
~/wf/
    venv/            the Python environment
    runs/<card>/     per experiment: the files of the commit named in its card, sent from the Mac (9.2), with
                     COMMIT (that commit) and MANIFEST (its list of files); the run writes its output here too
    logs/            job output; jobs are started from here
    stratego/        a git clone made on the cluster on 27 November, before the rule of section 4.3; kept by the
                     PI's decision and not used
```

A run folder is filled once from the Mac and afterwards changed only by what the run writes, so a run can
always be traced to its source, and later commits do not change a job that is waiting or running.

### 9.2 Moving code

All code and all git go through the Mac (section 4.3). Git runs on the Mac only, against GitHub. The cluster
gets the files of one commit, in a run folder of its own, by these commands, issued from the repository root
on the Mac:

```
c=$(git rev-parse --short=12 HEAD)                  # or the commit that the card names
git archive $c nca_drones | ssh h100 "mkdir -p ~/wf/runs/<card> ~/wf/logs && tar -x -f - -C ~/wf/runs/<card> && echo $c > ~/wf/runs/<card>/COMMIT"
git ls-tree -r -z $c nca_drones | ssh h100 "cat > ~/wf/runs/<card>/MANIFEST"
ssh h100 'python3 -I ~/wf/runs/<card>/nca_drones/cluster/check_manifest.py ~/wf/runs/<card>'
```

The last command recomputes git's hash of every file on the cluster, with the standard library and without
git. It must report that every listed file matches and that no other file is present; it is run right after
the files arrive, before anything writes into the folder.

Results come back by one ssh call per file, `ssh h100 'cat ~/wf/runs/<card>/nca_drones/<path>' > nca_drones/<path>`
in the working copy on the Mac, where they are committed and pushed to GitHub. The cluster never commits,
pulls, fetches or pushes, and it needs no git identity and no GitHub key. Before a push the Mac fetches,
because another session may have pushed to the branch in the meantime.

### 9.3 Running a command

Every command is one ssh call from the Mac: `ssh h100 '<command>'`. The single quotes send the whole command
to the cluster, so `~`, `$USER` and `$(...)` are evaluated there. The Python of the environment is called by
its path, `~/wf/venv/bin/python3`, which avoids activating anything.

Short things that the site allows on a login node run directly. Everything else goes through the scheduler:
`srun` for a command whose output is wanted at once, `sbatch` for anything longer than a few minutes. Nothing
waits in a loop on the Mac: a job is submitted, and its state is looked at later with one call.

### 9.4 Jobs

The templates assume Slurm, which is the most common scheduler on GPU clusters. The probe shows whether that is
right. Flags differ between sites: some want `--gpus=1` or `--gres=gpu:h100:1`, some need `--account`. The
`srun` lines in this file carry no `--mem`; add one where the site's default memory is small.

A one-line job:

```
ssh h100 'cd ~/wf/logs && sbatch --job-name=<name> --partition=<gpu partition> --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --output=%x-%j.out --wrap "cd ~/wf/runs/<card> && ~/wf/venv/bin/python3 <script and arguments>"'
```

A job script for longer runs lives in the repository under `cluster/jobs/`, so that it is committed with the
experiment:

```bash
#!/bin/bash
#SBATCH --partition=<gpu partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd ~/wf/runs/<card>/nca_drones
~/wf/venv/bin/python3 -m wf.train --config experiments/<card>.json --out ~/wf/runs/<card>/out --resume auto
```

Submitted with `ssh h100 'cd ~/wf/logs && sbatch --job-name=<card> ~/wf/runs/<card>/nca_drones/cluster/jobs/<card>.sh'`.

Rules for every job of the new system:

* It writes a small status file every minute: time, round, the latest metrics, how often it was restarted.
* It appends to a log and saves a checkpoint every 5 to 10 minutes, by writing to a temporary name and
  renaming, so that a checkpoint is never half written.
* When it is submitted again it resumes from its latest checkpoint. A job that hits the time limit is simply
  submitted again.
* One experiment is one committed configuration and at most a few hours on the available GPUs.

Without a scheduler, the same script is started so that it outlives the connection:
`ssh h100 'cd ~/wf/logs && CUDA_VISIBLE_DEVICES=0 nohup bash <script> > <name>.out 2>&1 < /dev/null & echo started'`.
The cluster connected on 27 November has no scheduler and is shared: there, jobs run as Docker containers
pinned to a GPU that nothing else uses, never to one that another job holds (the phase 0 card says which GPUs
are free), so the device number above must not be copied blindly.

### 9.5 Reading results

One ssh call each: `squeue -u $USER` for the queue, `tail -n 30 ~/wf/logs/<name>-<job id>.out` for the output,
`cat ~/wf/runs/<card>/out/status.json` for the status. Metric tables and text renderings of trajectories are
read the same way. Files that should be kept (result JSON, logs, the best checkpoint, figures) come back to the
Mac by one ssh call each and are committed and pushed there (section 9.2).

### 9.6 The order of work for every increment

1. Write the card first (`experiments/TEMPLATE.md`): one hypothesis, what would support it, what would refute
   it, the one thing that changes.
2. Write the code and its tests in the working copy on the Mac. Commit and push.
3. Send the commit's files to a new run folder on the cluster and check them (section 9.2), then run the whole
   test suite on the cluster's CPUs. Nothing gets GPU time before that passes.
4. A smoke run of a few rounds or a few hundred episodes. Read the status file and a handful of trajectories.
5. The full run as a scheduler job from that run folder.
6. Read the result. Look at trajectories of failures, not only at the number. Fill in the card, including the
   GPU-hours actually used. Commit.
7. One change per run. A failed gate is reported; nothing is added to the learner to get past it.

When nothing can be executed, for example while the cluster is down, code is still verified by reading: a
second pass that traces every assertion by hand, as was done for `wf/`. That finds some errors and is not a
substitute for running the tests.

### 9.7 Documents

The design page is rebuilt on the cluster with
`ssh h100 'cd ~/wf/runs/<card>/nca_drones/docs && ~/wf/venv/bin/python3 build_design_page_v2.py wavefront_v2_design.html --standalone'`
(standard library only), brought back to the Mac and committed there. Without `--standalone` the script writes the page body alone, which is
what the published copy is made from; only an assistant session can republish it, by reading the published
page first and then publishing to the same address.

## 10. Phase 0: arrival

**Goal.** Replace every estimate the plan rests on, and show that the cluster reproduces the stored result.
Nothing is trained. Card: `experiments/2026-11-27-p0-arrival.md`. Budget: under 5 GPU-hours.

**Step 1. Connection.** `ssh h100 'hostname && whoami && date -u'`. Expected: three lines, no prompt.

**Step 2. Code.** The commands of 9.2, with `<card>` = `p0` and the commit that the card names, issued from the
repository root on the Mac. Expected: `COMMIT` holds that commit, and the check reports that every listed file
matches and that no other file is present. (On 27 November this step first ran as a clone from GitHub on the
cluster, before the rule of section 4.3 was given; that clone is `~/wf/stratego` and is not used.)

**Step 3. Probe.** It only reads; it changes nothing.

```
ssh h100 'cd ~/wf/runs/p0 && bash nca_drones/cluster/probe.sh > ~/wf/logs/probe_login.txt 2>&1; cat ~/wf/logs/probe_login.txt'
ssh h100 'cd ~/wf/runs/p0 && srun --partition=<gpu partition> --gres=gpu:1 --time=00:05:00 bash nca_drones/cluster/probe.sh > ~/wf/logs/probe_gpu.txt 2>&1; cat ~/wf/logs/probe_gpu.txt'
```

Read from it: which scheduler commands exist; partitions with their time limits and GPUs; the GPU model and
memory; the Python version and whether PyTorch is already there; free disk space; the HTTP status of PyPI and
the PyTorch download server (000 means unreachable). The probe runs no git command and contacts no GitHub
address. The raw output is not committed, because it holds account and host names. Its facts go into the card.

**Step 4. Python environment.**

```
ssh h100 'cd ~/wf && python3 -m venv venv && venv/bin/pip install --upgrade pip && venv/bin/pip install torch==2.15.0 numpy matplotlib && venv/bin/python3 -c "import torch; print(torch.__version__, torch.version.cuda)"'
ssh h100 'srun --partition=<gpu partition> --gres=gpu:1 --time=00:05:00 ~/wf/venv/bin/python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"'
```

Expected: a version and a CUDA version from the first call; `True`, a count and an H100 name from the second.
If the second call prints `False`, the installed build does not fit the cluster's GPU driver; install the
build the site recommends and record it. PyTorch 2.15.0 is the version that produced the stored 8x8 numbers.
It needs Python 3.11 or newer, and the probe of step 3 prints the cluster's version. If `python3` is older,
build the environment with a newer interpreter from the site's modules or conda before falling back to
another PyTorch version, because another version removes the expectation of step 7 that the counts match
exactly. If 2.15.0 cannot be installed at all, take the newest one and record it. The code needs nothing beyond PyTorch, NumPy and, for the old
plotting scripts, Matplotlib. If the site provides Python or PyTorch through environment modules or conda,
build the environment with those instead; what matters is that `~/wf/venv/bin/python3` exists, or that the
commands of this file are adapted to the other path. If PyPI cannot be reached, stop and report; the way
around it (a site mirror, a module, wheels copied by hand) is the PI's choice.

**Step 5. Throughput.**

```
ssh h100 'cd ~/wf/logs && sbatch --job-name=p0-bench --partition=<gpu partition> --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --output=%x-%j.out --wrap "cd ~/wf/runs/p0 && ~/wf/venv/bin/python3 nca_drones/cluster/bench_rule.py --out nca_drones/results/bench/h100.json"'
ssh h100 'squeue -u $USER; tail -n 45 ~/wf/logs/p0-bench-*.out'
```

It times 36 arms: three lattice shapes, search (no gradient) and learning (with back-propagation), three
numeric formats (fp32, fp16, bf16) and two memory layouts. Each line ends with millions of voxel-iterations
per second, TFLOPS and peak memory. An arm that fails prints FAILED and the sweep goes on.

How to read it. The number that matters is the best search-mode line of the shape "24x24 proposed rule"; call
it P. The benchmark times a stand-in with the structure of the old rule, which runs the input channels
through the 3x3x3 convolution at every iteration: 241,664 multiply-adds per voxel and iteration. The rule of
design version 2 pays for its inputs once per step and costs 131,072 per iteration. So the throughput in the
unit of the cost model is about D = 1.84 x P, on the assumption that speed is proportional to multiply-adds.
The design assumed D between 110 and 440 million, which is P between 60 and 240 million. The numeric format
and memory layout of the best line become the defaults of the new code. The learning-mode lines give the
memory that back-propagation needs per lattice. Once the real rule exists (section 12.3), it is timed
directly and replaces D.

**Step 6. The old tests on a cluster CPU.**

```
ssh h100 'cd ~/wf/runs/p0/nca_drones && srun --partition=<cpu partition> --cpus-per-task=4 --time=00:10:00 ~/wf/venv/bin/python3 test_sim.py; echo exit=$?'
```

Expected: a few lines of output, no AssertionError, `exit=0`.

**Step 7. The stored checkpoints, evaluated again.**

```
ssh h100 'cd ~/wf/logs && sbatch --job-name=p0-eval --partition=<cpu partition> --cpus-per-task=8 --mem=16G --time=03:00:00 --output=%x-%j.out --wrap "cd ~/wf/runs/p0/nca_drones && mkdir -p results/p0 && ~/wf/venv/bin/python3 analyze.py results/v10/best_prereg.pt --prereg --out results/p0/v10_prereg.json && ~/wf/venv/bin/python3 analyze.py results/v11_noshared/best_prereg.pt --prereg --out results/p0/v11_prereg.json"'
```

Expected lines for `v10`, as stored in `results/v10/analysis_prereg.out`:

```
TEST (200 held-out recharge-needed tasks)  closed-loop: 164/200 = 0.820   open-loop dream: 158/200 = 0.790
[bf16] NEW 2000 (never posed in training: True): closed 1507/2000 = 0.7535 ...   open 1467/2000 = 0.7335 ...
[bf16] TEST 200: closed 164/200   open 158/200
[fp32] NEW 2000 (never posed in training: True): closed 1507/2000 = 0.7535 ...   open 1471/2000 = 0.7355 ...
[fp32] TEST 200: closed 165/200   open 159/200
```

For `v11_noshared`: test 135/200 closed and 134/200 open; new tasks 1309/2000 closed and 1275/2000 open in
bf16, 1314/2000 and 1280/2000 in fp32.

Gate. With the same PyTorch version the counts should match exactly. On another version or processor type a
few tasks can flip through rounding; the two numeric formats already differ by up to five tasks in 2,000 in
the stored output. Proposed tolerance: the closed-loop count on the 2,000 new tasks lies within 10 of the
stored one in both formats, and the difference is recorded. A larger gap is a failed gate to be understood,
not explained away. On the machine of the stored runs one such evaluation took about 30 minutes, judging by
file times; the limit of three hours for the two is generous.

**Step 8. Restart drill.** A short job of the old trainer is stopped and continued, to learn how the scheduler
behaves before a long run depends on it:

```
ssh h100 'cd ~/wf/logs && sbatch --job-name=p0-drill --partition=<cpu partition> --cpus-per-task=2 --mem=8G --time=01:00:00 --output=%x-%j.out --wrap "cd ~/wf/runs/p0/nca_drones && ~/wf/venv/bin/python3 train.py --name p0_drill --seed 0 --strict 1 --hind 0 --prune 0 --null_k 8 --shared 0.5 --ez 4 --eps_max 0.3 --prefix 0.6 --hard_frac 0.25 --rounds 120 --eval_closed 100 --eval_every 20 --threads 2 --lr_final 0.2"'
```

The trainer writes its checkpoint `runs/p0_drill/ckpt.pt` at rounds 0, 20, 40, 60, 80 and 100. Cancel the job
at any time after round 40 has appeared in `runs/p0_drill/log.jsonl` (`scancel <job id>`), then submit the
same line again with `--resume runs/p0_drill/ckpt.pt` added at the end of the python command. The first new
log line of the resumed job must carry the round after the last checkpoint: 41 if the cancel came before
round 60, 61 if it came between rounds 60 and 79, and so on. The job must then reach round 119. Rounds that
ran between that checkpoint and the cancel appear twice in the log. The old trainer does not save its replay
buffer, so this tests the mechanics of stopping and continuing and not exact resumption; the new trainer has
to save everything it needs (section 12.3).

**Step 9. The cost model again.** With D from step 5, on one GPU: seconds per round = 6 x 10^10 / D, and
GPU-hours for a 900-round run at full scale = 1.5 x 10^10 / D (136 hours at 110 million, 34 hours at 440
million). Section 15 holds the full model. If D is below 110 million, the dials of section 7.7 shrink before
phase 2 and every budget of section 13 is redone.

**Step 10. Close the card.** Fill in the result and the GPU-hours, commit `results/bench/`, `results/p0/` and
the card, and update section 3 of this file.

**Phase 0 passes** when the throughput is recorded, the cost model is redone, the old tests pass and step 7
is inside its tolerance.

## 11. Phase 1: the benchmark

**Goal.** A frozen benchmark whose properties are shown, without any learner. Budget: about 10 GPU-hours.
Steps 1 to 4 need no decision. Steps 5 to 10 wait for the decisions O2 to O7; step 10 also needs O8 and O9.

### 11.1 Step 1: the first run of the simulator tests

```
ssh h100 'cd ~/wf/runs/<card>/nca_drones && srun --partition=<cpu partition> --cpus-per-task=4 --time=00:10:00 ~/wf/venv/bin/python3 -m wf.test_sim; echo exit=$?'
```

Expected: 17 lines that start with `ok`, one per test, in alphabetical order, and `exit=0`. The tests are
`test_a_recharger_does_not_refill_itself`, `test_battery_grounding_and_refill`,
`test_branch_shares_worlds_and_steps_are_repeatable`, `test_capture`, `test_draw_target`,
`test_exploration_noise`, `test_ghost_is_ignored`, `test_lone_capturer_never_captures`,
`test_maps_and_teams`, `test_moves_by_kind`, `test_rectangular_grid`, `test_relocation_and_respawn`,
`test_sensing`, `test_settings_for_an_empty_drone`, `test_tables_do_not_run_out_silently`,
`test_target_flees_what_it_sees` and `test_two_capturers_close_a_corner`.

This is the first execution of all of `wf/`. Errors are expected. When a test fails:

1. Read the failing assertion and work the case out by hand again from the rules of section 5.2.
2. Decide which of the two is wrong, the test or the simulator, and change exactly that one. If the rule
   itself turns out to be unclear, that is a question for the PI, not a choice to make in code.
3. Record each fix in the card: what was wrong, on which side, and how it was found.
4. Run the whole file again after every fix.

### 11.2 Step 2: the same episodes on a CPU and on a GPU

To write: `wf/test_devices.py`. It builds a batch with `maps.make_batch`, runs 48 steps of noise-driven
actions once on the CPU and once on a GPU with the same seeds, and asserts that every field of the two final
states is equal. World tables and exploration noise are drawn on the CPU for this reason. It runs inside a
GPU allocation.

### 11.3 Step 3: a text rendering, and reading episodes

To write: `wf/render.py` with `render(cfg, st, b=0)`, which returns one episode of a batch as text. One
character per cell: `#` obstacle, `=` road, `.` other free cell; `C`, `R`, `O` for a capturer, recharger or
observer with power, lower case when its battery is empty; a digit when several drones share a cell; `T` the
target, `*` when it shares a cell with a drone. Under the grid, one line per drone (slot, kind, cell, battery)
and one for the target (cell, age, sensed or not). Run as a module, it prints one noise-driven episode step by
step: `python3 -m wf.render --side 12 --team 3,1,1 --steps 20 --seed 1`.

Then read one episode for each grid side and team of the table below, step by step: do drones stop when their
battery is empty, do rechargers stay on roads, does the target move away when a drone comes within two cells?
Reading episodes is the check that the unit tests cannot give.

### 11.4 Step 4: how often does an untrained team capture?

Card: `experiments/2026-11-27-p1-untrained-yield.md`. This answers the first risk of the programme. An
untrained rule has flat tubes, so its drones follow nothing but their exploration noise; the script therefore
needs no rule and no lattice.

```
ssh h100 'cd ~/wf/runs/<card>/nca_drones && srun --partition=<gpu partition> --gres=gpu:1 --time=00:20:00 ~/wf/venv/bin/python3 -m wf.yield_table --episodes 256 --out results/p1/untrained_yield_smoke.json'
ssh h100 'cd ~/wf/logs && sbatch --job-name=p1-yield --partition=<gpu partition> --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --output=%x-%j.out --wrap "cd ~/wf/runs/<card>/nca_drones && ~/wf/venv/bin/python3 -m wf.yield_table --out results/p1/untrained_yield.json"'
```

The full run uses 4,096 episodes of one 48-step leg for each of 27 table cells: grid sides 12, 16 and 24;
teams of 2 capturers and 1 recharger, of 3 capturers, 1 recharger and 1 observer, and the full team of
5, 3 and 2; target sight 2, 1 and 0. Maps are generated in plain Python on a CPU, 4,096 for each of nine
batches, which is probably the slowest part (an estimate: 10 to 30 minutes in total).

How to read it:

* **Rows with sight 2 are the PI's target.** Rows with sight 1 and 0 are reference rows that show how steep
  the cliff is. They are measurements of the environment and not training settings.
* **The coordination check.** In every row with sight 2, the field
  `episodes_captured_with_fewer_than_two_in_sight` must be an empty list, and the counts for 0 and 1 in
  `drones_in_sight_at_capture` must be zero. The generated maps have no free cell with fewer than two free
  neighbours, and on such maps a lone drone cannot capture. If one capture with fewer than two drones in
  sight appears, stop: either the simulator or the hand proof is wrong, and nothing else proceeds until that
  is understood. Print those episodes with the renderer. In the rows with sight 1 or 0, captures by a single
  drone are expected.
* **Can reward be reached?** Look at captures per 4,096 episodes with sight 2. If p is that share, a cut with
  128 screening clones contains at least one capture with probability 1 - (1 - p)^128: 47% for p = 0.5%, 12%
  for p = 0.1%. Fewer than 10 captures per 4,096 on 12x12 as well means that smaller grids do not make reward
  reachable; then the levers of O14 have to be asked for, with this table as the evidence.
* Also recorded: the mean step of capture, in how many captures the target had been sensed before, and the
  share of capturers that still have power after 48 steps.

### 11.5 Step 5: tools on the benchmark side (after O2)

To write, as a package `bench/` beside `wf/`, so that "the learner imports no benchmark code" is a check on
one directory name:

| File | Content |
|---|---|
| `bench/graph.py` | `distances(free, source)`, the breadth-first distances on free cells; `detour(free, a, b)`, path distance minus Manhattan distance; `remove_dead_ends(obstacle, road)` by the rule of section 5.2 |
| `bench/scripted.py` | The two scripted teams (step 9) |
| `bench/enumerate_small.py` | Enumeration of small games (step 7) |
| `bench/make_sets.py` | The validation and test sets with their certificates and hashes (step 8) |
| `bench/controls.py` | The control variants of an episode (section 14.2) |
| `bench/evaluate.py` | Evaluation of a scripted team or, later, of a checkpoint on a stored set, with an optional control; counts with intervals and paired differences (section 14.1) |
| `bench/test_bench.py` | Hand-worked cases for all of the above |

Additions to the environment that belong to this step, each with a test: in `wf/sim.py` a setting that
switches refills off and, only if O4 is decided that way, sight blocked by obstacles; in `wf/maps.py` a
`connected(mask)` check by flood fill, for the map families of step 6.

### 11.6 Step 6: the other map families

The existing generator is one family: a road network with loops and spurs, and rectangular obstacles that
keep one cell of distance from each other and from the border. Its guarantees hold by construction. The
other planned families (city blocks with alleys, braided mazes, and a fourth that is never trained on) cannot
be built that way; they generate, check connectivity, and draw again when the check fails. They live in
`wf/maps.py` with the connectivity check of step 5, because training needs them and `wf/` must not import
`bench/`. Each family gets its own block of seeds. Per family and grid side, phase 1 measures the share of free cells, the share of road
cells, and how far a free cell lies from the nearest road; the last value is fixed when the benchmark is
frozen. Twenty maps are drawn by hand as text files in the renderer's format.

### 11.7 Step 7: enumeration of small games

The claim "a lone capturer never captures where every free cell has two free neighbours" rests on a hand
proof. A capture happens within one step, so the check needs no game tree: for every map in a set of small
maps (all 4x4 and 5x5 maps without obstacles, and a few hundred random ones up to 6x6 with obstacles and
without dead ends), for every pair of a capturer cell and a target cell within two cells of each other, for
every capturer action and for every option the flee rule could choose under any tie-break, no capture may
occur. The same enumeration with two drones gives the share of positions from which two drones can force a
capture, which is useful for calibration. Any counterexample goes to the PI before anything else is built.

### 11.8 Step 8: the validation and test sets (after O5 and O6)

```
for each map seed in the set's block
    map = a generator family at 24x24 with that seed; remove dead ends
    for each episode on this map                       4 in the test set, 2 in the validation set
        even index: gathered start. Pick a road cell; place all ten drones on cells their kind may
                    enter within two cells of it; full batteries
        odd index:  spread start as in training: random cells, battery between 2 and full
        candidates = free cells without a drone that lie farther from every capturer than twice
                     that capturer's charge
        keep those for which, for the capturer nearest by Manhattan distance, the path distance on
                     free cells exceeds the Manhattan distance by 4 or more
        if none is left: draw the layout again and count the redraw
        target = one of the kept cells, drawn uniformly; world seed = drawn once and stored
        admit the episode if the first scripted team captures in at least 1 of 16 attempts;
            the attempts use the world seeds stored seed + 0 ... 15, and the first is the episode's own
```

With full batteries of 12 the target must be more than 24 cells from all five capturers, so a gathered start
has to sit towards a corner or an edge; the redraw takes care of that. Stored per episode: map, family,
layout type, cells, kinds, batteries, target cell, world seed, the two certificate values and the script's
result. A second, unfiltered set is built without the admission step, so that the share of rejected instances
and our rate on the unfiltered set can be reported. Proposed seed blocks, fixed in the pre-registration:
training from 1,000,000, validation from 2,000,000, test from 3,000,000, the held-out family from 4,000,000.

### 11.9 Step 9: scripted teams, calibration and controls without a learner

The first scripted team always knows the target's cell. A starting version: every recharger drives along
roads towards the capturer with the least battery that no other recharger is heading for. A capturer whose
battery is no more than its path distance to the nearest recharger plus two flies to that recharger; the
others fly to the target, each to a different free neighbour cell of it, and a capturer next to the target
steps onto the cell the target would flee to if another capturer is also within two cells. The second
scripted team is the same with the last sensed cell in place of the true one; while nothing has been sensed,
observers sweep the map in rows and capturers stay within two cells of a recharger. Both are hand-written
heuristics that exist to show that the benchmark is solvable and that scouting has value. They will need
several rounds of improvement, and they never enter the learner.

Calibration varies the values of O8 (batteries, radii, the step limit) until the properties below hold, and
then they are frozen:

| Check, on the validation set | Required |
|---|---|
| First scripted team | Captures in at least 90% of episodes |
| One capturer alone | Exactly 0 |
| Rechargers removed; rechargers present with refills off | Exactly 0 |
| Rechargers frozen where they start | At most 5% |
| Second scripted team with and without the observers' sensing | The sensing is worth at least 10 points |
| Random team | At most 1% |

**Kill criterion of phase 1:** if no setting of the values meets these, the values go back to the PI.

### 11.10 Step 10: freeze

Commit as benchmark version 1: a parameter file with every `Cfg` value and every generator parameter, the
generator version, the seed blocks, the control thresholds, the sets, and the SHA-256 hashes of the sets in a
new pre-registration file `EVAL_PROTOCOL_V2.md`, which also states the bar (O9), the checkpoint-selection
rule and the bootstrap. From here on the benchmark changes only by decision O10.

## 12. Phase 2: the core on 8x8

**Goal.** Show that the new lattice loses nothing on the task that was measured. About 20 runs plus six runs
of the old code. Budget: 25 to 95 GPU-hours.

### 12.1 First the old recipe, unchanged, on the cluster

Six runs on CPUs: seeds 0, 1 and 2, each with shared noise (`--shared 0.5`, the stored run `v10`) and without
(`--shared 0.0`, the stored run `v11_noshared`). The stored runs were started with 1,500 rounds, which sets
the learning-rate schedule, and were evaluated once with the `best.pt` that existed when round 900 was
reached (`EVAL_PROTOCOL.md`). The job does the same:

```bash
#!/bin/bash
# arguments: <name> <seed> <shared>      for example: p2a_shared_s0 0 0.5
set -euo pipefail
name=$1; seed=$2; shared=$3
py=~/wf/venv/bin/python3
cd ~/wf/runs/p2a/nca_drones
if [ -e runs/$name/log.jsonl ]; then echo "runs/$name exists already; use a new name"; exit 1; fi
$py train.py --name $name --seed $seed --strict 1 --hind 0 --prune 0 --null_k 8 --shared $shared --ez 4 \
    --eps_max 0.3 --prefix 0.6 --hard_frac 0.25 --rounds 1500 --eval_closed 100 --eval_every 50 \
    --threads 2 --lr_final 0.2 &
pid=$!
until grep -q '"round": 900,' runs/$name/log.jsonl 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "the trainer stopped early"; exit 1; }
    sleep 30
done
cp runs/$name/best.pt runs/$name/best_r900.pt
kill $pid 2>/dev/null || true
wait $pid 2>/dev/null || true
mkdir -p results/p2a
$py analyze.py runs/$name/best_r900.pt --prereg --out results/p2a/$name.json | tee results/p2a/$name.out
cp runs/$name/log.jsonl results/p2a/$name.log.jsonl
cp runs/$name/best_r900.pt results/p2a/$name.best_r900.pt
```

The script is committed as `cluster/jobs/p2a_old_recipe.sh`, the run folder `~/wf/runs/p2a` is filled from the
Mac by the commands of section 9.2, and each of the six jobs is submitted like this one:

```
ssh h100 'cd ~/wf/logs && sbatch --job-name=p2a_shared_s0 --partition=<cpu partition> --cpus-per-task=2 --mem=8G --time=05:00:00 --output=%x-%j.out ~/wf/runs/p2a/nca_drones/cluster/jobs/p2a_old_recipe.sh p2a_shared_s0 0 0.5'
```

The trainer appends to its log, so a name is used once; the script refuses a name whose log exists. Each job
needs 2 CPU cores and, on the machine that produced the stored runs, took about 8 seconds per round, so about
2.3 hours for 900 rounds plus about half an hour for the evaluation. `runs/` is ignored by git; what is kept
goes to `results/p2a/` in the run folder, comes back to the Mac and is committed there.

Gate: the three-seed mean of the closed-loop rate on the 2,000 never-posed tasks lies within 5 points of the
stored run, 75.4% with shared noise and 65.5% without. Report also whether seed 0 reproduces the stored run
exactly. The three seeds give the first estimate of run-to-run variation, which the 8x8 work never had.

### 12.2 The 8x8 task in the new simulator

The old task, expressed with `wf/sim.py`, so that the new core is measured on the same tasks:

| Old task (`env.py`) | New simulator |
|---|---|
| 8x8 grid, no obstacles | `H = 8, W = 8`, no obstacle; every cell is a road cell, so the recharger may go anywhere |
| Seeker with a battery of 5; recharger without one | A capturer and a recharger; `capacity = (5, 0, 0)` |
| An empty battery ends the episode | `empty_battery = "removed"` |
| A target that does not move and is known from the start | `p_move = 0.0`; `sense = (14, 14, 14)`, so every cell is always sensed |
| Reward when the seeker stands on the target | Capture |
| Horizon 20 | A leg limit of 20 steps; `T = 20` |
| Tasks: three distinct random cells, filtered to be solvable; success is measured on tasks whose target is more than 5 cells from the seeker | The same tasks, drawn with the old sampler (`env.sample_tasks`) by a small benchmark-side script |

The target is placed by setting the state's target cell directly, as the test helper `world` in
`wf/test_sim.py` does. Evaluation sets are the old ones: the 200-task test set (seed 12345), the validation
set (seed 999, the first 100 for choosing the checkpoint) and the 2,000 new tasks (seeds 2001 to 2010) with
every task removed that the run itself posed in training. The test and validation lists are written to a
file once. The list of 2,000 new tasks cannot be written in advance: `analyze.fresh_tasks` replaces every
task that the run posed by later draws, so the list is made per run after training, from that run's record
of posed tasks (the function reads the record from `analyze.load.posed`, which the script has to set). The
trainer therefore records the tasks it poses, with the key of `train.task_key`, and it never poses a task of
the validation or test sets, as the old trainer did not.

Nothing is hidden in this task, so every world is the same. The general search code is used with
W1 = W2 = W3 = 1, not a special case: the confirmation test then has a standard error of zero and reduces to
"strictly better than the greedy rule", and the weight of a leg is its own return. Per round: 64 tasks and 64
strategies per task, as in the old recipe.

### 12.3 What has to be written

All of it is the learner, in `wf/`. Each file comes with tests that run on the cluster's CPUs before anything
gets GPU time. Shapes are given with the batch first.

**`wf/lattice.py`: the rule and the window.**

* Constants: 3 past slices, 16 future slices, 20 in all; the present is slice index 3, and future slice j is
  index 3 + j. Tensors are [B, channels, 20, H, W].
* `Rule(n_in=68, C=32, hid=128, n_out=16)` with five layers: the input convolution 3x3x3 from 68 to 128; the
  state convolution 3x3x3 from 32 to 128 without bias; a 1x1x1 layer from 128 to 128; a 1x1x1 layer from 128
  to 32 that starts at zero, so that an untrained rule leaves the state at zero and its tubes are flat; the
  output layer 1x1x1 from 32 to 16. That is 366,896 weights.
* `encode(X)` gives c, once per step. `iterate(S, c, k, noise=0.0)` applies the update of section 7.1 k
  times. `tubes(S)` gives Y; it is a 1x1x1 layer, so it can be applied to the slices that are needed only.
* `bound(S)` divides each voxel's state vector by max(1, its root-mean-square over channels).
* `slide(S, X, map_slice)` drops the oldest slice and appends one with zero state and the map alone.
* Switches for the arms of 12.4: no bound; state reset to zero with K0 iterations at every step.
* Tests: shapes; `bound` leaves a vector with root-mean-square below 1 unchanged and scales a larger one to
  exactly 1; a new rule gives the same score to every cell; `slide` moves slice k to k - 1; running k and
  then k' iterations equals running k + k'.
* When this file exists, time the real rule on an H100 (search mode, the best format of phase 0) and replace
  the value D of section 10.

**`wf/inputs.py`: from an observation to the input channels.**

* `draw_codes(B, N, gen)`: for every episode, N distinct channel numbers out of 16.
* `present_slice(cfg, st, obs, codes)`: [B, 68, H, W]. Obstacle and road come from the state, everything else
  from `sim.observe`. Channels: 0 obstacle, 1 road, 2 sensed, 3 target found, 4 to 19 capturer codes, 20 to
  35 recharger codes, 36 to 51 observer codes, 52 to 67 battery level times code. Drones on one cell add up.
  A slot without a drone adds nothing.
* Switches for the arms: battery channels zeroed; one channel per kind instead of codes.
* Tests: a hand-made state with two drones on one cell, an empty drone and an unused slot gives exactly the
  expected channels.

**`wf/agent.py`: acting.**

* `start(rule, cfg, st, codes, K0)` builds S and X, clamps the first observation and runs K0 iterations.
* `scores(rule, cfg, st, lattice)`: [B, N, 5], the tube value of each drone's own channel in the next slice at
  the cell each action leads to (`sim.peek`). An impossible move leads to the drone's own cell and so scores
  like staying.
* `advance(rule, cfg, st_next, lattice, k_s)`: slide, clamp, k_s iterations.
* `run_greedy(rule, cfg, st, steps)`: the closed loop with one lattice, no noise and no search, until capture
  or the step limit. The trainer uses it on validation episodes; `bench/evaluate.py` uses it for everything
  else.
* Tests: with tubes set by hand along a path, a drone follows the path; two drones with different codes
  follow two different paths.

**`wf/explore.py`: one change.** Today `choose` draws fresh noise at every call, so a strategy cannot be run
again. Search needs to run the same strategy in other worlds (confirm, harvest, the sensing test), so the
noise of a strategy has to be drawn once per cut, for all steps of the leg, and indexed by step. Clones of
one strategy in different worlds then share its noise, and their excursions coincide because excursions do
not depend on the scores.

**`wf/search.py`: screen, confirm, harvest at a cut.**

* Input: the simulator state and the lattice at the cut, for a group of trunks with the same grid size.
* Screen: `sim.branch(cfg, st, M, W1, seed)` puts clone (m, w) of trunk b at index (b M + m) W1 + w; then
  `draw_target`. Strategy 0 is the greedy one. Clones run until capture or the leg limit. The value of a
  clone is 0.97 to the power of its steps to capture, or 0.
* Confirm: the E best strategies by mean value and the greedy one, in W2 worlds from a different seed. The
  test of section 7.4, with the standard error taken as zero when W2 = 1 and "mean of d above zero" required
  in every case.
* Harvest: the kept strategy in W3 worlds from a third seed. Per world it records the observations and
  actions of every step, the capture step and the return.
* Inside a leg the simulator's own respawn and relocation stay off. The trainer ends a leg at the capture or
  at the limit, removes the target, and takes the next cut.
* Continuation of the trunk by the rule of section 7.4.
* The record of a leg holds what the learner needs without running the simulator again: the map, the kinds
  and codes, for every step of the trunk up to the cut and of the leg the observation (cells, battery levels,
  the sensed mask, the target's cell if found), and for the leg the credits and weights. A record is a few
  tens of kilobytes. (The design page proposed to store seeds and actions and to regenerate observations;
  storing observations is simpler and equally exact.)
* A switch that also stores the greedy rule's captures (arm 5 of section 12.4).
* Memory: one clone at 24x24 holds roughly 10 to 15 MB (state, inputs, the encoded inputs and the hidden
  layer; an estimate). The three stages run one after the other. The largest, the screen, holds 32 x 128 =
  4,096 clones for 32 trunks, about 40 to 60 GB before temporaries. Whether that fits on one 80 GB card is
  open, so the code processes a cut for a chunk of trunks at a time, and the chunk size is set from the
  measured peak memory.
* The standing audit of selection (section 14.3) is part of this file.
* Tests: with scores fixed by hand, the numbers of clones and their indices are as stated; identical
  strategies give d = 0 and are not kept; a hand-made table of values passes and fails the test as computed
  by hand.

**`wf/credit.py`: who earned the capture.**

* `order_shares`: the first test of section 7.5. Simulator only. For each harvest world, each of R orders
  (an order and its reverse) and each j from 0 to n, K replays with the same random draws for every j.
  With n = 10, K = 8, R = 2 and 8 worlds that is 1,408 simulator clones per kept leg.
* `sensing_shares`: the second test, lattice runs with one drone's `sense_on` switched off.
* `combine`: the mean share per drone over orders and worlds, its standard error over worlds, and the
  threshold. The starting multiple is 2 standard errors, to be set by the ghost audit. With one harvest
  world, as on the 8x8 task, there is no standard error over worlds: no threshold is applied and the credit
  is the mean share over orders and replays, as in the old recipe.
* A switch for the old form (one drone at random, no orders), which is an arm.
* Tests: the two hand-worked cases of the old `test_sim.py`, expressed in the new simulator: a recharger gets
  no credit when the seeker reached the target on one battery, and full credit goes to a seeker whose random
  replacement cannot reach a far target. On teams of up to five, the sampled shares agree with the exact
  ones, obtained by averaging over every order.

**`wf/learn.py`: the loss of section 7.6.**

* Legs are grouped by grid size; a group is one tensor batch. Padding different sizes into one batch is not
  allowed, because zero padding is how the rule sees the edge of the map.
* Per leg: replay the trunk up to the cut without gradient and with redrawn iteration counts; then, step by
  step through the leg, clamp, iterate with noise 0.02, compute the cross-entropy for j = 1 to 16, and after
  every second step back-propagate, update and detach.
* Tests: a drone with credit 0 receives no gradient; fitting one leg repeatedly makes the greedy rule
  reproduce it.

**`wf/train.py`: the rounds.** Configuration from a JSON file named in the card. It writes `status.json`
every minute; `log.jsonl` per round with those items of section 14.4 that use no task notion (the shares of
cuts with a capture and with a confirmed strategy, legs per trunk, credit per kind, the size of the state,
the loss); the harvested leg records, from which the benchmark-side script computes the other items
afterwards; `ckpt.pt` by write-and-rename with the weights, the optimiser, the round, the random generators
and the record of posed tasks; and `best.pt`, chosen with `run_greedy` on the validation episodes.
`--resume auto` continues from `ckpt.pt` if it exists. For one GPU, search and learning alternate in one
process. For several GPUs they become separate processes, one search worker per GPU and one learner: workers
read the weights from a file that the learner replaces by renaming, and write harvested legs as files.

**`bench/evaluate.py`: evaluation, on the benchmark side.** It may import `wf/`; the reverse stays forbidden.
One lattice per episode, no noise, no search, through `wf.agent.run_greedy`. It takes a checkpoint
and a stored set, applies an optional control, and writes counts with Wilson intervals, and for two models
on the same episodes the paired difference with its interval.

**`wf/check_purity.py`: the contract as a test.** It fails if any learner file (`lattice.py`, `inputs.py`,
`agent.py`, `explore.py`, `search.py`, `credit.py`, `learn.py`, `train.py`) contains one of the words
distance, meeting, escort, pincer, sweep, low battery, sighting or role, or if any file in `wf/` imports
from `bench/`. The environment files `sim.py` and `maps.py` are not scanned for the words, because the rules
of the world are stated in them. It is part of the test suite.

**An end-to-end smoke test:** three rounds on 8x8 on a CPU run without error, write a status file and a
checkpoint, and continue from the checkpoint.

### 12.4 The runs

Three seeds each, 900 rounds, a take-off test (section 7.7) before the first full run. The checkpoint is
chosen on 100 validation tasks every 50 rounds and evaluated once on the test set and on the 2,000
never-posed tasks.

| Run | What differs from the new core |
|---|---|
| The new core | Nothing: window with kept and bounded state, codes, battery input, improvements only, credit over two orders, no shared noise |
| Arm 1 | State without the bound |
| Arm 2 | State reset and regrown with K0 iterations at every step |
| Arm 3 | One channel per kind instead of codes |
| Arm 4 | Battery input zeroed, as in the old task |
| Arm 5 | The greedy rule's captures are stored too |
| Arm 6 | Credit in the old form: one drone at random, no orders |
| Strand run (form to be decided, O12) | Trained on tasks with one to five seekers, one recharger and a common target that stays after a capture; the gate of section 14.1 is measured on it |

One change per run: an arm never differs from the new core in more than its one switch. The strand run uses
placements in which every seeker needs a recharge and all drones can gather within one battery life, the
filter of the earlier probe (`probe_seekers` in `probe_extensions.py`). The same tasks are also given to the
three checkpoints of the new core, which never saw more than one seeker; that number is reported beside the
gate, as the comparison with the 34% of the old rule.

### 12.5 Gates of phase 2

* The new core reaches at least 75% on the never-posed tasks, as the mean over three seeds. It has no shared
  noise, so the fair comparison in the old recipe is 65.5%, and 75% asks for the level that the old recipe
  needed shared noise for.
* The iteration gate (section 14.1), measured on the three checkpoints of the new core.
* The strand gate, in the form that O12 decides, measured on the strand run of section 12.4.
* **Kill criterion:** no arm passes the iteration gate. Then the question of how a kept state can be made
  stable goes back to design before anything is scaled up.

## 13. Phases 3 to 6

Phases 2 to 5 are development steps. The claim in phase 6 is made by one recipe trained from scratch. A phase
that fails its gate is analysed on trajectories and then repeated or stopped; nothing outside the dials of
section 7.7 is added to the learner to get past a gate. GPU-hours are estimates at 900 rounds per run.

| Phase | Goal | Gate | Runs | GPU-hours |
|---|---|---|---|---|
| 0 Arrival | Replace every estimate | Throughput measured; stored checkpoint reproduced | none | under 5 |
| 1 Benchmark | A frozen benchmark with shown properties, no learner | The table of section 11.9; parameters, seeds and hashes committed | none | about 10 |
| 2 Core on 8x8 | The new lattice loses nothing on the measured task | Section 12.5 | about 20, and 6 of the old code | 25 to 95 |
| 3 Pursuit | Coordinated capture of a visible fleeing target on 12x12 | At least 60% with three capturers | about 12 | 70 to 300 |
| 4 Hidden target | Scouting through the value of information, 16x16 | Hidden reaches three quarters of visible | about 9 | 140 to 560 |
| 5 Full scenario | 24x24, ten drones, from scratch | At least 80% on validation | 3 | 105 to 420 |
| 6 Certification | A number that survives an audit | The bar on the test set | 5 | 175 to 700 |

### 13.1 Phase 3: pursuit

**Setting.** 12x12 maps of the training families. The target flees with a sight of two cells and is always
visible to the team: the sensing radius of every kind is set to 22, the largest distance on the grid. Teams of
2 to 5 capturers and 0 to 3 rechargers, no observers. Batteries as frozen in phase 1. Trunks of 96 steps, a
leg limit of 48, respawn after a capture.

**Evaluation of this phase** (a proposal, to be fixed in the phase's first card): 200 maps without dead ends,
2 episodes each, three capturers and two rechargers, start cells and battery levels as in training, 64 steps,
greedy readout. The recharge certificate cannot be required here: the largest distance on a 12x12 grid is
22, so it can never hold for a capturer whose charge is 11 or 12. The phase reports instead in what share of
captured episodes a capturer was refilled before the capture.

**Runs.** The main recipe with three seeds. Arms with one seed each: two commit groups and as many groups as
drones, at equal iterations; block redraws of one to three drones switched on; no selection (the best
screened strategy is kept without confirmation); credit in the old form; credit by closed-loop reruns with
drones removed. An arm that differs from the main run is repeated with two more seeds before it is believed.
Not training runs: the first three audits of section 14.3 (selection, ghost, exact shares). If O13 is
approved, the baselines run here.

**Gate.** At least 60% with three capturers. One capturer alone: exactly 0 on maps without dead ends. No kind
of drone with zero mean credit. The ghost drone clears the credit threshold in at most 2% of legs.
**Kill criterion:** under 30% after 600 rounds with every lever that needs no approval (more clones, smaller
grids in the mixture).

### 13.2 Phase 4: hidden target

**Setting.** 16x16 maps, the real sensing radii, up to six drones with observers. The proposal for the
evaluation team is three capturers, one recharger and two observers, 96 steps.

**Runs, about nine.** The same recipe with the target visible, hidden with observers, and hidden without
observers. Observers flying with their sensing withheld. The cut taken before the target is drawn against a
cut taken after the draw, where every clone meets the same hidden cell. More worlds per stage. A past window
of 3, 8 and 16 slices, and a
back-propagation span of 2, 8 and 16 steps. Beside them, on the benchmark side: a readout of the target's
cell fitted afterwards on saved states, to see whether the state still holds where the target was once the
sighting has left the window.

Two things change in the learner in this phase and need their decisions first: the weights over worlds that
the team cannot tell apart (O11), and the sensing test of credit, which becomes a weight once a ghost drone
passes it in at most 2% of legs.

**Gate.** The rate with a hidden target reaches three quarters of the rate with a visible one. Withholding
the observers' sensing costs at least 10 points. Kept strategies stay ahead of the greedy rule on a further
set of worlds. **Kill criterion:** no observer effect. The question then goes back to the PI, and no scouting
bonus is added.

### 13.3 Phase 5: the full scenario

**Setting.** The frozen benchmark: 24x24, ten drones, trained from scratch on the mixture of grid sides 12,
16 and 24 and of team sizes.

**Runs.** Three seeds of the full recipe. The reach gate. Every control of section 14.2. A review of 50
failed trajectories per run, written into the card.

**Gate.** At least 80% on validation, as a point estimate. The reach gate is passed. Frozen rechargers: at
most 5%. **Kill criterion:** under 40% after three full runs while the rate with a visible target is at
least 90%. The failed gate is reported and no heuristic is added.

### 13.4 Phase 6: certification

The protocol and the hashes of the sets are committed first. Then: five seeds from scratch; the 2,000 test
episodes; the held-out map family and the hand-drawn maps; other flee rules; every control on identical
seeds; sweeps of the compute dials; an independent second implementation of simulator and readout, written
from the documents, that reproduces the stored numbers. **Gate:** the bar of section 5.3 on the test set,
with every control behaving as stated.

## 14. Gates, controls and audits

### 14.1 Gates

| Gate | What must hold | The 8x8 rule today |
|---|---|---|
| More computation never hurts (the iteration gate) | K0 and k_s are multiplied together by 2, 4 and 8. On 2,000 held-out tasks the lower 95% bound of the paired difference in closed-loop success stays above -2 points. A longer window and more clones are checked by retraining at the end of phases 3 and 5 | Fails: open loop, at twice the iterations the two checkpoints lost 5 and 18 points |
| One drone, one strand | On 8x8 with one recharger and two, three or five seekers, all seekers arrive in at least 60% of tasks (the exact form is decision O12) | 34% for two seekers in one channel, never trained that way |
| Tubes, not arrows | With everything visible, clamping and iterating only every fourth step, while drones follow the tubes already grown, costs at most 5 points. Also reported: the share of steps in which a drone's planned cell two steps ahead changes although no new fact arrived | Flying the whole 8x8 dream without replanning cost 2 to 3 points |
| Reach | Capture rate falls by at most 10 points between episodes whose first sighting is within 10 cells of the nearest capturer and those where it is 30 to 40 cells away, and doubling K0 does not improve it | Not applicable |
| Search does not keep luck | The selection audit of 14.3 passes in at most 5% of cuts | Not applicable |
| Credit does not pay bystanders | A ghost drone clears the credit threshold in at most 2% of legs | Not applicable |
| No domain concept in the learner | `wf/check_purity.py` passes | Passed in the audited path |
| The evaluator is independent | A second implementation reproduces the stored numbers | Done once, for the first passing checkpoint |

Statistics. A single rate is reported with its Wilson 95% interval. Two models on the same episodes are
compared by the paired difference: per episode, success of one minus success of the other; the mean, plus or
minus 1.96 times the standard deviation divided by the square root of the number of episodes. The final
number uses a bootstrap that resamples training seeds first and maps second.

### 14.2 Controls

| Control | Certifies | Expected | How it is made |
|---|---|---|---|
| Rechargers removed | A recharge is required | Exactly 0. Any capture is a bug in the benchmark | Their slots are marked as not existing |
| Rechargers present, refills switched off | The same, with the team's input unchanged | Exactly 0 | A simulator setting, to be added in phase 1 |
| Rechargers frozen where they start | Recharging must be organised, not merely available | At most 5% | The evaluator replaces their actions by "stay" |
| One capturer alone | Coordination is required | Exactly 0 | All other slots are marked as not existing |
| One capturer with all rechargers and observers | How much capture comes from herding by drones that cannot capture | Reported | The other capturers are removed |
| Obstacle channel zeroed in the input | Obstacles must be flown around | At most half the headline rate | A switch in the evaluator's input |
| Observers removed; observers with their sensing withheld | Scouting matters, and through information | A drop whose interval excludes zero | Slots removed; `sense_on` switched off |
| Scripted team that always sees the target | Capture is possible on the evaluation set | At least 90%, published as a reference | `bench/scripted.py` |
| Target always visible | The price of partial observability | Reported | Sensing radius of every kind set to H + W |
| Batteries unlimited | The price of energy | Reported | `capacity = (0, 0, 0)` |
| A readout of the target's cell, fitted afterwards on saved states | The state still holds where the target was after the sighting has left the window | Better than "where it was last seen", eight steps later | Benchmark side, a separate process |
| Random team | The task is not solved by chance | At most 1% | Uniformly random actions |

Controls that apply to a team are first run with the scripted teams, to show that the benchmark has the
property, and later with every model that is reported. If a control fails to behave as stated, the learner is
never adjusted to pass it.

### 14.3 Audits

* **Selection audit.** In one cut of sixteen, screening and confirmation are run a second time with M + 1
  strategies that share one setting of temperature, excursion rate and hold and differ only in their noise.
  One of them stands in for the greedy rule. They are equal in expectation, so every pass is a false pass.
  Required: at most 5% of audited cuts. If it fails: more confirmation worlds and fewer strategies at equal
  cost.
* **Ghost audit.** In two of the 32 trunks of a round, one extra slot holds a ghost drone. The simulator
  ignores it; the lattice sees it like any other drone. Its credit should be zero. Required: it clears the
  threshold in at most 2% of legs; otherwise the threshold multiple and the number of replays go up. Legs of
  these two trunks are not learned from.
* **Exact shares.** On teams of up to five drones, the credit shares from two orders are compared with the
  exact ones, obtained by averaging over every order of the drones. The mean absolute difference is reported.
* **Independent evaluator.** In phase 6, a second implementation written from the documents.

### 14.4 What every training run logs

Each of these is the first sign of a predicted failure (section 16). Several of them use notions that the
learner must not contain, such as the distance to the nearest capturer. They are therefore computed from the
stored leg records by a script on the benchmark side, not inside the learner's files.

* The share of cuts in which any strategy captures, and the share in which one is confirmed.
* The distance from the new target to the nearest capturer, in harvested legs against all legs.
* Steps survived per capturer, and legs per trunk.
* Credit per kind of drone.
* Recharges per captured leg.
* How many drones the target saw at the moment of capture, and whether the capturing drone was moving
  towards it.
* The size of the state: its mean root-mean-square per voxel and the share of voxels at the bound.
* How often two drones of one kind share a cell.
* The share of "stay" actions per kind.

## 15. Cost model and budget

Every figure here is hand arithmetic on a throughput that has not been measured. Phase 0 replaces it.

* The window at 24x24: 20 slices of 576 cells, 11,520 voxels. One iteration costs 131,072 multiply-adds per
  voxel. The input convolution runs once per step and costs as much as 1.8 iterations.
* One clone-step: 5.8 iterations on 11,520 voxels, about 67,000 voxel-iterations.
* One cut: 216 clones of at most 48 steps, about 10,400 clone-steps or 6.9 x 10^8 voxel-iterations. Clones
  start from the trunk's lattice, so K0 is paid once per trunk.
* One round: search at 64 cuts, 4.4 x 10^10; the sensing test, 11 reruns in each of 8 harvest worlds for
  every kept strategy, up to 1.8 x 10^10; rebuilding trunks and learning, about 0.3 x 10^10. Together about
  6 x 10^10 voxel-iterations when half of the cuts keep a strategy.
* The simulator is asked for about 3 million team-steps per round, of which about 2.2 million are the credit
  replays, and about 2.7 billion per run of 900 rounds. Simulator steps are cheap next to lattice iterations.
* With a throughput of D voxel-iterations per second on one H100: seconds per round = 6 x 10^10 / D, and
  GPU-hours per 900-round run = 1.5 x 10^10 / D. The design assumed D between 110 and 440 million, which
  gives two to nine minutes per round and **34 to 136 GPU-hours for a full-scale run**, which the design and
  the budgets of section 13 round to 35 to 140. A 12x12 run costs about a quarter of that or less, a 16x16
  run a little under half.
* The 900 rounds are carried over from the 8x8 runs. Under a hidden target the gain per round is smaller, and
  the number of rounds may be several times higher. Budgets are set again after the throughput benchmark and
  a 25-round take-off test.

The roadmap holds about 49 training runs of the new system (20, 12, 9, 3 and 5 in phases 2 to 6) and comes
to roughly 500 to 2,100 GPU-hours if 900 rounds per run suffice. Evaluation, controls and sweeps add about a
tenth. The plan assumed seven H100s, a number the PI mentioned in a question and never confirmed; on seven
that is 3 to 12 days of computing, spread over several weeks of experiments and analysis. Every run is sized
to finish within hours.

## 16. Predicted failures

| Failure | First sign | Response |
|---|---|---|
| No reward for an untrained team at full scale | Fewer than 10 captured legs per 4,096 untrained clones in phase 1 | More clones; smaller grids in the mixture. After that the levers of O14, which need approval |
| The team learns to wait for the target to walk into it | Captured legs in which the capturing drone never moved towards the target | Count such legs before trusting any early curve; nothing greedy is stored |
| Capturers chase one by one and never close from two sides | Harvested legs start within two cells of a capturer and stay there over rounds; one drone in the target's sight at capture | Block redraws and more strategies; two commit groups |
| Search keeps luck | The selection audit fails; kept strategies do not stay ahead on a further set of worlds | More confirmation worlds, fewer strategies, at equal cost |
| The kept state drifts on long trunks | State pinned at its bound; capture rate falls with the number of the leg; plans change without new facts | More update noise; iteration counts redrawn more widely |
| Noisy credit brings back "stand still" | The share of stay actions per kind rises; the ghost drone clears the threshold too often | More replays and orders; a higher threshold |
| Observers add nothing, or only herd | Withholding their sensing changes nothing | Check the value of information in the benchmark first; more harvest worlds. Never a coverage bonus |
| What was seen is forgotten once it leaves the window | Capturers return to cells just found empty; the readout loses the target a few steps after the sighting | Longer back-propagation span, longer past window, wider state |
| The reach gate fails | A sharp drop of capture rate with the distance of the first sighting | More iterations per step. Then coarser lattice levels, which need approval |
| Throughput far below the estimate | The arrival benchmark | A narrower rule and smaller dials; every budget rescaled before phase 2 |
| The long evaluation episode is never assembled from short legs | Training legs are won, evaluation episodes are not; failures happen after the first recharge | This is the main risk of the programme. Longer legs and trunks first; it is a question for the PI before anything else is tried |

## 17. Proposed but not decided

### 17.1 Baselines (decision O13)

No baseline has ever been run in this project, so nothing can be said today about how the method compares
with standard multi-agent reinforcement learning. The proposal from the discussion of 27 November: recurrent
MAPPO and QMIX on the benchmarks of phases 3 and 5, with the same pooled information as the lattice and the
same GPU-hours as the phase's main runs, reported beside our number with the simulator steps each method
used. It costs about as much again as those phases' main runs and changes nothing in the learner.

### 17.2 Making search cheaper

The method spends simulator steps and lattice iterations freely by design. These levers lower the cost
without adding knowledge of the task. They come after the first run that works, not before, and each is
accepted only if a paired comparison shows the same result at lower cost. The measure is confirmed
improvements per million clone-steps.

1. Stop a screening clone as soon as its strategy can no longer reach the best E even if it captured at the
   next step. This changes nothing in what is kept.
2. Keep harvested legs for several rounds instead of using them only in the round that found them.
3. Let the clones of one strategy share one lattice until their observations differ. With a hidden target
   all worlds look alike until the first sighting.
4. Compute credit on a sample of the kept legs.
5. Give more strategies to cuts where screening found something rare and fewer to cuts the greedy rule
   already wins.
6. Draw temperature, excursion rate and hold more often from the ranges that produced confirmed
   improvements.

## 18. Known gaps, and where this file departs from the design page

1. **The throughput benchmark times a stand-in,** not the rule of design version 2. Its figure is converted
   with the factor 1.84 (section 10, step 5). The card of phase 0 says the same.
2. **The strand gate** is written in the design page with "seekers that each have their own target". The
   simulator has one target, and no input says whose target a cell is. Decision O12 proposes the form of the
   probe that produced the 34%: a common target that stays after a capture. That needs one more simulator
   setting. It is a weaker test than separate targets; the sharper evidence for separate strands is the share
   of captures in phase 3 with two drones in the target's sight.
3. **Leg records store observations** instead of seeds and actions (section 12.3).
4. **The evaluations of phases 3 and 4** are not defined in the design page. Sections 13.1 and 13.2 propose
   them. The recharge certificate cannot hold on 12x12.
5. **Diagnostic logs that use task notions** are computed on the benchmark side (section 14.4); otherwise
   the purity check would rightly fail.
6. **The selection audit** is defined here with strategies that are equal in expectation. Literally
   identical strategies would always tie and test nothing.
7. **Legs of trunks with a ghost drone** are not learned from.
8. **One map family exists** of the three planned for training; the held-out family and the hand-drawn maps
   do not exist yet.
9. **Missing pieces of phase 1:** the text rendering, the comparison of CPU and GPU, a setting for refills
   switched off, removal of dead ends, the evaluation generator, the scripted teams.
10. **`wf/explore.py` draws fresh noise at every call** and has to become replayable per strategy before
    search is written.
11. **The old trainer does not save its replay buffer,** so a resumed run is not identical to an
    uninterrupted one. The stored run `v10b` was a continuation of that kind.
12. **The tolerance of the reproduction check** in phase 0 (within 10 of 2,000) is a proposal of this file.
13. **Seven H100s** is an assumption (section 15).
14. **Literature.** The research sweep behind the design could not open primary sources from its
    environment. Its claims are low-confidence, and no number in the design relies on them.
15. **The design page is rebuilt on the cluster** from now on (section 9.7). The page itself still says that
    the assistant's container builds it.
16. **Where the session runs.** The design page records that the Mac folder is linked to the kickoff session.
    That link never came about: the list of folders offered to the session stayed empty. Section 8.2
    therefore recommends a session that runs on the Mac, which the PI had also asked about.
17. **The runs of phase 2.** The design page speaks of one change per run and four arms. Section 12.4 builds
    the new core with all changes and takes one away per arm, with six arms and a strand run. No run isolates
    staged search, because on a task where nothing is hidden it reduces to "improvements only", which is
    arm 5.
18. **One process on one GPU.** The design page separates search workers and learner into processes from the
    start. Section 12.3 alternates them in one process on a single GPU and separates them for several.
19. **The evaluator sits on the benchmark side** (`bench/evaluate.py`), because it applies the controls. The
    learner keeps only the plain greedy loop that it needs to choose a checkpoint on validation episodes.
20. **The connectivity check on training maps** is named in rule 6 and in O2 as the one benchmark-type check
    that touches training. The design page lists connectivity checks among the benchmark-side tools without
    saying so.
21. **Starting values that this file adds** and that the design page does not give: a credit threshold of 2
    standard errors; the selection audit in one cut of sixteen; a ghost drone in two of 32 trunks; the seed
    blocks of section 11.8; the evaluation settings of phases 3 and 4; the definition of the take-off test.
    All are proposals to be fixed in the card of the first experiment that uses them.
22. **Code reaches the cluster from the Mac only.** The design page (section 14) has the cluster check out
    commits, and commit and push its results. By the PI's rule of 27 November all code and all git go through
    the Mac: the cluster runs no git, the Mac sends the files of a commit over ssh, and results come back to
    the Mac (sections 4.3 and 9.2).

## 19. History: the 8x8 result

Details and all tables are in `README.md`.

**The task.** An 8x8 grid, 20 steps, two drones. A seeker has to reach a target but its battery lasts 5
steps; a recharger refills it whenever the two share a cell. Reward 1 when the seeker stands on the target,
nothing else. Success is measured on the tasks the seeker cannot solve alone.

**The model.** One local rule shared by all 21 x 8 x 8 voxels, iterated 32 times: a 3x3x3 convolution from 16
state and 3 seed channels to 64, two 1x1x1 layers, a residual update, per-voxel normalisation, and two tube
channels, one per drone type. 38,130 weights. A drone moves to the reachable cell with the highest tube value
in the next slice.

**The method.** Per round: 64 tasks; 64 joint rollouts per task that follow the tubes with noise; the
earliest rewarded rollout is kept; credit by replaying it with one drone's actions replaced by random ones;
imitation by cross-entropy per time slice. About 8 seconds per round on two CPU threads, 2.3 hours for 900
rounds.

**The result.** The audited run `v10`: 164 of 200 held-out tasks (82.0%, interval 76.1 to 86.7) with the
closed-loop readout and 158 of 200 with the open-loop dream; 1,507 of 2,000 tasks never posed in training
(75.4%, interval 73.4 to 77.2). The stated bar of 75% on the 200-task set was met; the stronger requirement,
a lower bound above 75% on the 2,000 tasks, was not. One training seed per recipe.

**Measured facts the new design leans on.**

| Fact | Number |
|---|---|
| Without shared exploration noise (`v11_noshared`) | 135 of 200; 1,309 of 2,000 = 65.5% |
| An untrained rule finds a rewarded rollout among 64 | On 1.2% of tasks with independent noise, on 9.1% with half of the noise shared |
| 256 independent rollouts against 64 with shared noise, untrained | 7.0% against 9.0% |
| More iterations than in training | At one and a half times, `v10` lost 8.5 points and the earlier run nothing; at twice, the earlier run lost 5 points and `v10` 18 |
| Regrowing the lattice at every step instead of flying one dream | 22 points gained where the recharger starts far away, 5 lost where it starts near |
| Two seekers placed in one tube channel, never trained that way | Both arrive in 34% of cases; each alone in 81.5% |
| Without credit | The recharger learned to stand still. It is irrelevant in 56% of tasks |
| The same rule against a target that flees at full speed, never trained for it | 0% |
| The same rule on larger grids, never trained for them | 48 to 53% on 12x12, 22 to 25% on 16x16 |
| Independent second implementation of simulator and readouts | Reproduced the stored numbers of the first passing checkpoint exactly |

**What was learned about the method.** Sparse reward does not say whose behaviour earned it, and asking the
simulator is the cheap fix. Where the exploration noise lives decides whether cooperation is ever sampled.
A plan grown once is right about the route and imprecise about time, and feeding the realised past back in
repairs most of that.
