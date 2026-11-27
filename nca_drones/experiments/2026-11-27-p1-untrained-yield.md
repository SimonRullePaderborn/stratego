# Experiment card: 2026-11-27-p1-untrained-yield

## Before the run

* **Phase and gate it serves:** phase 1 (benchmark, no learner). It answers the first risk of the programme:
  does a team that follows nothing but its exploration noise ever capture the target?
* **Hypothesis (one sentence):** against the full target (sight of two cells), a noise-driven team captures in
  well under 1% of legs on 24x24 and noticeably more often on 12x12, and every such capture has at least two
  drones in the target's sight.
* **What would support it:** capture rate per leg of 48 steps, by grid side (12, 16, 24) and team, with Wilson
  intervals over at least 4,096 episodes per cell of the table. The count of captures with fewer than two drones
  in the target's sight is zero.
* **What would refute it:** (a) any capture with one drone in sight: then the hand proof of "coordination is
  required" or the simulator is wrong, and nothing else proceeds until that is understood. (b) Fewer than 10
  captured legs per 4,096 clones on 12x12 as well: then smaller grids do not make reward reachable, and the
  levers that need the PI's approval (a target that sometimes fails to move, or sees less far) have to be asked
  for, with this table as the evidence.
* **The one thing that changes** relative to which earlier run: first measurement on the new simulator.
* **Reference rows, clearly marked as not the PI's target:** target sight 0 and 1, to see how steep the cliff is.
  They are measurements of the environment, not training settings.
* **Verification done on the cluster's CPUs first:** `python3 -m wf.test_sim` passes; one episode of each table
  cell is printed step by step and read.
* **Commit:**            **Configuration file:** `wf/yield_table.py` defaults
* **Cluster job ids:**   **Estimated GPU-hours:** under 1

## After the run

* **Result:**
* **Trajectories looked at:**
* **Supported, refuted or inconclusive:**
* **What was learned:**
* **Next step, and the card it becomes:**
* **Actual GPU-hours:**
