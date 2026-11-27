# Experiment card: 2026-11-27-p2-s4-new-simulator

## Before the run

* **Phase and gate it serves:** phase 2, step 4 of the chain in `PROGRAMME.md` section 12.4.
* **Hypothesis (one sentence):** the new simulator `wf/sim.py`, behind an adapter with the old simulator's interface
  (`wf/task8.py`), plays the 8x8 task exactly as the old simulator does.
* **What would support it:** (a) `python3 -m wf.test_task8` passes: on four sets of 4,000 episodes of 20 steps
  (random actions against a target; a scripted pair with frequent captures and recharges; free flight; free flight
  with a battery that never runs out), positions, battery, done, success, time of termination, number of recharges,
  reward and the cells each action leads to are equal after every step. (b) Two training runs of 8 rounds with the
  flags of step 3 and the same seed, one with `--sim old` and one with `--sim wf`, write the same `log.jsonl` digit
  for digit, clock fields aside (`cluster/same_logs.py`), including the closed-loop validation at rounds 0, 4 and 7.
* **What would refute it:** any difference. Then the case is worked by hand from the rules to decide which of the two
  simulators is wrong; a question about the rule itself goes to the PI.
* **The one thing that changes** relative to step 3: `--sim wf`. Every simulator the trainer and the evaluation
  create (rollouts, credit replays, closed-loop evaluation) is the adapter on `wf/sim.py`: the seeker is a capturer
  with capacity 5, the recharger has no battery, every cell is a road cell, the target never acts, sensing covers
  the grid, and an empty drone is removed. The adapter adds the end of an episode, which `wf/sim.py` does not have.
* **Why no long run:** if (a) and (b) hold, a run with `--sim wf` does the same arithmetic as the run of step 3 and
  gives the same result; a statistical comparison would add nothing. The number reported for this step is the count
  of compared episodes and rounds.
* **Verification done on the cluster's CPUs first:** `test_sim.py` (old code with the switches of steps 1 and 2) and
  `python3 -m wf.test_sim` (the new simulator's own 17 tests, which passed on their first execution on 27 November).
* **How it runs:** CPU only, from `~/wf/runs/p2s4` filled from the Mac; the four commands above in short containers.
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2s4/COMMIT`.
* **Cluster job ids:** containers `wf-p2s4-*`.   **Estimated GPU-hours:** 0

## After the run

Run on 27 November 2026 from commit `edbf630`, sent to `~/wf/runs/p2s4` and checked file by file (103 of 103).

* **Result:** (a) `python3 -m wf.test_task8` passed on its first execution: the two simulators agree in every field
  after every step on 4 sets of 4,000 episodes of 20 steps; the random set holds 3,818 empty batteries, the scripted
  set 1,430 captures and 876 episodes with a recharge (`results/p2/p2s4_test_task8.out`). (b) Two runs of 8 rounds
  with seed 0, the flags of step 3 and 8 threads, one with `--sim old` and one with `--sim wf`, wrote logs that are
  identical in all 22 compared fields for all 8 rounds, clock fields aside (`results/p2/p2s4_eq_old.log.jsonl`,
  `p2s4_eq_wf.log.jsonl`, `p2s4_same_logs.out`). Before both: `test_sim.py` exit 0, and the new simulator's own 17
  tests passed.
* **Trajectories looked at:** not applicable
* **Supported, refuted or inconclusive:** supported, exactly: no field of any compared episode or round differs.
* **What was learned:** `wf/sim.py` plays the 8x8 task exactly as the old simulator does, so `--sim wf` changes no
  result and needs no training run of its own. Every later step runs on the new simulator. This verdict does not
  depend on steps 1 to 3, which were still training when it was reached.
* **Next step, and the card it becomes:** step 5, the trainer on a GPU
* **Actual GPU-hours:** 0 (a few minutes on 8 CPU cores)
