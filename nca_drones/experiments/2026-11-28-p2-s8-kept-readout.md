# Experiment card: 2026-11-28-p2-s8-kept-readout

## Before the run

* **Phase and gate it serves:** phase 2, step 8 of the chain in `PROGRAMME.md` section 12.4: part of the look at a
  step that was not kept (step 8 scored 19.9% where step 7 scored 94.2%). No training.
* **What was seen:** the run of step 8 did not learn for 450 rounds and then began to (the rule alone solved 30% of
  the easy tasks at round 500 and 90% at round 900). The iteration gate, read on 28 November, shows that step 7's
  rule holds at twice its 32 iterations (96.15%) and breaks at four times (41.55%), while step 6's rule loses 5
  points there. A kept state at 4 iterations per step reaches 72 iterations after ten steps and 112 after twenty.
  It is not known whether step 8 fails because a state cannot be flown kept, or because learning with a kept
  state, as step 8 does it, is slow.
* **Hypothesis (one sentence):** step 7's rule, which was trained with regrowth only, solves the never-posed tasks
  with a kept state at 4 iterations per step almost as well as with regrowth: it loses at most 5 points.
* **What would support it:** on the checkpoint of step 7 (`p2s7_bound_s0b`), the paired difference in closed-loop
  success between the kept readout at 4 iterations per step and the run's own evaluation, on the same 2,000
  never-posed tasks (bf16), has its whole 95% interval above -5 points.
* **What would refute it:** the whole interval below -5 points. An interval that holds -5 is inconclusive.
* **The one thing that changes:** the evaluation flies the same checkpoint with a kept state
  (`analyze.py --kept 1 --ks 4`, new in this commit) in place of regrowing at every step. Weights, tasks and
  everything else are those of the run's own evaluation.
* **Also reported, with no threshold:** the same checkpoint at 8 and 16 iterations per step; the checkpoint of step
  6 (`p2s6_closed_s0`, the measured update without the bound) at 4, 8 and 16.
* **What the result decides:** where step 8 fails. If a rule that never trained with a kept state flies with one
  at little cost, then keeping the state is not the difficulty and learning through the kept replay is. A version
  of the step that keeps the state in search and in evaluation, where 96 to 98% of a round's time goes, and learns
  by regrowth as step 7 does, would then exist; it changes what step 8 is, so whether it is run is the PI's
  decision. If the kept readout breaks the rule, a kept state has to be trained for, and the dials of the card
  `2026-11-27-p2-s8-kept-dials` are what is left inside the step.
* **Verification done on the cluster's CPUs first:** `test_sim.py`. The new switch is checked in the run itself: on
  the checkpoint of step 8, `--kept 1 --ks 4` is what the run was trained and evaluated with, so the paired
  comparison with its own evaluation must show no task lost and none gained.
* **How it runs:** one Docker container on an idle GPU, after the check on the host that nothing else holds it and
  with the guard inside, from `~/wf/runs/p2kr` filled from the Mac, with the three checkpoints and the runs' own
  evaluations copied in on the cluster: `bash cluster/jobs/p2_kept_readout.sh p2s7_bound_s0b 4 8 16`, the same for
  `p2s6_closed_s0`, and `bash cluster/jobs/p2_kept_readout.sh p2s8_kept_s0b 4`.
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2kr/COMMIT`.
* **Cluster job ids:** container `wf-p2kr`.   **Estimated GPU-hours:** under 0.2

## After the run

* **Result:**
* **Trajectories looked at:**
* **Supported, refuted or inconclusive:**
* **What was learned:**
* **Next step, and the card it becomes:**
* **Actual GPU-hours:**
