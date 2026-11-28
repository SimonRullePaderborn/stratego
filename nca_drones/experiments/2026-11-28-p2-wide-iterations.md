# Experiment card: 2026-11-28-p2-wide-iterations

## Before the run

* **Phase and gate it serves:** phase 2, the iteration gate of `PROGRAMME.md` section 14.1. This is not a step of
  the chain of section 12.4. It is a measurement for the PI's decision on how the chain goes on, now that every
  checkpoint that solves the task has failed the gate.
* **What was seen:** paired differences in closed-loop success at 2, 4 and 8 times the iterations: step 6 +0.15,
  -5.00 and -12.30 points; step 7 +1.95, -52.65 and -90.05; the one run with a kept state that learned (8
  iterations per step, seed 0) -20.25, -52.05 and -65.55. In all of these, learning redraws the iterations of
  growth between 28 and 36, as the measured recipe does. The design's learning (section 7.6) redraws them from 32
  to 128, and section 16 names a wider redraw as the answer to a state that drifts. No run has had it yet.
* **Hypothesis (one sentence):** a rule whose iterations of growth are redrawn from 32 to 128 in learning holds at
  more iterations: step 6's recipe with that redraw keeps its closed-loop success at 2 and at 4 times its 32
  iterations.
* **What would support it:** on every seed of this card, the lower 95% bound of the paired difference between 64
  and 32 iterations, and between 128 and 32, is above -2 points (bf16, the checkpoint's own 2,000 never-posed
  tasks).
* **What would refute it:** a lower bound at or below -2 points at 64 or at 128 iterations on any seed. Both counts
  lie inside what the rule saw in learning, so a failure there means the redraw does not deliver.
* **The one thing that changes** relative to step 6: `--K_max 128` (new in this commit). In learning, the number of
  iterations with which a volume is grown is drawn for every batch uniformly from 32 to 128, where it was 28 to
  36. Search, the validation during training and the final evaluation grow with 32 iterations as before, and the
  update is the measured one, without the bound.
* **Also reported, with no threshold:** the gate at 8 times (256 iterations, twice what the rule saw in learning);
  with it the gate of section 14.1 is passed or failed as a whole. The closed-loop rate at 32 iterations against
  step 6's three seeds, by the rule of section 12.4 on means with s = 4.68: what the redraw costs. The seconds per
  round.
* **What the result decides:** whether the design's own means are enough for the gate on a rule that regrows. If
  they are, the proposal to the PI is to put the redraw into the chain before the kept state, because a kept state
  multiplies the iterations a rule runs. If they are not, the gate asks for something the design does not have,
  and that goes back to design.
* **Verification done on the cluster's CPUs first:** `test_sim.py`. The run checks the new switch itself: 20 rounds
  of step 6's recipe on seed 0 with the switch off must give the first 21 rounds of the log of `p2s6_closed_s0`
  in every field both logs have (`cluster/same_logs.py`).
* **How it runs:** Docker containers on idle GPUs, each after the check on the host that nothing else holds the
  GPU and with the guard inside, from `~/wf/runs/p2wi` filled from the Mac. With F for
  `--battery 1 --codes 1 --C 32 --hid 128 --threads 8 --sim wf --device cuda --closed_search 1`: first
  `STOP_ROUND=20 bash cluster/jobs/p2_old_recipe.sh p2wi_off_s0 0 0.5 F` for the check, then
  `bash cluster/jobs/p2_old_recipe.sh p2wi_s<seed> <seed> 0.5 F --K_max 128` for seeds 0, 1 and 2, each followed in
  its container by `bash cluster/jobs/p2_iteration_gate.sh p2wi_s<seed>`. Seeds start as GPUs come free.
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2wi/COMMIT`.
* **Cluster job ids:** containers `wf-p2wi-off`, `wf-p2wi-s0`, `-s1`, `-s2`.
  **Estimated GPU-hours:** about 6.5 (three runs of a little over 2 hours, and the gates)

## After the run

Not run. The card, the switch (off unless given) and the folder `~/wf/runs/p2wi` (314 of 314 files, tests passed)
were prepared on 28 November; the runs waited for the PI's word, and at 02:40 UTC the PI had all runs stopped.

* **Result:**
* **Trajectories looked at:**
* **Supported, refuted or inconclusive:**
* **What was learned:**
* **Next step, and the card it becomes:**
* **Actual GPU-hours:**
