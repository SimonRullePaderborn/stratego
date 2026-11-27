# Experiment card: 2026-11-27-p2-codes-by-width

## Before the run

* **Phase and gate it serves:** phase 2, the analysis that the rule of `PROGRAMME.md` section 12.4 asks for when a
  step is not kept. Step 2 (codes at the measured rule's width) scored 70.5% on its 2,000 never-posed tasks against
  82.5% for step 1, a drop of 12 points where the rule allows max(2 s, 2) = 5.9 (s = 2.95 points, the standard
  deviation of the three baseline rates 75.35%, 70.45% and 70.05%). Step 3 (codes at the design's width) scored
  83.7%. The same recipe as step 3 on a GPU (step 5) scored 77.9%, so single runs differ by about six points from
  rounding alone, and the combination "wide rule without codes" was never run.
* **Hypothesis (one sentence):** at the design's width (32 state channels, 128 hidden units) codes cost nothing, and
  at the measured width (16 and 64) they cost more than the run-to-run spread.
* **What would support it:** with three seeds per cell, the mean closed-loop rate of "wide, codes" is not below that
  of "wide, no codes" by more than twice the standard error of the difference of the two means, and the mean of
  "narrow, codes" is below that of "narrow, no codes" by more than that.
* **What would refute it:** "wide, codes" clearly below "wide, no codes": then codes still cost at the design's
  width, and the capacity for codes (a wider state, as a reviewer proposed) becomes the next question before the
  chain goes on. Or "narrow, codes" not below "narrow, no codes": then step 2's drop was noise.
* **The one thing that changes** between cells: `--codes 1` or not, and `--C 32 --hid 128` or the default 16 and 64.
  Every run has `--battery 1 --sim wf --device cuda --threads 8`, the old search, shared noise and the protocol of
  `EVAL_PROTOCOL.md` through `cluster/jobs/p2_old_recipe.sh`. Seeds 0, 1 and 2 per cell. The cell "wide, codes",
  seed 0, exists already as `p2s5_gpu_s0b` (77.9%) and is not repeated.
* **What the result decides:** whether codes and width stay in the chain as they are (steps 2 and 3 judged together
  against step 1), and how many seeds later steps need: the pooled standard deviation of these twelve runs replaces
  s from three CPU seeds as the yardstick for the GPU steps.
* **Verification done on the cluster's CPUs first:** `test_sim.py` in the run folder.
* **How it runs:** four Docker containers, one per idle GPU (0, 3, 6 and 7, each after the check on the host that
  nothing else holds it, with the guard inside), each working through its cell's seeds one after the other, from
  `~/wf/runs/p2cw` filled from the Mac. Run names `p2cw_<cell>_s<seed>` with cells `narrow`, `wide`, `narrowcodes`
  and `widecodes`. An open-loop run takes about 7 minutes on a GPU.
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2cw/COMMIT`.
* **Cluster job ids:** containers `wf-p2cw-g0`, `-g3`, `-g6`, `-g7`.   **Estimated GPU-hours:** about 1.5 (11 runs of 7 minutes)

## After the run

* **Result:**
* **Trajectories looked at:**
* **Supported, refuted or inconclusive:**
* **What was learned:**
* **Next step, and the card it becomes:**
* **Actual GPU-hours:**
