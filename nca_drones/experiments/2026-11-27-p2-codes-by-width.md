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

The eleven runs ran on 27 November 2026 from commit `38e4068` in `~/wf/runs/p2cw`, containers `wf-p2cw-g0`, `-g3`,
`-g6` and `-g7`, one GPU each. Files: `results/p2/p2cw_*` (evaluation, outputs and logs; the checkpoints stay in the
run folder on the cluster).

* **Result:** closed loop on each run's 2,000 never-posed tasks (bf16), in percent:

  | Cell | Seed 0 | Seed 1 | Seed 2 | Mean | Standard deviation | Open loop, mean |
  |---|---|---|---|---|---|---|
  | narrow, no codes (step 1's recipe) | 79.75 | 65.65 | 73.60 | 73.00 | 7.07 | 65.53 |
  | wide, no codes | 72.05 | 75.20 | 75.25 | 74.17 | 1.83 | 72.95 |
  | narrow, codes (step 2's recipe) | 79.60 | 85.00 | 85.50 | 83.37 | 3.27 | 78.88 |
  | wide, codes (step 5's recipe) | 77.90 (`p2s5_gpu_s0b`) | 71.05 | 80.40 | 76.45 | 4.84 | 70.45 |

  The pooled standard deviation of the twelve runs is 4.68 points (8 degrees of freedom). The standard error of the
  difference between two cell means is 3.82 points, twice that 7.64. Codes at the design's width: +2.28 points.
  Codes at the measured width: +10.37 points. Width without codes: +1.17; width with codes: -6.92. Steps 2 and 3
  together ("wide, codes") against step 1 ("narrow, no codes"): +3.45.
* **Trajectories looked at:** none; the question was one of means.
* **Supported, refuted or inconclusive:** refuted in its second half. At the design's width codes cost nothing, as
  stated. At the measured width they do not cost either: the mean with codes is 10 points above the mean without,
  2.7 standard errors in the other direction. Step 2's drop of 12 points was noise between two single runs.
* **What was learned:** a single run of this recipe has a standard deviation of 4.7 points, not the 2.95 of the three
  CPU baseline seeds, so every verdict of steps 1 to 3 that rested on one run each was within noise: with three
  seeds, the battery input changes nothing that can be measured (73.0% against the baseline's 72.0%), codes help or
  cost nothing, and the design's width has not paid on 8x8 (the best cell is the narrow rule with codes, 83.4%).
  Codes and width stay in the chain as they are: together they are 3.5 points above step 1, and the width is there
  for sixteen codes and ten drones, which this task does not ask for. For the steps on a GPU the yardstick s is
  4.68 points, so the rule of section 12.4 allows a single run to lie max(2 s, 2) = 9.4 points below the single run
  before it. It was measured with open-loop search; whether a run with closed-loop search (step 6 on) scatters as
  much is not known until one of those steps has three seeds.
* **Next step, and the card it becomes:** none of its own; the chain goes on from step 5 as run.
* **Actual GPU-hours:** 1.0 (four containers of 12 to 16 minutes)
