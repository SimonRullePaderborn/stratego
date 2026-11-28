# Experiment card: 2026-11-27-p2-s8-kept-dials

## Before the run

* **Phase and gate it serves:** phase 2, step 8 of the chain in `PROGRAMME.md` section 12.4: the look at a run that
  is failing, and the dials of the step, both of which the card `2026-11-27-p2-s8-kept-state` names for this case.
* **What was seen (27 November, run `p2s8_kept_s0b` after 230 of 900 rounds):** the kept recipe is not learning.
  Over rounds 125 to 149 the rule alone solves 15% of the easy tasks posed in search (step 7 at the same rounds:
  90%, step 6: 92%), the loss stays between 2.8 and 3.2 (steps 6 and 7: 1.6), and at round 150 the open-loop readout
  of a freshly grown volume solves 13% of the easy validation tasks (step 7: 90%). The last number is the surprise:
  about half of every batch are samples without a past, which are trained exactly as in step 7, and they are not
  learned either.
* **Hypothesis (one sentence):** with 4 iterations per step the rule has too little time to lay its plan anew after
  a move that left it, so the samples with a past pull the weights towards blurred tubes, and with 16 iterations
  per step the kept recipe learns the easy tasks as regrowth does.
* **What would support it:** in the run with `--ks 16` the rule alone solves at least 80% of the easy tasks over
  rounds 125 to 149 (the mean of `greedy_easy` in the log; regrowth stands at 90% there, the kept recipe with 4
  iterations at 15%).
* **What would refute it:** below 50% there. Between 50% and 80% the card is inconclusive.
* **The one thing that changes** between the runs: every run has the flags of step 7
  (`--battery 1 --codes 1 --C 32 --hid 128 --threads 8 --sim wf --device cuda --closed_search 1 --update bound`),
  seed 0, and stops at round 150 (`STOP_ROUND=150`, no evaluation by the protocol, which is defined for round 900).

  | Run | Further flags | What it is |
  |---|---|---|
  | `p2k_regrow_s0` | none | step 7's recipe, the reference |
  | `p2k_ks4_s0` | `--kept 1` | step 8's recipe again |
  | `p2k_ks8_s0` | `--kept 1 --ks 8` | 8 iterations per step (6 to 10 in learning) |
  | `p2k_ks16_s0` | `--kept 1 --ks 16` | 16 iterations per step (14 to 18 in learning) |
  | `p2k_full_s0` | `--kept 1 --grad_its 400` | 4 per step, back-propagation through the whole replay |

  The trainer changes in its log only: every round it now also writes the norm of the gradient before clipping
  (`gnorm`) and the loss split into samples without and with a realised past (`loss_fresh`, `loss_past`), each
  also over the slice of the next move alone (`next_fresh`, `next_past`). Nothing of this reaches the weights.
* **Also reported, with no threshold:** the same number for the runs with 8 iterations and with full
  back-propagation; for all five the split loss and the gradient norm over rounds 100 to 150, and the easy and hard
  validation numbers of round 150; the seconds per round, which say what a larger number of iterations costs.
* **What the result decides:** if 16 or 8 iterations per step learn, step 8 is run again in full (900 rounds,
  evaluated by the protocol, its own card) with the smaller of the two that does, and judged against step 7 by the
  rule of section 12.4; the saving against regrowth is then 4 or 2 times in place of 8. If only full
  back-propagation learns, the full run uses that. If none learns, step 8 is reported as failed, nothing is added
  to the learner, and the PI decides how the chain goes on, because step 9 builds on a kept state.
* **Verification done on the cluster's CPUs first:** `test_sim.py`, which now ends with `test_loss_parts` (the two
  parts add up to the loss; the next-slice part is that of slice tp + 1). The runs check the rest themselves: the
  log of `p2k_ks4_s0` must equal that of `p2s8_kept_s0b`, and the log of `p2k_regrow_s0` that of `p2s7_bound_s0b`,
  round for round in every field both logs have (`cluster/same_logs.py`). If they do, the new log fields changed
  nothing.
* **How it runs:** Docker containers on idle GPUs, each GPU after the check on the host that nothing else holds it
  and with the guard inside, from `~/wf/runs/p2k` filled from the Mac:
  `STOP_ROUND=150 bash cluster/jobs/p2_old_recipe.sh <run> 0 0.5 <flags of step 7> <further flags>`.
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2k/COMMIT`.
* **Cluster job ids:** containers `wf-p2k-*`.   **Estimated GPU-hours:** about 1.5 (five runs of 10 to 20 minutes)

## After the run

The five runs ran on 28 November 2026 from commit `e8b37cd` in `~/wf/runs/p2k` (151 of 151 files), containers
`wf-p2k-regrow`, `-ks4`, `-ks8`, `-ks16` and `-full`, on GPUs 4, 5, 0, 1 and 3, one each. Files:
`results/p2/p2k_*.log.jsonl` and `.train.out`; the means below are those of `cluster/log_means.py`.

* **Result:**

  | Run | Easy tasks solved by the rule alone, rounds 125 to 149 | Hard tasks: rule alone / reward in search | Loss, rounds 100 to 150: without a past / with a past | Loss on the next move's slice: without / with a past | Gradient norm | Validation at round 150: easy / hard closed loop | Seconds per round |
  |---|---|---|---|---|---|---|---|
  | `p2k_regrow_s0` (step 7) | 89.5% | 21.7% / 50.2% | 1.60 / 1.36 | 1.09 / 0.92 | 10.9 | 0.90 / 0.13 | 6.94 |
  | `p2k_ks4_s0` (step 8) | 15.0% | 2.3% / 19.9% | 2.83 / 3.27 | 2.08 / 2.95 | 12.2 | 0.13 / 0.01 | 4.08 |
  | `p2k_ks8_s0` | 82.8% | 7.3% / 26.5% | 1.75 / 1.95 | 1.10 / 1.46 | 7.7 | 0.87 / 0.11 | 4.26 |
  | `p2k_ks16_s0` | 90.4% | 6.1% / 20.8% | 1.57 / 1.89 | 0.92 / 1.38 | 6.4 | 0.97 / 0.12 | 4.90 |
  | `p2k_full_s0` | 17.1% | 0.3% / 21.3% | 2.70 / 3.26 | 1.63 / 2.62 | 8.8 | 0.29 / 0.01 | 4.31 |

  A loss of 4.16 is that of a uniform guess over the 64 cells. Both checks held: the log of `p2k_ks4_s0` equals
  that of `p2s8_kept_s0b`, and the log of `p2k_regrow_s0` that of `p2s7_bound_s0b`, in all 22 common fields for
  all 151 rounds. The new log fields changed nothing, and a run on a GPU repeats exactly.
* **Trajectories looked at:** none; the split loss took their place.
* **Supported, refuted or inconclusive:** supported. With 16 iterations per step the rule alone solves 90.4% of the
  easy tasks, where the card asked for at least 80%. With 8 it solves 82.8%.
* **What was learned:**
  * The number of iterations per step decides whether the kept recipe learns: 15% at 4, 83% at 8, 90% at 16, where
    regrowth has 90%.
  * The span of back-propagation does not: through the whole replay at 4 per step, 17%. The default of 32 stays.
  * The split loss shows the way the damage runs. At 4 per step the samples with a past stay near a guess on the
    slice of their own next move (2.95), and they pull the samples without a past with them (2.08, where the same
    samples reach 1.09 in regrowth), although those are trained exactly as in step 7: the weights are shared. At 8
    per step both recover (1.46 and 1.10).
  * The size of the gradients is not the cause: regrowth has norms as large (10.9) as the run that fails (12.2).
  * On the hard tasks the kept runs are behind regrowth at round 150: search finds a reward in 27% of them at 8 per
    step and 21% at 16, where regrowth finds one in 50%, and the rule alone solves 7% and 6% against 22%. The
    validation numbers are level (0.11 and 0.12 against 0.13). Whether that gap closes is for the full run to show.
  * Cost per round: 4.26 s at 8 per step and 4.90 s at 16, against 6.94 s for regrowth.
* **Next step, and the card it becomes:** step 8 in full with 8 iterations per step, the smaller number that learns:
  card `2026-11-28-p2-s8-kept-ks8`.
* **Actual GPU-hours:** 1.0 (five runs of 10 to 18 minutes)
