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

* **Result:**
* **Trajectories looked at:**
* **Supported, refuted or inconclusive:**
* **What was learned:**
* **Next step, and the card it becomes:**
* **Actual GPU-hours:**
