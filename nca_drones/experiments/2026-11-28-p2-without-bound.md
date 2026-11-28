# Experiment card: 2026-11-28-p2-without-bound

## Before the run

* **Phase and gate it serves:** phase 2, steps 7 and 8 of the chain in `PROGRAMME.md` section 12.4. Step 7 is judged
  again with three seeds on each side, and the kept state is measured on step 6: "the state without its bound",
  which section 12.4 names as a run that can be measured against the step it would change.
* **What was seen (28 November):** the kept recipe with 8 iterations per step scored 67.95%, 18.85% and 8.70% on
  seeds 0, 1 and 2. Seeds 1 and 2 stalled as the first run of step 8 had: for hundreds of rounds the rule alone
  solved about 15% of the easy tasks posed in search. The same stall shows without a kept state, in step 7's own
  recipe: its seed 2 solved 15% of the easy tasks at round 150 and 28% at round 550, where seeds 0 and 1 solved
  87% and 94% at round 150. None of the sixteen runs with the measured update (three baseline seeds, the twelve of
  the codes table, step 6) ended below 65%. The two updates differ in more than the bound: the measured one normalises the
  state on its way into the convolution, so a state that is still small is read at full size; the bounded one
  reads it as it is, and both start from a state of zero and an update layer of zero.
* **Hypothesis (one sentence):** the stall comes with the bounded update and not with the kept state: with the
  measured update, closed-loop search learns on every seed, regrown and with a kept state at 8 iterations per step.
* **What would support it:** in all five runs of this card the rule alone solves at least 50% of the easy tasks
  posed in search, as the mean of `greedy_easy` over rounds 125 to 149. Stalled runs stand near 15% there, runs
  that learn at 80 to 95%.
* **What would refute it:** one of the five below 30%. Between 30% and 50% the card is inconclusive.
* **The one thing that changes:** `--update bound` is taken out of two recipes that have been run with it.
  With F for `--battery 1 --codes 1 --C 32 --hid 128 --threads 8 --sim wf --device cuda --closed_search 1`:

  | Runs | Flags | Stands against |
  |---|---|---|
  | `p2s6_closed_s1`, `_s2` | F (step 6's recipe; seed 0 is `p2s6_closed_s0`) | step 7, seeds 0, 1, 2 |
  | `p2s8_nobound_s0`, `_s1`, `_s2` | F `--kept 1 --ks 8` | `p2s8_ks8_s0`, `_s1`, `_s2` |

* **Decided by the same runs at round 900, by the rule of section 12.4 on means of three seeds** with the yardstick
  of the GPU runs, s = 4.68 points (card `2026-11-27-p2-codes-by-width`): a recipe is not below another when its
  mean closed-loop rate is not lower by more than max(2 s sqrt(2/3), 2) = 7.6 points. Two verdicts: whether step
  7 (bounded) is kept against step 6, and whether the kept state on step 6 is kept against step 6. The card
  `2026-11-28-p2-s8-kept-ks8` took its yardstick from the runs it compared, which lets a recipe that fails on some
  seeds widen its own margin; this card does not.
* **Also reported, with no threshold:** the iteration gate for every new checkpoint (times 2, 4 and 8, paired); the
  seconds per round.
* **What the result decides:** which version the chain stands on. If step 7 falls against step 6, the last version
  that worked is step 6, and by the rule of the chain the kept state is built on it. The design's rule has the
  bound (section 7.1), so going on without it is a departure that the PI is told of and can overrule. If the kept
  state stalls without the bound as well, the stall is the kept state's, and how the chain goes on is the PI's
  decision.
* **Verification done on the cluster's CPUs first:** `test_sim.py` in the run folder. The first rounds of each run
  are read for a loss that is not finite before the runs are left alone.
* **How it runs:** five Docker containers, one per idle GPU, each after the check on the host that nothing else
  holds the GPU and with the guard inside, from `~/wf/runs/p2nb` filled from the Mac:
  `bash cluster/jobs/p2_old_recipe.sh <run> <seed> 0.5 <flags>`, followed in the same container by
  `bash cluster/jobs/p2_iteration_gate.sh <run>`.
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2nb/COMMIT`.
* **Cluster job ids:** containers `wf-p2nb-r1`, `-r2` (regrown) and `wf-p2nb-k0`, `-k1`, `-k2` (kept).
  **Estimated GPU-hours:** about 8 (two runs of 2 hours, three of 1.2 hours, and the gates)

## After the run

The five runs started on 28 November 2026 at 02:00 UTC from commit `8e434ea` in `~/wf/runs/p2nb` (271 of 271
files), containers `wf-p2nb-r1`, `-r2`, `-k0`, `-k1`, `-k2` on GPUs 0, 1, 3, 4 and 5. **They were stopped at 02:40
UTC at the PI's request**, after about 305, 309, 551, 528 and 540 rounds. None reached round 900, so none was evaluated
by the protocol and no iteration gate was measured. Files: `results/p2/<run>.stopped.log.jsonl` (the log as it
stood) and `.train.out`. The folder on the cluster still holds each run's last checkpoint.

* **Result:** the part of the card that concerns rounds 125 to 149 was reached by all five runs. Easy tasks solved
  by the rule alone there, with the runs that have the bound beside them:

  | Recipe | Seed 0 | Seed 1 | Seed 2 |
  |---|---|---|---|
  | Regrown, measured update (step 6) | 91.8% | 90.5% | 89.1% |
  | Regrown, bounded update (step 7) | 89.5% | 94.9% | 15.8% |
  | Kept, 8 per step, measured update | 84.2% | 72.4% | 91.6% |
  | Kept, 8 per step, bounded update | 82.8% | 15.3% | 14.7% |

  Where the runs stood when stopped, on the 100 validation tasks that need a recharge, closed loop: step 6's recipe
  0.68 (seed 1) and 0.54 (seed 2) at round 300, where seed 0 had 0.64; the kept state without the bound 0.14
  (seed 0, round 550; 0.55 at round 500), 0.32 (seed 1, round 500) and 0.38 (seed 2, round 500), where step 6
  had 0.82 at round 500. In search, the kept runs found a reward in 39 to 50% of the hard tasks at round 500,
  step 6 in 90%.
* **Trajectories looked at:** none.
* **Supported, refuted or inconclusive:** supported, for what the card asked of rounds 125 to 149: all five runs are
  above 50% (72.4% to 91.6%), and no run with the measured update stalled. The two verdicts at round 900, step 7
  against step 6 and the kept state on step 6 against step 6, were not reached.
* **What was learned:** the stall comes with the bounded update. With it, three of six runs stalled; without it,
  none of the six did, on the same seeds, regrown and kept. What is not known: where the kept state without the
  bound ends at round 900. At the stop it learned the easy tasks on every seed but was well behind regrowth on the
  tasks that need a recharge, and its search found a reward for about half as many of them.
* **Next step, and the card it becomes:** none started. All runs are stopped; how the chain goes on is the PI's
  decision.
* **Actual GPU-hours:** 3.3 (five containers of 40 minutes)
