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

* **Result:**
* **Trajectories looked at:**
* **Supported, refuted or inconclusive:**
* **What was learned:**
* **Next step, and the card it becomes:**
* **Actual GPU-hours:**
