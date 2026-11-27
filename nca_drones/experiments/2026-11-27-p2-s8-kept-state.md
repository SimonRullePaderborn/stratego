# Experiment card: 2026-11-27-p2-s8-kept-state

## Before the run

* **Phase and gate it serves:** phase 2, step 8 of the chain in `PROGRAMME.md` section 12.4, and the iteration gate
  of section 14.1.
* **Hypothesis (one sentence):** keeping the lattice's state from step to step, with 4 iterations after each new
  observation in place of regrowing with 32, does not lower closed-loop success on the 2,000 never-posed tasks,
  compared with step 7.
* **What would support it:** the closed-loop rate of this run (bf16, its own 2,000 never-posed tasks) is not below
  the rate of step 7 by more than max(2 s, 2 points), with s the standard deviation of the three rates of step 0
  (rule of section 12.4).
* **What would refute it:** a rate further below. Then the run is looked at on trajectories before step 9 is built
  on it; the dials that belong to this step (iterations per step, the span of back-propagation) are tried before
  anything else, each as its own run.
* **The one thing that changes** relative to step 7: `--kept 1` (with `--ks 4 --ks_jitter 2 --grad_its 32`). In
  closed loop, in search and in evaluation, a volume is grown once, with 32 iterations on the first observation;
  after every step the new cells and battery level are burned in and the rule runs 4 iterations on the state it
  already has. Learning rebuilds that state for a stored rollout by replaying its realised past the same way
  (`replay_kept`): 32 iterations (plus or minus 4, as before) on the first observation, then 2 to 6 iterations per
  step, redrawn for every batch, with gradient through the last 32 iterations, which is the whole growth for a
  sample without a past and up to eight steps for one with a long past. The share of samples with a past (60%), the
  loss, credit, search noise, widths and the bounded update are those of step 7. The open-loop readout still grows
  a volume once and flies it.
* **Why the goal needs it:** regrowing for every search episode at every step costs eight times the iterations; at
  24x24 with ten drones that is the difference between a run of a day and a run of a week.
* **Also reported, with no threshold in this card:** the seconds per round against steps 6 and 7; the iteration gate
  (every iteration count times 2 and times 4 at evaluation, paired on the same tasks; `analyze.py --mult`); the
  largest root-mean-square of the state at the end of an episode.
* **Verification done on the cluster's CPUs first:** `test_sim.py`, which now ends with `test_kept_state`: the state
  that learning rebuilds equals the state grown one step at a time, for pasts of different lengths in one batch; a
  greedy kept rollout equals the kept closed-loop readout; gradient reaches the weights. Then a run of 8 rounds
  without `--kept 1`, whose log must equal that of step 7's code (the new code changes nothing when switched off),
  and a smoke run on the GPU.
* **How it runs:** as steps 6 and 7, from `~/wf/runs/p2s8` filled from the Mac:
  `bash cluster/jobs/p2_old_recipe.sh p2s8_kept_s0 0 0.5 --battery 1 --codes 1 --C 32 --hid 128 --threads 8 --sim wf --device cuda --closed_search 1 --update bound --kept 1`.
  If an earlier step is not kept, this run is void.
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2s8/COMMIT`.
* **Cluster job ids:** container `wf-p2s8-s0`.   **Estimated GPU-hours:** under 1 (to be narrowed by the smoke run)

## After the run

* **Result:**
* **Trajectories looked at:**
* **Supported, refuted or inconclusive:**
* **What was learned:**
* **Next step, and the card it becomes:** step 9, the sliding window
* **Actual GPU-hours:**
