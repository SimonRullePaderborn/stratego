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
  If an earlier step is not kept, this run is void. (Added before the run: the PI allowed the node's idle GPUs on
  27 November, so this run starts at once on GPU 5, after the same check and with the same guard, while steps 6 and
  7 are still running on GPUs 1 and 4.)
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2s8/COMMIT`.
* **Cluster job ids:** container `wf-p2s8-s0`.   **Estimated GPU-hours:** under 1 (to be narrowed by the smoke run)

## A first start that was stopped, and the repeat (written before the repeat)

The run `p2s8_kept_s0` from commit `e8b6552` was started on GPU 5 and stopped after about a minute, for the reason
given in the card of step 7: the bounded update, which this step builds on, gave a NaN loss from round 0. The smoke
run before it had shown the NaN; the run was started all the same, because only the exit code of the smoke run was
checked. From now on a smoke run's output is read for a loss that is not finite before a run starts. The run is
repeated as `p2s8_kept_s0b` from `~/wf/runs/p2s8b`, at the commit that holds this paragraph.

## After the run

The run `p2s8_kept_s0b` ran on 27 November 2026 from commit `9c264c1` in `~/wf/runs/p2s8b`, container
`wf-p2s8b-s0`, GPU 5 only. Files: `results/p2/p2s8_kept_s0b.*`. The iteration gate was measured on 28 November with
the code of commit `1eb96e4` in `~/wf/runs/p2gate` (container `wf-p2gate-b`, GPU 7); files
`results/p2/p2s8_kept_s0b.x*.json`, `.x*.out` and `.gate.out`.

* **Result:** closed loop on its 2,000 never-posed tasks: 398 = 19.9% (bf16; interval 18.2 to 21.7) and 406 (fp32).
  Open loop: 415 = 20.75%. Test set: 47 of 200 closed, 53 open; of the 200, 51.5% never met and 25% ran out of
  time. Chosen checkpoint: round 900. Step 7: 94.2%.
  The course of the run: until round 450 the rule alone solved 11 to 22% of the easy tasks posed in search and the
  open-loop readout 10 to 18% of the easy validation tasks. From round 500 it learned: 30% of the easy tasks at
  round 500, 54% at 650, 74% at 750, 90% at 900. At round 900 its validation numbers (easy 0.94, hard 0.26 open and
  0.32 closed) stood where step 7's stood near round 200 (0.97, 0.345 and 0.44). The run was still climbing when
  the protocol ended it.
  Cost: 4.2 s per round over the last 400 rounds, against 7.47 s for step 7. The iterations of a search episode
  fell from 640 to 112, but about 3 s of a round are not iterations (the simulator, credit, the buffer), so on 8x8
  the round is 1.8 times faster, not 6. The run took 1.3 hours; its first 150 rounds were slowed by other jobs on
  the same cores until the container was given eight of its own.
  Iteration gate, on this checkpoint (growth and iterations per step times 2, 4 and 8; paired difference to 19.9%,
  bf16): -0.05 points (95% interval -0.94 to +0.84), 0.00 (-0.93 to +0.93) and -0.60 (-1.54 to +0.34). All three
  lower bounds are above -2: the gate is passed, by a rule that solves a fifth of the tasks. In fp32: -0.75, -0.20
  and -0.70. Open loop: 20.50%, 20.40% and 20.25%.
  Not measured: the largest root-mean-square of the state at the end of an episode; under the bound it cannot
  exceed 1.
* **Trajectories looked at:** none drawn. The look at the failing run is in two cards: the short runs of
  `2026-11-27-p2-s8-kept-dials`, and the kept readout of the checkpoints of steps 6 and 7 in
  `2026-11-28-p2-s8-kept-readout`.
* **Supported, refuted or inconclusive:** refuted. 74.3 points below step 7: step 8 with 4 iterations per step is
  not kept.
* **What was learned:** with 4 iterations per step the kept recipe starts to learn about 450 rounds late, so the
  900 rounds of the protocol end it far from finished. The delay is in learning, not in flying: step 7's
  checkpoint, which never trained with a kept state, solves 84.8% when flown with one at 4 iterations per step.
  And training with a kept state gives what the bound alone did not: a rule that is unchanged at 8 times its
  iterations, where step 7's rule falls to 4%.
* **Next step, and the card it becomes:** the dials of this step (card `2026-11-27-p2-s8-kept-dials`), then the
  full run with the dial they select (card `2026-11-28-p2-s8-kept-ks8`). Step 9 waits for that run.
* **Actual GPU-hours:** 1.3 for the run, 0.1 for the gate
