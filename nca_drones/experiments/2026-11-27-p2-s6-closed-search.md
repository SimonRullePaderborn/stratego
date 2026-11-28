# Experiment card: 2026-11-27-p2-s6-closed-search

## Before the run

* **Phase and gate it serves:** phase 2, step 6 of the chain in `PROGRAMME.md` section 12.4.
* **Hypothesis (one sentence):** searching in closed loop, with every search episode regrowing its own volume at
  every step from its realised past, does not lower closed-loop success on the 2,000 never-posed tasks, compared
  with step 5.
* **What would support it:** the closed-loop rate of this run (bf16, its own 2,000 never-posed tasks) is not below
  the rate of step 5 by more than max(2 s, 2 points), with s the standard deviation of the three rates of step 0
  (rule of section 12.4).
* **What would refute it:** a rate further below. Then the run is looked at on trajectories before step 7 is built
  on it.
* **The one thing that changes** relative to step 5: `--closed_search 1`. Until now the 64 rollouts of a task all
  followed one volume, grown once from the first observation. Now each of the 64 x 64 search episodes has a volume
  of its own, regrown from scratch (32 iterations) at every step with that episode's realised cells and battery
  levels burned in, exactly as the closed-loop evaluation does; the drones step along the next slice of that
  volume. Exploration noise, the choice of the earliest rewarded rollout, credit, the replay buffer, learning and
  evaluation are unchanged. Rollout 0 of a task is now the closed-loop greedy readout.
* **Why the goal needs it:** a target that moves cannot be met by a plan grown once; from phase 3 on, search has to
  react to what happens.
* **Also reported, with no threshold:** the seconds per round and the GPU memory against step 5 (the cost of
  regrowing for every episode at every step, which step 7 is meant to remove); the share of tasks where search finds
  a reward, round by round, against step 5; open-loop success, which this step no longer trains for directly.
* **Verification done on the cluster's CPUs first:** `test_sim.py`, which now ends with `test_closed_loop_search`: a
  greedy closed-loop rollout equals `evaluate_closed` step for step on 24 tasks, rollout 0 of a search equals that
  rollout, and the recorded actions replay to the recorded outcome. Then `python3 -m wf.test_task8` and a short
  smoke run on the GPU.
* **How it runs:** as step 5 (one container on GPU 1 after the host check, guard inside, 8 CPU cores), from
  `~/wf/runs/p2s6` filled from the Mac:
  `bash cluster/jobs/p2_old_recipe.sh p2s6_closed_s0 0 0.5 --battery 1 --codes 1 --C 32 --hid 128 --threads 8 --sim wf --device cuda --closed_search 1`.
  It starts when step 5 has left the GPU. If one of steps 1 to 3 or 5 is not kept, this run is void. (Added before
  the run: it uses the job script with `--stop_round 900`, for the reason given in the card of step 5, and an
  evaluation that also stores success task by task, for paired comparisons and the iteration gate; the run folder
  is `~/wf/runs/p2s6c`.)
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2s6/COMMIT`.
* **Cluster job ids:** container `wf-p2s6-s0`.   **Estimated GPU-hours:** 1 to 4 (to be narrowed by the smoke run)

## After the run

The run `p2s6_closed_s0` ran on 27 November 2026 from commit `ca39397` in `~/wf/runs/p2s6c`, container
`wf-p2s6-s0`, GPU 1 only. Files: `results/p2/p2s6_closed_s0.*`. The iteration gate was measured on 28 November with
the code of commit `1eb96e4` in `~/wf/runs/p2gate` (`cluster/jobs/p2_iteration_gate.sh`, container `wf-p2gate-b`,
GPU 7); files `results/p2/p2s6_closed_s0.x*.json`, `.x*.out` and `.gate.out`.

* **Result:** closed loop on its 2,000 never-posed tasks: 1,781 = 89.05% (bf16; interval 87.6 to 90.3) and 1,786
  (fp32). Open loop: 1,791 = 89.55%. Test set: 185 of 200 closed, 184 open. Chosen checkpoint: round 800 (0.95 on the
  100 validation tasks; round 900 had 0.93). Step 5: 77.9% closed and 68.8% open; the three seeds of step 5's
  recipe average 76.45%.
  Cost: 7.97 s per round against 0.43 s in step 5, 18.5 times; the run took 2.0 hours on one GPU.
  Search and the rule alone on the hard tasks (share with a reward in search / solved greedily): 0.25 / 0.05 at
  round 50, 0.47 / 0.20 at round 100, 0.65 / 0.38 at round 150, 0.87 / 0.66 at round 400, 0.98 / 0.91 at round 900.
  Step 5 stood at 0.30 / 0.11 around round 100, 0.47 / 0.26 around round 300 and 0.79 / 0.71 at the end.
  Iteration gate, reported for comparison (the same checkpoint, the same 2,000 tasks, growth with 2, 4 and 8 times
  the 32 iterations; paired difference to 89.05%, bf16): times 2: 89.20%, +0.15 points (95% interval -0.92 to
  +1.22); times 4: 84.05%, -5.00 (-6.48 to -3.52); times 8: 76.75%, -12.30 (-14.10 to -10.50). The gate asks for a
  lower bound above -2 points: passed at 2, failed at 4 and at 8. Open loop falls faster: 88.05%, 71.65% and
  50.55%. In fp32 the differences are +0.10, -5.10 and -12.55. The repeat of the run's own evaluation with the
  newer code changed no task.
* **Trajectories looked at:** none; the step is kept. The breakdown of the evaluation shows where it fails: 157 of
  the 219 failed tasks are those in which the recharger starts 9 or 10 cells from the seeker (62 of 141 and 3 of 81
  solved, against 96.5% at distances 1 to 8). Step 5 solved 25 of 140 and 0 of 83 of them.
* **Supported, refuted or inconclusive:** supported: 11.2 points above step 5, where the rule asks only that it is
  not more than 5.9 below (9.4 with the spread measured on GPUs, card `2026-11-27-p2-codes-by-width`).
* **What was learned:** search in closed loop costs 18 times the time per round and loses nothing; on this seed it
  gains 11 points over step 5 and 12.6 over the mean of step 5's three seeds, which is 2.3 standard errors with the
  spread of step 5's recipe. The rule alone solves hard tasks much earlier in training. The measured update, which
  normalises the rule's input, fails the iteration gate at 4 and 8 times the iterations, but gently: it loses 5 and
  12 points.
* **Next step, and the card it becomes:** step 7, the bounded update
* **Actual GPU-hours:** 2.0 for the run, 0.1 for the gate
