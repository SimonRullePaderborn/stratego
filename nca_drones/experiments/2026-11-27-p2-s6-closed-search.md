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
  the run: it uses the job script with `--stop_round 900`, from `~/wf/runs/p2s6b`, for the reason given in the card
  of step 5.)
* **Commit:** the commit that holds this card, named in `~/wf/runs/p2s6/COMMIT`.
* **Cluster job ids:** container `wf-p2s6-s0`.   **Estimated GPU-hours:** 1 to 4 (to be narrowed by the smoke run)

## After the run

* **Result:**
* **Trajectories looked at:**
* **Supported, refuted or inconclusive:**
* **What was learned:**
* **Next step, and the card it becomes:** step 7, the state kept and bounded
* **Actual GPU-hours:**
