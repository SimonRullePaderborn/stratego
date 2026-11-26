# Pre-registered final evaluation (written before any v10 checkpoint was evaluated on these tasks)

Run under test: `v10` (audited path, `--strict 1`, seed 0), trained from scratch.

1. **Checkpoint**: `runs/v10/best.pt`, i.e. the checkpoint with the best closed-loop success on the
   first 100 validation tasks (seed 999). No other checkpoint of v10 is evaluated on the tasks below.
2. **Tasks**
   * the original held-out test set: 200 recharge-needed tasks, seed 12345 (excluded from training by key);
   * 2000 new recharge-needed tasks: seeds 2001, 2002, ..., 2010, 200 tasks each, drawn with
     `analyze.fresh_tasks`, which removes every task that is in the validation/test sets **or was ever
     posed to the learner during training** (the run logs a bitmap of posed tasks).
3. **Readouts**: open-loop dream (grown once from t = 0) and closed-loop (regrown each step with the
   realised joint history burned in). Both in bfloat16 (as trained) and in fp32.
4. **Reported**: success counts, Wilson 95% intervals, success by seeker-recharger start distance.
5. **Criterion**: >= 75% on the 200-task test set (the PI's wording) and, as the stronger population
   statement, a 95% lower confidence bound above 75% on the 2000 new tasks.

The ablation `v11_noshared` (same, `--shared 0`) is evaluated the same way with its own `best.pt`.

**Stopping rule (added at v10 round ~600, before any evaluation on the tasks above):** v10 and
`v11_noshared` are each evaluated once, with the `best.pt` that exists when the run reaches round 900
(the round at which the earlier run v6 was stopped). No earlier or later checkpoint is evaluated on
the test set or on the 2000 new tasks.

## Outcome of the pre-registered evaluation

* v10 (`best.pt` at round 900 = the round-850 checkpoint): test 164/200 closed-loop, 158/200 open-loop;
  2000 new tasks 1507/2000 = 75.35% closed-loop (95% CI 73.4-77.2), 1467/2000 = 73.35% open-loop.
  Criterion part 1 (>= 75% on the 200-task test set): **met**. Criterion part 2 (lower confidence
  bound above 75% on the 2000 new tasks): **not met**.
* v11_noshared (`best.pt` at round 900): test 135/200 and 134/200; 2000 new tasks 65.45% and 63.75%.

## Follow-up, declared after seeing the outcome above

v10 is continued from its round-900 state to round 1500 as `v10b` (same flags; the replay buffer
restarts empty). Its `best.pt` at the end of training is evaluated once on 2000 further new tasks,
seeds 3001..3010, with the same filters. This is a separate, later claim about a longer run; it does
not replace the result above, and the original test set is not re-used for it.
