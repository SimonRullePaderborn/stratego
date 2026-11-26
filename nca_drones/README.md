# Growing coupled worldlines: a space-time NCA that learns drone rendezvous from sparse reward

A neural cellular automaton grows, in a (t, x, y) lattice, the worldlines ("tubes") of two
drones that have to cooperate: a **seeker** that must reach a target but runs out of battery
on the way, and a **recharger** drone that has to come and meet it. Everything is learned from
a simulator and its sparse 0/1 reward. No solver, no dynamic programming, no distance
features, no shaped reward, no demonstrations.

Status of the headline number is in [Results](#results); the honest caveats are in
[What is and is not hand-made](#what-is-and-is-not-hand-made).

## The task

`env.py`

* 8x8 grid, discrete time, horizon T = 20. Two drones, 5 actions each (stay, N, S, W, E).
* The seeker's battery holds B = 5 and loses one unit **every** step (hovering costs too).
  When both drones are on the same cell at the same time, the battery is refilled.
* Reward 1 when the seeker stands on the target. The episode ends on the reward, on an empty
  battery, or at t = T. There is no other signal.
* Tasks are three distinct uniformly random cells (seeker, recharger, target), filtered so
  that every task is solvable: if the target is more than B steps away, the recharger must
  start within 2B steps of the seeker (with T >= B + 14 this condition is exact).
* **Recharge-needed tasks** are those with Manhattan(seeker, target) > B: the seeker cannot
  make it alone. 44% of all tasks. The success criterion is measured on these only.

The battery level is never shown to the learner. Under a uniformly random policy, a
recharge-needed task at distance 6 is solved in about 0.04% of rollouts, and never beyond
distance 7.

## The model

`nca.py` - one local rule shared by all 21 x 8 x 8 voxels, iterated K = 32 times:

```
perception  = Conv3d(16 state + 3 seed channels -> 64, kernel 3x3x3, zero padding)
update      = ReLU -> 1x1x1 conv (64 -> 64) -> ReLU -> 1x1x1 conv (64 -> 16), added to the state
tube logits = 1x1x1 conv (16 -> 2)          # channel 0 seeker, channel 1 recharger
```

38,130 parameters. No coordinates, no pooling, no attention, no iteration counter. The state
is normalised per voxel across channels (a local operation). The lattice is told exactly one
thing: slice t = 0 holds three one-hot maps (seeker, recharger, target). In closed-loop mode
the slices of the past additionally hold where the two drones have actually been.

**Steering.** At time t a drone looks at the cells it can reach and moves to the one with the
highest tube logit in slice t+1. That is the whole controller. Two readouts are reported:

* *open-loop dream*: grow the volume once from the t = 0 seeds and fly the whole episode from it;
* *closed-loop*: after every step, burn the realised positions into the past slices and
  regrow the future ("the environment is pushed through the dreamed volume").

## How it learns

`train.py`, one round:

1. **Sample** 64 tasks uniformly. No curriculum.
2. **Grow** the volume with the current rule.
3. **Search**: 64 joint rollouts per task follow the two tubes stochastically in the simulator.
4. **Reward**: keep, per task, the rollout that was rewarded earliest (ties: the one the tissue
   finds most likely). Unrewarded rollouts are thrown away.
5. **Credit**: replay the seeker's path against 8 random-walk rechargers. If it is still
   rewarded, the recharger was irrelevant for this reward and its part is not imitated.
6. **Learn**: per-slice cross-entropy pulls the tubes towards the rewarded rollout. In 60% of
   the samples a random-length prefix of that rollout is burned into the past, so the rule
   also learns to regrow a future around a fixed past.

Three ingredients turned out to decide whether the coupling is learned at all; see
[What made it learnable](#what-made-it-learnable).

## Results

_To be filled in from `runs/<name>/analysis_*.json` once the final checkpoint is fixed._

## What made it learnable

## What is and is not hand-made

## Reproducing

```
python3 test_sim.py                      # simulator / relabelling / credit / cross-play consistency checks
python3 train.py --name strict --hind 0 --prune 2 --rounds 1500 --hard_frac 0.25 --null_k 8 \
        --shared 0.5 --ez 4 --eps_max 0.3 --prefix 0.6 --eval_closed 100 --eval_every 50 --lr_final 0.2
python3 analyze.py runs/strict/best.pt --full
```
