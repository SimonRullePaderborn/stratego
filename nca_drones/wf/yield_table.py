"""Phase 1, first measurement (experiments/2026-11-27-p1-untrained-yield.md). RUN ON THE CLUSTER ONLY.

How often does a team that follows nothing but its exploration noise capture the target within one leg?
There is no rule, no lattice and no learning here: an untrained rule has flat tubes, so its drones do exactly
this. The table says whether reward is reachable at all before training, per grid size and team.

    cd nca_drones && python3 -m wf.yield_table --out results/p1/untrained_yield.json

Target sight 2 is the PI's target. Sight 1 and 0 are reference rows that show how steep the cliff is; they
are measurements of the environment, not training settings.
"""
import argparse
import json
import math
import os
import time

import torch

from . import explore, maps, sim

SIDES = (12, 16, 24)
TEAMS = ((2, 1, 0), (3, 1, 1), (5, 3, 2))        # capturers, rechargers, observers
SIGHTS = (2, 1, 0)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def run_cell(batch, side, sight, steps, seed, device):
    """One cell of the table: B episodes of one leg each, drones driven by exploration noise alone."""
    cfg = sim.Cfg(H=side, W=side, target_sight=sight, T=steps)
    st = sim.draw_target(cfg, sim.reset(cfg, *batch, seed=seed + 1))
    B, N = st.kind.shape
    gen = torch.Generator(device=device).manual_seed(seed + 2)
    strategy = explore.draw_strategies(B, N, gen, device)
    flat = torch.zeros(B, N, 5, device=device)               # the scores of an untrained rule
    done = torch.zeros(B, dtype=torch.bool, device=device)
    when = torch.zeros(B, dtype=torch.long, device=device)
    in_sight = torch.zeros(B, dtype=torch.long, device=device)
    seen_before = torch.zeros(B, dtype=torch.bool, device=device)
    for t in range(steps):
        actions, strategy = explore.choose(strategy, flat, gen)
        active = st.exists & ~st.ghost & sim.powered(cfg, st)
        near = (active & ((st.pos - st.tpos[:, None, :]).abs().sum(-1) <= cfg.target_sight)).sum(1)
        was_seen = st.target_seen
        st, captured = sim.step(cfg, st, actions)
        first = captured & ~done
        when = torch.where(first, torch.full_like(when, t + 1), when)
        in_sight = torch.where(first, near, in_sight)
        seen_before = torch.where(first, was_seen, seen_before)
        done = done | captured
    k = int(done.sum())
    lo, hi = wilson(k, B)
    capturers = st.exists & (st.kind == sim.CAPTURER)
    return {
        "side": side, "sight": sight, "episodes": B, "steps": steps,
        "captured": k, "rate": k / B, "wilson95": [lo, hi],
        "mean_step_of_capture": float(when[done].float().mean()) if k else None,
        "drones_in_sight_at_capture": {str(n): int((in_sight[done] == n).sum()) for n in range(0, 6)},
        "target_was_sensed_before_capture": int(seen_before[done].sum()),
        "capturers_with_power_at_the_end": float((capturers & sim.powered(cfg, st)).sum() / capturers.sum().clamp(min=1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/p1/untrained_yield.json")
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=48, help="the leg limit")
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows, t0 = [], time.time()
    for side in SIDES:
        for team in TEAMS:
            cfg = sim.Cfg(H=side, W=side)
            n_cap, n_rech, n_obs = team
            batch = maps.make_batch(cfg, args.episodes, args.seed + side, device=args.device,
                                    n_cap=(n_cap, n_cap), n_rech=(n_rech, n_rech), n_obs=(n_obs, n_obs))
            for sight in SIGHTS:
                row = run_cell(batch, side, sight, args.steps, args.seed, args.device)
                row["team"] = {"capturers": n_cap, "rechargers": n_rech, "observers": n_obs}
                rows.append(row)
                print(f'{side}x{side}  team {n_cap}+{n_rech}+{n_obs}  sight {sight}:  {row["captured"]:5d} / {row["episodes"]}'
                      f' = {100 * row["rate"]:6.2f}%  [{100 * row["wilson95"][0]:.2f}, {100 * row["wilson95"][1]:.2f}]'
                      f'  in sight at capture {row["drones_in_sight_at_capture"]}  ({time.time() - t0:.0f} s)', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
