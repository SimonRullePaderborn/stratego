"""Zero-shot probes: what does a tissue trained on the 8x8 task do when the world is changed?

NO training happens here.  A trained checkpoint is grown on altered lattices /
observations and flown in the (correspondingly altered) simulator.  The point is
to see what transfers for free before designing the next environments.

  python3 probe_extensions.py <ckpt> <size|target|seekers|escort> [out.json]

  size     the same rule on larger grids and longer horizons (no retraining)
  target   closed-loop pursuit of a moving target; target hidden at first
  seekers  two battery-limited drones in the seeker channel, one recharger
  escort   when does the recharger let go of the seeker?
"""
import json
import sys

import torch

from analyze import load
from diag_search import mix
from env import Cfg, DroneEnv, MOVES, manhattan, sample_tasks
from nca import make_cond
from train import evaluate, evaluate_closed, grow, rollout

AMP = True


def rate(x):
    return round(float(x.float().mean()), 3) if len(x) else None


def outcomes(o, cfg):
    ok, met = o["success"], o["n_recharge"] > 0
    return dict(success=rate(ok), never_met=rate(~ok & ~met), met_then_died=rate(~ok & met & (o["tau"] < cfg.T)),
                timeout=rate(~ok & met & (o["tau"] >= cfg.T)))


@torch.no_grad()
def search_yield(model, s, r, g, cfg, K, gen, M=64):
    """greedy success and 'any of the 64 training-style rollouts is rewarded'."""
    N = len(s)
    logits = torch.cat([grow(model, s[i:i + 100], r[i:i + 100], g[i:i + 100], cfg, K, AMP, bs=50) for i in range(0, N, 100)])
    eps, temp, sh, gr = mix(N, M)
    ro = rollout(logits, s, r, g, cfg, M=M, eps=eps, temp=temp, greedy=gr, gen=gen, ez=4, shared=sh)
    succ = ro["success"].reshape(N, M)
    return succ[:, 0], succ.any(1)


# ------------------------------------------------------------------ larger lattices
@torch.no_grad()
def probe_size(model):
    out = {}
    base = Cfg()
    gen = torch.Generator().manual_seed(7001)
    s, r, g = sample_tasks(500, base, gen, "need")
    for K in (32, 48, 64):
        out[f"8x8 T=20 K={K} (as trained)"] = rate(evaluate(model, (s, r, g), base, K, AMP)["success"])
    # only the time axis is longer
    c = Cfg(H=8, W=8, T=30, B=5)
    for K in (32, 48, 64):
        out[f"8x8 T=30 K={K} (longer horizon, same tasks)"] = rate(evaluate(model, (s, r, g), c, K, AMP)["success"])
    # the same tasks placed inside a 12x12 lattice: do the walls / absolute position matter?
    c12 = Cfg(H=12, W=12, T=20, B=5)
    for off in (0, 2, 4):
        o = torch.tensor([off, off])
        out[f"same tasks inside 12x12, offset {off}, K=32"] = rate(evaluate(model, (s + o, r + o, g + o), c12, 32, AMP)["success"])
    print(json.dumps(out, indent=1), flush=True)
    # uniformly sampled tasks on larger grids (longer escorts than ever trained)
    for L, T, Ks, n in ((12, 27, (32, 48, 64), 400), (16, 35, (48, 64, 96), 300)):
        cL = Cfg(H=L, W=L, T=T, B=5)
        s2, r2, g2 = sample_tasks(n, cL, gen, "need")
        dsg, dsr = manhattan(s2, g2), manhattan(s2, r2)
        for K in Ks:
            o = evaluate(model, (s2, r2, g2), cL, K, AMP)
            ok = o["success"]
            row = outcomes(o, cL)
            row.update(target_within_14=rate(ok[dsg <= 14]), target_beyond_14=rate(ok[dsg > 14]),
                       recharger_within_7=rate(ok[dsr <= 7]), recharger_8_to_10=rate(ok[dsr >= 8]),
                       share_beyond_14=rate(dsg > 14))
            out[f"{L}x{L} T={T} K={K}"] = row
            print(f"{L}x{L} T={T} K={K}", row, flush=True)
        gz, anyz = search_yield(model, s2, r2, g2, cL, Ks[1], gen)
        out[f"{L}x{L} search at K={Ks[1]}"] = dict(greedy=rate(gz), any_of_64=rate(anyz), any_target_within_14=rate(anyz[dsg <= 14]),
                                                  any_target_beyond_14=rate(anyz[dsg > 14]))
        print(f"{L}x{L} search", out[f"{L}x{L} search at K={Ks[1]}"], flush=True)
    return out


# ------------------------------------------------------------------ moving / hidden target (closed loop)
@torch.no_grad()
def closed_loop(model, s, r, g0, cfg, K, reveal=0, p_move=0.0, refresh=True, gen=None, bs=256, evade=False):
    """Closed-loop readout with a target that may move (lazy random walk) and may be hidden for the first steps.

    What the tissue is told about the target: its CURRENT cell, written into slice 0 as if it had
    always been there (the tissue was never trained on anything else).  refresh=False keeps the stale t=0 cell.
    Order within a step: the tissue observes, both drones choose, the target moves, the drones move."""
    model.eval()
    n = s.shape[0]
    env = DroneEnv(s, r, g0, cfg)
    lim = torch.tensor([cfg.H - 1, cfg.W - 1])
    hist = torch.zeros(n, 2, cfg.T + 1, dtype=torch.long)
    hist[:, 0, 0] = s[:, 0] * cfg.W + s[:, 1]
    hist[:, 1, 0] = r[:, 0] * cfg.W + r[:, 1]
    g_cur = g0.clone()
    arn = torch.arange(n)
    for t in range(cfg.T):
        act = torch.nonzero(~env.done).squeeze(1)
        if len(act) == 0:
            break
        g_seen = g_cur if refresh else g0
        logits = torch.zeros(n, 2, cfg.T + 1, cfg.H, cfg.W)
        for i in range(0, len(act), bs):
            j = act[i:i + bs]
            cond = make_cond(s[j], r[j], g_seen[j], cfg, hist[j], torch.full((len(j),), t))
            if t < reveal:
                cond[:, 2] = 0
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=AMP):
                lg, _ = model(cond, K)
            logits[j] = lg.float()
        acts = []
        for d, pos in ((0, env.ps), (1, env.pr)):
            c = env.cand(pos)
            acts.append(logits[arn[:, None], d, t + 1, c[..., 0], c[..., 1]].argmax(1))
        if p_move > 0:
            mv = torch.rand(n, generator=gen) < p_move
            if evade:                                  # flee: step to the neighbouring cell farthest from the seeker (ties at random)
                opts = g_cur[:, None, :] + MOVES[None]
                opts = torch.minimum(torch.maximum(opts, torch.zeros_like(opts)), lim.expand_as(opts))
                dist = (opts - env.ps[:, None, :]).abs().sum(-1).float() + 0.01 * torch.rand(n, 5, generator=gen)
                cand = opts[arn, dist.argmax(1)]
            else:                                      # lazy random walk
                a = torch.randint(1, 5, (n,), generator=gen)
                cand = torch.minimum(torch.maximum(g_cur + MOVES[a], torch.zeros_like(g_cur)), lim.expand_as(g_cur))
            g_cur = torch.where(mv[:, None], cand, g_cur)
            env.g = g_cur.clone()
        env.step(acts[0], acts[1])
        hist[:, 0, t + 1] = env.ps[:, 0] * cfg.W + env.ps[:, 1]
        hist[:, 1, t + 1] = env.pr[:, 0] * cfg.W + env.pr[:, 1]
    return dict(success=env.success.clone(), tau=env.tau.clone(), n_recharge=env.n_recharge.clone())


@torch.no_grad()
def probe_target(model, K=32):
    cfg = Cfg()
    out = {}
    gen = torch.Generator().manual_seed(7002)
    s, r, g = sample_tasks(300, cfg, gen, "need")
    for p in (0.0, 0.25, 0.5):
        o = closed_loop(model, s, r, g, cfg, K, p_move=p, refresh=True, gen=torch.Generator().manual_seed(1))
        out[f"moving target p={p}, tissue told current cell"] = outcomes(o, cfg)
        print(f"target moves with p={p} per step, re-seeded each step:", out[f"moving target p={p}, tissue told current cell"], flush=True)
    o = closed_loop(model, s, r, g, cfg, K, p_move=0.25, refresh=False, gen=torch.Generator().manual_seed(1))
    out["moving target p=0.25, tissue only knows the t=0 cell"] = outcomes(o, cfg)
    print("target moves with p=0.25, tissue keeps the stale t=0 cell:", out["moving target p=0.25, tissue only knows the t=0 cell"], flush=True)
    for p in (0.5, 1.0):
        o = closed_loop(model, s, r, g, cfg, K, p_move=p, refresh=True, gen=torch.Generator().manual_seed(2), evade=True)
        out[f"evasive target, speed {p}"] = outcomes(o, cfg)
        print(f"target flees from the seeker at speed {p}:", out[f"evasive target, speed {p}"], flush=True)
    for rv in (2, 4, 6):
        o = closed_loop(model, s, r, g, cfg, K, reveal=rv)
        out[f"static target hidden until step {rv}"] = outcomes(o, cfg)
        print(f"static target hidden until step {rv}:", out[f"static target hidden until step {rv}"], flush=True)
    # no target at all: what is the default behaviour?  (free flight, open-loop dream)
    s2, r2, g2 = sample_tasks(1000, cfg, gen, "all")
    keep = manhattan(s2, r2) <= 2 * cfg.B
    s2, r2, g2 = s2[keep], r2[keep], g2[keep]
    cond = make_cond(s2, r2, g2, cfg)
    cond[:, 2] = 0
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=AMP):
        logits = torch.cat([model(cond[i:i + 128], K)[0].float() for i in range(0, len(s2), 128)])
    ro = rollout(logits, s2, r2, None, cfg, M=1)
    dsr = manhattan(s2, r2)
    moved_s = manhattan(ro["traj_s"][:, 5], s2).float()
    moved_r = manhattan(ro["traj_r"][:, 5], r2).float()
    nt = dict(n=int(len(s2)), survived_past_step_5=rate(ro["tau"] > cfg.B), alive_at_T=rate(ro["tau"] >= cfg.T),
              mean_lifetime=round(float(ro["tau"].float().mean()), 2),
              met_by_distance={int(d): rate((ro["tau"] > cfg.B)[dsr == d]) for d in sorted(dsr.unique().tolist())},
              seeker_displacement_after_5=round(float(moved_s.mean()), 2), recharger_displacement_after_5=round(float(moved_r.mean()), 2))
    out["no target shown at all (free flight)"] = nt
    print("no target shown:", nt, flush=True)
    return out


# ------------------------------------------------------------------ two seekers, one recharger
@torch.no_grad()
def probe_seekers(model, K=32):
    cfg = Cfg()
    HW, B = cfg.H * cfg.W, cfg.B
    gen = torch.Generator().manual_seed(7003)
    lim = torch.tensor([cfg.H - 1, cfg.W - 1])
    out = {}
    # geometry of the environment itself: with the current battery rule every seeker must be reached within B steps.
    idx = torch.rand(200000, HW, generator=gen).argsort(1)[:, :4]
    c = torch.stack([idx // cfg.W, idx % cfg.W], -1)
    cells_all = torch.stack([torch.arange(HW) // cfg.W, torch.arange(HW) % cfg.W], -1)

    def gather_ok(points):                                             # exists a cell every listed drone reaches within B steps
        d = torch.stack([(p[:, None, :] - cells_all[None]).abs().sum(-1) for p in points], 0)   # [k, n, HW]
        return (d <= B).all(0).any(1)
    out["P(1 seeker + recharger can gather within B)"] = rate(gather_ok([c[:, 0], c[:, 2]]))
    out["P(2 seekers + recharger can gather within B)"] = rate(gather_ok([c[:, 0], c[:, 1], c[:, 2]]))
    out["P(3 seekers + recharger can gather within B)"] = rate(gather_ok([c[:, 0], c[:, 1], c[:, 2], c[:, 3]]))
    print({k: v for k, v in out.items()}, flush=True)

    # tasks: two seekers that each need a recharge to reach a common target, all able to gather in time
    a, b, r, g = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    keep = (manhattan(a, g) > B) & (manhattan(b, g) > B) & gather_ok([a, b, r])
    a, b, r, g = (x[keep][:1200] for x in (a, b, r, g))
    n = len(a)
    arn = torch.arange(n)
    cond = make_cond(a, r, g, cfg)
    cond[arn, 0, 0, b[:, 0], b[:, 1]] = 1.0                            # second seeker in the SAME seeker channel
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=AMP):
        logits = torch.cat([model(cond[i:i + 128], K)[0].float() for i in range(0, n, 128)])

    def cand(pos):
        cc = pos[:, None, :] + MOVES[None]
        return torch.minimum(torch.maximum(cc, torch.zeros_like(cc)), lim.expand_as(cc))

    pos = [a.clone(), b.clone()]
    pr = r.clone()
    bat = [torch.full((n,), B), torch.full((n,), B)]
    done = [torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)]
    succ = [torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)]
    nre = [torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)]
    for t in range(cfg.T):
        new = []
        for i in (0, 1):
            cc = cand(pos[i])
            ai = logits[arn[:, None], 0, t + 1, cc[..., 0], cc[..., 1]].argmax(1)
            new.append(torch.where(done[i][:, None], pos[i], cc[arn, ai]))
        cc = cand(pr)
        ar_ = logits[arn[:, None], 1, t + 1, cc[..., 0], cc[..., 1]].argmax(1)
        pr = torch.where((done[0] & done[1])[:, None], pr, cc[arn, ar_])
        for i in (0, 1):
            act = ~done[i]
            pos[i] = new[i]
            bat[i] = torch.where(act, bat[i] - 1, bat[i])
            meet = act & (pos[i] == pr).all(-1)
            bat[i] = torch.where(meet, torch.full_like(bat[i], B), bat[i])
            nre[i] += meet.long()
            at = (pos[i] == g).all(-1)
            s_, f_ = act & at, act & ~at & (bat[i] <= 0)
            succ[i] |= s_
            done[i] |= s_ | f_
    both, one = succ[0] & succ[1], succ[0] ^ succ[1]
    alone_a = evaluate(model, (a, r, g), cfg, K, AMP)["success"]
    alone_b = evaluate(model, (b, r, g), cfg, K, AMP)["success"]
    dab = manhattan(a, b)
    res = dict(n=n, both=rate(both), exactly_one=rate(one), none=rate(~succ[0] & ~succ[1]),
               each_alone_with_the_recharger=rate(torch.cat([alone_a, alone_b])), both_would_succeed_alone=rate(alone_a & alone_b),
               both_by_seeker_separation={k: rate(both[m]) for k, m in (("<=2", dab <= 2), ("3-5", (dab >= 3) & (dab <= 5)), (">=6", dab >= 6))},
               share_by_separation={k: rate(m) for k, m in (("<=2", dab <= 2), ("3-5", (dab >= 3) & (dab <= 5)), (">=6", dab >= 6))},
               seekers_never_recharged=rate(torch.cat([nre[0] == 0, nre[1] == 0])))
    out["two seekers, common target (zero-shot)"] = res
    print(res, flush=True)
    return out


# ------------------------------------------------------------------ does the recharger ever let go?
@torch.no_grad()
def probe_escort(model, K=32):
    cfg = Cfg()
    gen = torch.Generator().manual_seed(7004)
    s, r, g = sample_tasks(1000, cfg, gen, "need")
    o = evaluate(model, (s, r, g), cfg, K, AMP)
    ok, tau = o["success"], o["tau"]
    ts = torch.arange(cfg.T + 1)[None]
    meet = (o["traj_s"] == o["traj_r"]).all(-1) & (ts >= 1) & (ts <= tau[:, None])
    last = (meet.long() * ts).amax(1)
    first = torch.where(meet, ts.expand_as(meet), torch.full_like(meet, 99, dtype=torch.long)).amin(1)
    gap = (tau - last)[ok]
    together = (meet.sum(1).float() / (tau - first + 1).clamp(min=1).float())[ok]
    res = dict(n_success=int(ok.sum()), first_meeting_step=round(float(first[ok].float().mean()), 2),
               recharges_per_success=round(float(meet.sum(1)[ok].float().mean()), 2),
               steps_flown_alone_at_the_end={int(k): rate(gap == k) for k in range(0, cfg.B + 1)},
               share_of_time_together_after_first_meeting=round(float(together.mean()), 3),
               remaining_distance_at_first_meeting=round(float(manhattan(o["traj_s"][torch.arange(len(s)), first.clamp(max=cfg.T)], g)[ok].float().mean()), 2))
    print(res, flush=True)
    return {"escort": res}


if __name__ == "__main__":
    model, _, a, rnd = load(sys.argv[1])
    which = sys.argv[2]
    res = dict(ckpt=sys.argv[1], round=rnd)
    res.update({"size": probe_size, "target": probe_target, "seekers": probe_seekers, "escort": probe_escort}[which](model))
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=1)
