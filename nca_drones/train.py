"""Reward-only training of the space-time NCA.

One "round":
  1. sample tasks uniformly (no curriculum, no distance features),
  2. GROW the space-time volume with the current NCA,
  3. SEARCH: follow the two grown tubes stochastically in the simulator
     (M joint rollouts per task),
  4. collect reward:
       - for the posed target (sparse 0/1),
       - in HINDSIGHT for every cell the seeker actually arrived at while its
         battery was still alive: replaying the same joint actions against that
         cell as target is, by the simulator's own rule, a rewarded episode,
  5. LEARN: train the NCA to grow the tubes of rewarded joint rollouts.

The only teacher is the simulator's 0/1 reward.  No solver, no dynamic
programming, no distance feature, no shaped reward.  The battery level is
never observed; the learner only ever sees which rollouts were rewarded.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from env import Cfg, DroneEnv, manhattan, sample_tasks
from nca import SpaceTimeNCA, make_cond

BIG = 10_000


# --------------------------------------------------------------------------
# steering from voxels
# --------------------------------------------------------------------------
def _steer(logits, tid, d, t, env, pos, eps, temp, greedy, gen):
    """One drone, one step: look at the tube logits of the reachable cells in slice t+1."""
    c = env.cand(pos)                                                   # [n,5,2]
    lg = logits[tid[:, None], d, t + 1, c[..., 0], c[..., 1]].float()   # [n,5]
    logq = F.log_softmax(lg / temp, -1)
    p = (1 - eps[:, None]) * logq.exp() + eps[:, None] / 5.0
    a = torch.multinomial(p, 1, generator=gen).squeeze(1)
    a = torch.where(greedy, lg.argmax(1), a)
    return a, logq[torch.arange(len(a)), a]


def rollout(logits, s, r, g, cfg, M=1, eps=None, temp=1.0, greedy=None, gen=None):
    """Run M simulator episodes per task against target g, steering from the grown volume.

    logits: [N, 2, T+1, H, W] tube logits (channel 0 seeker, 1 recharger).
    g=None gives target-free flight (used by the search phase for hindsight).
    """
    N = s.shape[0]
    tid = torch.arange(N).repeat_interleave(M)
    n = N * M
    env = DroneEnv(s[tid], r[tid], g[tid] if g is not None else None, cfg)
    if eps is None:
        eps = torch.zeros(n)
    if greedy is None:
        greedy = torch.ones(n, dtype=torch.bool)
    traj_s, traj_r, acts = [env.ps.clone()], [env.pr.clone()], []
    logp = torch.zeros(n)
    cum = [logp.clone()]
    ret = torch.zeros(n)
    for t in range(cfg.T):
        active = ~env.done
        a_s, lp_s = _steer(logits, tid, 0, t, env, env.ps, eps, temp, greedy, gen)
        a_r, lp_r = _steer(logits, tid, 1, t, env, env.pr, eps, temp, greedy, gen)
        logp = logp + torch.where(active, lp_s + lp_r, torch.zeros(n))
        ret = ret + env.step(a_s, a_r)
        traj_s.append(env.ps.clone()); traj_r.append(env.pr.clone())
        acts.append(torch.stack([a_s, a_r], -1)); cum.append(logp.clone())
    return dict(
        traj_s=torch.stack(traj_s, 1), traj_r=torch.stack(traj_r, 1), acts=torch.stack(acts, 1),
        success=ret > 0, tau=env.tau.clone(), logp=logp, cumlogp=torch.stack(cum, 1), tid=tid,
        n_recharge=env.n_recharge.clone(),
    )


def hindsight_table(ro, N, M, cfg):
    """For every task and every cell: the earliest rewarded arrival among its M rollouts.

    A free-flight rollout whose seeker arrives at cell c at time t <= (time its
    battery ran out) is a rewarded episode for target c (see env.replay).
    Returns best_time [N, HW] (BIG if never reached) and best_roll [N, HW]
    (index into the N*M rollouts).  Ties in time go to the rollout the tissue
    itself finds most likely.
    """
    HW = cfg.H * cfg.W
    n = N * M
    cell = ro["traj_s"][..., 0] * cfg.W + ro["traj_s"][..., 1]          # [n,T+1]
    tau = ro["tau"]                                                     # battery death (or T)
    fv = torch.full((n, HW), BIG, dtype=torch.long)                     # first-visit time
    ar = torch.arange(n)
    for t in range(cfg.T, 0, -1):
        ok = t <= tau
        fv[ar[ok], cell[ok, t]] = t
    lp = ro["cumlogp"].gather(1, fv.clamp(max=cfg.T))                   # logp of the prefix up to arrival
    score = torch.where(fv < BIG, -fv.float() * 1000.0 + lp.clamp(min=-900.0), torch.full_like(lp, -1e9))
    score = score.reshape(N, M, HW)
    best_m = score.argmax(1)                                            # [N,HW]
    best_roll = torch.arange(N)[:, None] * M + best_m
    best_time = fv.reshape(N, M, HW).gather(1, best_m[:, None]).squeeze(1)
    return best_time, best_roll, fv


def make_labels(ro, roll, tau, cfg):
    """Tube labels [k, 2, T+1] (cell index per slice) for rollouts `roll` cut at arrival time `tau`.

    After the reward both drones stay where they are (the episode is over)."""
    ts = torch.arange(cfg.T + 1)[None].expand(len(roll), -1)
    tt = torch.minimum(ts, tau[:, None])
    ps = ro["traj_s"][roll].gather(1, tt[..., None].expand(-1, -1, 2))
    pr = ro["traj_r"][roll].gather(1, tt[..., None].expand(-1, -1, 2))
    return torch.stack([ps[..., 0] * cfg.W + ps[..., 1], pr[..., 0] * cfg.W + pr[..., 1]], 1)


def tube_loss(logits, labels, cfg):
    """Per-time-slice cross-entropy: each slice is a distribution over cells."""
    n = logits.shape[0]
    lg = logits.float().reshape(n, 2, cfg.T + 1, cfg.H * cfg.W)[:, :, 1:]
    return F.cross_entropy(lg.reshape(-1, cfg.H * cfg.W), labels[:, :, 1:].reshape(-1))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def task_key(s, r, g, cfg):
    hw = cfg.H * cfg.W
    return ((s[:, 0] * cfg.W + s[:, 1]) * hw + (r[:, 0] * cfg.W + r[:, 1])) * hw + (g[:, 0] * cfg.W + g[:, 1])


def grow(model, s, r, g, cfg, K, amp=False, noise=0.0, bs=128):
    outs = []
    for i in range(0, s.shape[0], bs):
        cond = make_cond(s[i:i + bs], r[i:i + bs], g[i:i + bs], cfg)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(amp)):
            lg, _ = model(cond, K, noise=noise)
        outs.append(lg.float())
    return torch.cat(outs)


@torch.no_grad()
def evaluate(model, tasks, cfg, K, amp=False):
    """Greedy open-loop readout of both tubes, executed in the simulator against the real target."""
    model.eval()
    s, r, g = tasks
    logits = grow(model, s, r, g, cfg, K, amp)
    out = rollout(logits, s, r, g, cfg, M=1)
    model.train()
    return out


class Buffer:
    """FIFO replay of rewarded joint rollouts."""

    def __init__(self, cap, cfg):
        self.cap = cap
        self.s = torch.zeros(cap, 2, dtype=torch.long)
        self.r = torch.zeros(cap, 2, dtype=torch.long)
        self.g = torch.zeros(cap, 2, dtype=torch.long)
        self.lab = torch.zeros(cap, 2, cfg.T + 1, dtype=torch.long)
        self.hard = torch.zeros(cap, dtype=torch.bool)   # search found reward where the greedy readout got none
        self.meta = torch.zeros(cap, 3, dtype=torch.long)  # bookkeeping for logs only: tau, needs-recharge, source
        self.n = 0
        self.ptr = 0

    def add(self, s, r, g, lab, hard, meta):
        k = s.shape[0]
        idx = (self.ptr + torch.arange(k)) % self.cap
        self.s[idx], self.r[idx], self.g[idx], self.lab[idx], self.hard[idx], self.meta[idx] = s, r, g, lab, hard, meta
        self.ptr = int((self.ptr + k) % self.cap)
        self.n = min(self.n + k, self.cap)

    def sample(self, bs, gen, hard_frac=None):
        """Uniform replay, optionally with a fixed share of 'search beat the tissue' samples.

        The priority flag is defined by reward alone: the greedy readout earned 0,
        a sampled rollout earned 1."""
        hard = self.hard[: self.n]
        if hard_frac is not None and hard.any() and (~hard).any():
            nh = int(round(bs * hard_frac))
            idx_h = torch.nonzero(hard).squeeze(1)
            idx_e = torch.nonzero(~hard).squeeze(1)
            a = idx_h[torch.randint(len(idx_h), (nh,), generator=gen)]
            b = idx_e[torch.randint(len(idx_e), (bs - nh,), generator=gen)]
            idx = torch.cat([a, b])
        else:
            idx = torch.randint(self.n, (bs,), generator=gen)
        return self.s[idx], self.r[idx], self.g[idx], self.lab[idx]


def cell_of(p, cfg):
    return p[:, 0] * cfg.W + p[:, 1]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--update", default="res")
    ap.add_argument("--fire_rate", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--K_jitter", type=int, default=4)
    ap.add_argument("--N", type=int, default=64, help="tasks per round")
    ap.add_argument("--M", type=int, default=64, help="rollouts per task")
    ap.add_argument("--G", type=int, default=4, help="gradient steps per round")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lr_final", type=float, default=1.0, help="cosine-decay the lr to this fraction")
    ap.add_argument("--buf", type=int, default=20000)
    ap.add_argument("--eps_max", type=float, default=0.4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--B", type=int, default=5)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--hind", type=int, default=2, help="hindsight targets added per task and round (0 = off)")
    ap.add_argument("--hard_frac", type=float, default=-1.0, help="share of each batch drawn from samples where search beat the greedy readout")
    ap.add_argument("--taps", type=int, default=0, help="deep supervision: also apply the loss this many iterations before the end")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = Cfg(T=args.T, B=args.B)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "log.jsonl"), "a")
    json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"))

    # held-out sets (never trained on): validation for monitoring, test for the final claim
    val_need = sample_tasks(200, cfg, torch.Generator().manual_seed(999), "need")
    val_easy = sample_tasks(100, cfg, torch.Generator().manual_seed(998), "easy")
    test_need = sample_tasks(200, cfg, torch.Generator().manual_seed(12345), "need")
    held = torch.cat([task_key(*val_need, cfg), task_key(*test_need, cfg), task_key(*val_easy, cfg)]).unique()

    model = SpaceTimeNCA(C=args.C, hid=args.hid, update=args.update, fire_rate=args.fire_rate)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_round = 0
    if args.resume:
        ck = torch.load(args.resume)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_round = ck.get("round", 0)
    buf = Buffer(args.buf, cfg)
    print(f"params: {sum(p.numel() for p in model.parameters())}", flush=True)

    # exploration levels for the M rollouts of each task; rollout 0 is the greedy readout
    N, M = args.N, args.M
    eps = torch.linspace(0, args.eps_max, M).repeat(N)
    greedy = torch.zeros(N * M, dtype=torch.bool)
    greedy[::M] = True
    arN = torch.arange(N)

    t0 = time.time()
    ema = {}
    for rnd in range(start_round, args.rounds):
        t_r0 = time.time()
        if args.lr_final < 1.0:
            frac = rnd / max(1, args.rounds - 1)
            for pg in opt.param_groups:
                pg["lr"] = args.lr * (args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * frac)))

        # ---------------- search -----------------
        s, r, g = sample_tasks(N * 2, cfg, gen, "all")
        keep = ~torch.isin(task_key(s, r, g, cfg), held)
        s, r, g = s[keep][:N], r[keep][:N], g[keep][:N]
        need = manhattan(s, g) > cfg.B
        with torch.no_grad():
            logits = grow(model, s, r, g, cfg, args.K, args.amp)
            ro = rollout(logits, s, r, None, cfg, M=M, eps=eps, temp=args.temp, greedy=greedy, gen=gen)
        best_time, best_roll, fv = hindsight_table(ro, N, M, cfg)
        gc = cell_of(g, cfg)
        # (a) the posed task
        any_succ = best_time[arN, gc] < BIG
        greedy_succ = fv[arN * M, gc] < BIG                 # rollout 0 of each task is the greedy readout
        if any_succ.any():
            i = torch.nonzero(any_succ).squeeze(1)
            tau_i = best_time[i, gc[i]]
            lab = make_labels(ro, best_roll[i, gc[i]], tau_i, cfg)
            meta = torch.stack([tau_i, need[i].long(), torch.zeros_like(tau_i)], 1)
            buf.add(s[i], r[i], g[i], lab, ~greedy_succ[i], meta)
        # (b) hindsight: cells the seeker reached alive are rewarded targets for the same joint actions
        n_h = 0
        if args.hind > 0:
            reach = best_time < BIG
            reach[arN, gc] = False
            reach[arN, cell_of(s, cfg)] = False
            reach[arN, cell_of(r, cfg)] = False             # tasks have three distinct cells
            w = reach.float() + 1e-9
            pick = torch.multinomial(w, args.hind, generator=gen)       # [N,hind] cells, uniform over reached
            ti = arN[:, None].expand(-1, args.hind).reshape(-1)
            ci = pick.reshape(-1)
            ok = reach[ti, ci]
            ti, ci = ti[ok], ci[ok]
            hg = torch.stack([ci // cfg.W, ci % cfg.W], -1)
            okh = ~torch.isin(task_key(s[ti], r[ti], hg, cfg), held)   # never train on a held-out task
            ti, ci, hg = ti[okh], ci[okh], hg[okh]
            if len(ti):
                tau_h = best_time[ti, ci]
                lab = make_labels(ro, best_roll[ti, ci], tau_h, cfg)
                h_need = manhattan(s[ti], hg) > cfg.B
                meta = torch.stack([tau_h, h_need.long(), torch.ones_like(tau_h)], 1)
                buf.add(s[ti], r[ti], hg, lab, torch.zeros(len(ti), dtype=torch.bool), meta)
                n_h = len(ti)
        t_search = time.time() - t_r0

        stats = dict(
            round=rnd,
            search_easy=float(any_succ[~need].float().mean()) if (~need).any() else float("nan"),
            search_need=float(any_succ[need].float().mean()) if need.any() else float("nan"),
            greedy_easy=float(greedy_succ[~need].float().mean()) if (~need).any() else float("nan"),
            greedy_need=float(greedy_succ[need].float().mean()) if need.any() else float("nan"),
            buf=buf.n, buf_need=float(buf.meta[: buf.n, 1].float().mean()) if buf.n else 0.0,
            buf_hard=float(buf.hard[: buf.n].float().mean()) if buf.n else 0.0,
            buf_tau=float(buf.meta[: buf.n, 0].float().mean()) if buf.n else 0.0,
            n_hind=n_h, alive_T=float(ro["tau"].float().mean()), t_search=t_search,
        )

        # ---------------- learn ------------------
        losses = []
        if buf.n >= args.bs:
            for _ in range(args.G):
                bs_, br_, bg_, blab = buf.sample(args.bs, gen, args.hard_frac if args.hard_frac >= 0 else None)
                cond = make_cond(bs_, br_, bg_, cfg)
                K = args.K + int(torch.randint(-args.K_jitter, args.K_jitter + 1, (1,), generator=gen))
                taps = (K - args.taps,) if args.taps > 0 else ()
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(args.amp)):
                    res = model(cond, K, taps=taps)
                loss = tube_loss(res[0], blab, cfg)
                for tl in (res[2] if taps else ()):
                    loss = loss + 0.3 * tube_loss(tl, blab, cfg)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
        stats["loss"] = sum(losses) / len(losses) if losses else float("nan")
        stats["time"] = time.time() - t0
        for k in ("search_easy", "search_need", "greedy_easy", "greedy_need", "loss"):
            v = stats[k]
            if v == v:
                ema[k] = v if k not in ema else 0.95 * ema[k] + 0.05 * v

        if rnd % args.eval_every == 0 or rnd == args.rounds - 1:
            ev = evaluate(model, val_need, cfg, args.K, args.amp)
            ee = evaluate(model, val_easy, cfg, args.K, args.amp)
            stats["val_need"] = float(ev["success"].float().mean())
            stats["val_easy"] = float(ee["success"].float().mean())
            print(f"[{rnd:5d}] t={stats['time']:.0f}s loss={ema.get('loss', float('nan')):.3f} "
                  f"search e/n={ema.get('search_easy', 0):.2f}/{ema.get('search_need', 0):.2f} "
                  f"greedy e/n={ema.get('greedy_easy', 0):.2f}/{ema.get('greedy_need', 0):.2f} "
                  f"VAL easy={stats['val_easy']:.2f} need={stats['val_need']:.3f} "
                  f"buf={buf.n} need={stats['buf_need']:.2f} hard={stats['buf_hard']:.2f} tau={stats['buf_tau']:.1f} alive={stats['alive_T']:.1f}",
                  flush=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), round=rnd + 1, args=vars(args)),
                       os.path.join(out_dir, "ckpt.pt"))
            if stats["val_need"] >= ema.get("best_val", -1):
                ema["best_val"] = stats["val_need"]
                torch.save(dict(model=model.state_dict(), round=rnd + 1, args=vars(args), val_need=stats["val_need"]),
                           os.path.join(out_dir, "best.pt"))
        logf.write(json.dumps(stats) + "\n")
        logf.flush()

    ev = evaluate(model, test_need, cfg, args.K, args.amp)
    print(f"FINAL test_need success = {float(ev['success'].float().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
