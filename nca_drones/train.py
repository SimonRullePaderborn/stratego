"""Reward-only training of the space-time NCA.

One "round" (the audited path, --strict 1):
  1. sample tasks uniformly (no curriculum, no distance features),
  2. GROW the space-time volume with the current NCA from the t=0 observation,
  3. SEARCH: follow the two grown tubes stochastically in the simulator
     (M joint rollouts per task against the posed target),
  4. REWARD: the simulator's 0/1 reward says which rollouts worked; per task the
     earliest rewarded rollout is kept (ties: the one the tissue finds most likely),
  5. CREDIT: counterfactual replays in the simulator (credit.py) say how much the
     reward depended on each drone's own behaviour,
  6. LEARN: train the NCA to grow the tubes of rewarded joint rollouts.

The only teacher is the simulator's 0/1 reward.  No solver, no dynamic
programming, no distance feature, no shaped reward.  The battery level is
never observed.

Older experimental paths are kept for the record and are OFF in the audited
configuration: hindsight relabelling (--hind), label pruning (--prune 1,
prune.py), cross-play (--cross).  Manhattan distance and B appear in this file
only to label log lines (`need`) and never reach selection, weights, replay
priorities or the network.
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
from credit import null_partner_credit
from prune import credit_batch, prune_batch, verify_labels

BIG = 10_000


# --------------------------------------------------------------------------
# steering from voxels
# --------------------------------------------------------------------------
def rollout(logits, s, r, g, cfg, M=1, eps=None, temp=1.0, greedy=None, gen=None, ez=0, shared=None, kinematic=False):
    """Run M simulator episodes per task against target g, steering from the grown volume.

    logits: [N, 2, T+1, H, W] tube logits (channel 0 seeker, 1 recharger).
    At time t a drone on cell p looks at the cells it can reach and moves to one
    of them according to the tube logits in slice t+1.
    g=None gives target-free flight (used by the search phase for hindsight).
    kinematic=True only SAMPLES the two worldlines for all T steps (no battery,
    no termination); they are judged afterwards by the simulator (see crossplay).

    Exploration (search phase only; evaluation is greedy):
      eps    [n]  probability per step of starting a random excursion (a random
                  action held for 1..ez steps; ez=0 -> a single random action),
      temp   [n]  sampling temperature,
      shared [n]  bool: the sampling noise is a property of the VOXEL, not of the
                  drone - both drones read the same random field G(t,x,y) and the
                  same excursion draws.  Two drones on the same cell whose tubes
                  agree then make the same move, so they stay together.
    """
    N = s.shape[0]
    tid = torch.arange(N).repeat_interleave(M)
    n = N * M
    arn = torch.arange(n)
    env_cfg = Cfg(H=cfg.H, W=cfg.W, T=cfg.T, B=10 ** 6) if kinematic else cfg
    env = DroneEnv(s[tid], r[tid], g[tid] if g is not None else None, env_cfg)
    if eps is None:
        eps = torch.zeros(n)
    if greedy is None:
        greedy = torch.ones(n, dtype=torch.bool)
    if shared is None:
        shared = torch.zeros(n, dtype=torch.bool)
    if not torch.is_tensor(temp):
        temp = torch.full((n,), float(temp))
    all_greedy = bool(greedy.all())
    zmax = max(ez, 1)
    ex_act = torch.zeros(n, 2, dtype=torch.long)
    ex_left = torch.zeros(n, 2, dtype=torch.long)
    traj_s, traj_r, acts = [env.ps.clone()], [env.pr.clone()], []
    logp = torch.zeros(n)
    cum = [logp.clone()]
    lpd = torch.zeros(n, 2)
    cumd = [lpd.clone()]
    ret = torch.zeros(n)

    def gumbel(*shape):
        u = torch.rand(*shape, generator=gen).clamp_(1e-9, 1 - 1e-9)
        return -torch.log(-torch.log(u))

    for t in range(cfg.T):
        active = ~env.done
        step_acts = []
        lp_t = torch.zeros(n)
        if not all_greedy:
            G = gumbel(n, cfg.H, cfg.W)
            u = torch.rand(n, 2, generator=gen)
            na = torch.randint(0, 5, (n, 2), generator=gen)
            nl = torch.randint(1, zmax + 1, (n, 2), generator=gen)
            u[:, 1] = torch.where(shared, u[:, 0], u[:, 1])
            na[:, 1] = torch.where(shared, na[:, 0], na[:, 1])
            nl[:, 1] = torch.where(shared, nl[:, 0], nl[:, 1])
        for d, pos in ((0, env.ps), (1, env.pr)):
            c = env.cand(pos)                                                   # [n,5,2] reachable cells
            lg = logits[tid[:, None], d, t + 1, c[..., 0], c[..., 1]].float()   # [n,5] their tube logits
            a = lg.argmax(1)
            logq = F.log_softmax(lg / temp[:, None], -1)
            if not all_greedy:
                gs = G[arn[:, None], c[..., 0], c[..., 1]]
                noise = torch.where(shared[:, None], gs, gumbel(n, 5))
                a_s = (logq + noise).argmax(1)
                start = (u[:, d] < eps) & (ex_left[:, d] == 0)
                ex_act[:, d] = torch.where(start, na[:, d], ex_act[:, d])
                ex_left[:, d] = torch.where(start, nl[:, d], ex_left[:, d])
                a_s = torch.where(ex_left[:, d] > 0, ex_act[:, d], a_s)
                ex_left[:, d] = (ex_left[:, d] - 1).clamp(min=0)
                a = torch.where(greedy, a, a_s)
            lp_t = lp_t + logq[arn, a]
            lpd[:, d] = lpd[:, d] + logq[arn, a]
            step_acts.append(a)
        logp = logp + torch.where(active, lp_t, torch.zeros(n))
        ret = ret + env.step(step_acts[0], step_acts[1])
        traj_s.append(env.ps.clone()); traj_r.append(env.pr.clone())
        acts.append(torch.stack(step_acts, -1)); cum.append(logp.clone()); cumd.append(lpd.clone())
    return dict(
        cumlogp_d=torch.stack(cumd, 1),
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


def crossplay(cs, cr, gcell, cfg):
    """Judge every (seeker worldline i, recharger worldline j) pair of a task.

    cs, cr: [N, M, T+1] cell indices of M sampled worldlines per drone; gcell [N].
    Returns arrival [N, M, M]: the time at which the pair is rewarded, BIG if never.
    This is a batched mirror of DroneEnv.step (tested against it); the pair that is
    finally imitated is re-checked by an authoritative replay in the real simulator.
    """
    N, M, T1 = cs.shape
    bat = torch.full((N, M, M), cfg.B, dtype=torch.long)
    alive = torch.ones(N, M, M, dtype=torch.bool)
    arrival = torch.full((N, M, M), BIG, dtype=torch.long)
    for t in range(1, T1):
        meet = cs[:, :, None, t] == cr[:, None, :, t]
        at_goal = (cs[:, :, t] == gcell[:, None])[:, :, None].expand(-1, -1, M)
        bat = bat - 1
        bat = torch.where(meet, torch.full_like(bat, cfg.B), bat)
        succ = alive & at_goal
        arrival = torch.where(succ, torch.full_like(arrival, t), arrival)
        alive = alive & ~at_goal & (bat > 0) & (t < cfg.T)
    return arrival


def make_labels(ro, roll, tau, cfg):
    """Tube labels [k, 2, T+1] (cell index per slice) for rollouts `roll` cut at arrival time `tau`.

    After the reward both drones stay where they are (the episode is over)."""
    ts = torch.arange(cfg.T + 1)[None].expand(len(roll), -1)
    tt = torch.minimum(ts, tau[:, None])
    ps = ro["traj_s"][roll].gather(1, tt[..., None].expand(-1, -1, 2))
    pr = ro["traj_r"][roll].gather(1, tt[..., None].expand(-1, -1, 2))
    return torch.stack([ps[..., 0] * cfg.W + ps[..., 1], pr[..., 0] * cfg.W + pr[..., 1]], 1)


def tube_loss(logits, labels, cfg, w=None):
    """Per-time-slice cross-entropy: each slice is a distribution over cells.

    w [n, 2, T+1]: per-slice credit weights (see prune.prune_batch)."""
    n = logits.shape[0]
    lg = logits.float().reshape(n, 2, cfg.T + 1, cfg.H * cfg.W)[:, :, 1:]
    ce = F.cross_entropy(lg.reshape(-1, cfg.H * cfg.W), labels[:, :, 1:].reshape(-1), reduction="none")
    if w is None:
        return ce.mean()
    w = w[:, :, 1:].reshape(-1)
    return (ce * w).sum() / w.sum().clamp(min=1.0)


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


@torch.no_grad()
def evaluate_closed(model, tasks, cfg, K, amp=False, stride=1, bs=256):
    """Closed-loop readout: the environment is pushed through the dreamed volume.

    At every step the positions the simulator actually produced are burned into
    the past slices, the tissue regrows the future from scratch, and both
    drones take the greedy step along their tubes.  The network never sees the
    battery, the reward or a done flag - only where the two drones have been.
    """
    model.eval()
    s, r, g = tasks
    n = s.shape[0]
    env = DroneEnv(s, r, g, cfg)
    hist = torch.zeros(n, 2, cfg.T + 1, dtype=torch.long)
    hist[:, 0, 0] = s[:, 0] * cfg.W + s[:, 1]
    hist[:, 1, 0] = r[:, 0] * cfg.W + r[:, 1]
    logits = None
    for t in range(cfg.T):
        act = torch.nonzero(~env.done).squeeze(1)
        if len(act) == 0:
            break
        if t % stride == 0:
            logits = torch.zeros(n, 2, cfg.T + 1, cfg.H, cfg.W)
            for i in range(0, len(act), bs):
                j = act[i:i + bs]
                cond = make_cond(s[j], r[j], g[j], cfg, hist[j], torch.full((len(j),), t))
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(amp)):
                    lg, _ = model(cond, K)
                logits[j] = lg.float()
        acts = []
        for d, pos in ((0, env.ps), (1, env.pr)):
            c = env.cand(pos)
            lg = logits[torch.arange(n)[:, None], d, t + 1, c[..., 0], c[..., 1]]
            acts.append(lg.argmax(1))
        env.step(acts[0], acts[1])
        hist[:, 0, t + 1] = env.ps[:, 0] * cfg.W + env.ps[:, 1]
        hist[:, 1, t + 1] = env.pr[:, 0] * cfg.W + env.pr[:, 1]
    model.train()
    return dict(success=env.success.clone(), tau=env.tau.clone(), n_recharge=env.n_recharge.clone(), hist=hist)


class Buffer:
    """FIFO replay of rewarded joint rollouts."""

    def __init__(self, cap, cfg):
        self.cap = cap
        self.s = torch.zeros(cap, 2, dtype=torch.long)
        self.r = torch.zeros(cap, 2, dtype=torch.long)
        self.g = torch.zeros(cap, 2, dtype=torch.long)
        self.lab = torch.zeros(cap, 2, cfg.T + 1, dtype=torch.long)
        self.w = torch.ones(cap, 2, cfg.T + 1)
        self.hard = torch.zeros(cap, dtype=torch.bool)   # search found reward where the greedy readout got none
        self.meta = torch.zeros(cap, 3, dtype=torch.long)  # bookkeeping for logs only: tau, needs-recharge, source
        self.n = 0
        self.ptr = 0

    def add(self, s, r, g, lab, hard, meta, w=None):
        k = s.shape[0]
        idx = (self.ptr + torch.arange(k)) % self.cap
        self.s[idx], self.r[idx], self.g[idx], self.lab[idx], self.hard[idx], self.meta[idx] = s, r, g, lab, hard, meta
        self.w[idx] = w if w is not None else 1.0
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
        return self.s[idx], self.r[idx], self.g[idx], self.lab[idx], self.w[idx]


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
    ap.add_argument("--prune", type=int, default=0, help="1: simulator-verified pruning of rewarded rollouts; 2: no label editing, credit weights only")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--strict", type=int, default=0, help="audited path: search against the posed target only, reward straight from the simulator's step, "
                    "credit by null-partner replays in the real simulator (no hindsight, no label editing, no learner-side copy of the battery rule)")
    ap.add_argument("--cross", type=int, default=0, help="cross-play search: judge all (seeker worldline, recharger worldline) pairs of a task in the simulator")
    ap.add_argument("--ema", type=float, default=0.0, help="decay of the weight average used for search and evaluation (0 = use the raw weights)")
    ap.add_argument("--prefix", type=float, default=0.0, help="share of training samples seeded with a random-length realised past (the tissue regrows the future)")
    ap.add_argument("--eval_closed", type=int, default=0, help="also evaluate the closed-loop readout on this many held-out recharge-needed tasks")
    ap.add_argument("--null_k", type=int, default=0, help="null-partner test: train each drone only on what the reward needed (0 = off)")
    ap.add_argument("--post_w", type=float, default=0.0, help="loss weight of slices after the reward (only with --null_k)")
    ap.add_argument("--shared", type=float, default=0.0, help="share of rollouts whose sampling noise is a shared random medium on the voxels")
    ap.add_argument("--ez", type=int, default=0, help="temporally-extended exploration: max length of a random excursion (0 = single-step noise)")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    cfg = Cfg(T=args.T, B=args.B)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
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
        model.load_state_dict(ck.get("raw", ck["model"]))
        opt.load_state_dict(ck["opt"])
        start_round = ck.get("round", 0)
    import copy
    ema_model = copy.deepcopy(model) if args.ema > 0 else model
    if args.resume and args.ema > 0 and "ema" in ck:
        ema_model.load_state_dict(ck["ema"])
    posed = torch.zeros((cfg.H * cfg.W) ** 3, dtype=torch.bool)      # every task ever posed to the learner (for strictly disjoint evaluation)
    if args.resume and "posed" in ck:
        posed = ck["posed"]
    buf = Buffer(args.buf, cfg)
    print(f"params: {sum(p.numel() for p in model.parameters())}", flush=True)

    # exploration levels for the M rollouts of each task; rollout 0 is the greedy readout
    N, M = args.N, args.M
    n_sh = int(round(args.shared * (M - 1)))
    n_in = M - 1 - n_sh
    eps_m = torch.cat([torch.zeros(1), torch.linspace(0.02, args.eps_max, n_in), torch.linspace(0.0, args.eps_max, n_sh)])
    temp_m = torch.cat([torch.ones(1 + n_in) * args.temp, torch.tensor([1.0, 2.0, 3.0]).repeat(n_sh // 3 + 1)[:n_sh]])
    shared_m = torch.cat([torch.zeros(1 + n_in, dtype=torch.bool), torch.ones(n_sh, dtype=torch.bool)])
    eps, temp_v, shared_v = eps_m.repeat(N), temp_m.repeat(N), shared_m.repeat(N)
    greedy = torch.zeros(N * M, dtype=torch.bool)
    greedy[::M] = True
    arN = torch.arange(N)
    import random
    rng = random.Random(args.seed)

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
        need = manhattan(s, g) > cfg.B                      # for logs only
        posed[task_key(s, r, g, cfg)] = True
        with torch.no_grad():
            logits = grow(ema_model, s, r, g, cfg, args.K, args.amp)
            ro = rollout(logits, s, r, g if args.strict else None, cfg, M=M, eps=eps, temp=temp_v, greedy=greedy, gen=gen, ez=args.ez,
                         shared=shared_v, kinematic=bool(args.cross))
        gc = cell_of(g, cfg)
        pr_stats = []
        xstats = {}
        if args.strict:
            # the simulator ran every rollout against the posed target; its reward is all we use
            succ = ro["success"].reshape(N, M)
            tau_m = ro["tau"].reshape(N, M)
            score = torch.where(succ, -tau_m.float() * 1000.0 + ro["logp"].reshape(N, M).clamp(min=-900.0), torch.full((N, M), -1e9))
            best = score.argmax(1)                              # earliest reward; ties: most likely under the tissue
            any_succ = succ.any(1)
            greedy_succ = succ[:, 0]                            # rollout 0 is the greedy readout
            if any_succ.any():
                i = torch.nonzero(any_succ).squeeze(1)
                sel = i * M + best[i]
                tau_i = ro["tau"][sel]
                lab = torch.stack([ro["traj_s"][sel][..., 0] * cfg.W + ro["traj_s"][sel][..., 1],
                                   ro["traj_r"][sel][..., 0] * cfg.W + ro["traj_r"][sel][..., 1]], 1)      # the rollout exactly as it happened
                if args.null_k > 0:
                    w_s, w_r = null_partner_credit(s[i], r[i], g[i], ro["acts"][sel], cfg, args.null_k, gen)
                else:
                    w_s = w_r = torch.ones(len(i))
                ts = torch.arange(cfg.T + 1)[None]
                live = ((ts >= 1) & (ts <= tau_i[:, None])).float()                                        # slices up to the reward
                wts = torch.stack([live * w_s[:, None], live * w_r[:, None]], 1) + args.post_w * (ts > tau_i[:, None]).float()[:, None]
                pr_stats.append((tau_i.float().mean(), tau_i.float().mean(), w_s.mean(),
                                 w_s[need[i]].mean() if need[i].any() else torch.tensor(float("nan")),
                                 w_r.mean(), w_r[need[i]].mean() if need[i].any() else torch.tensor(float("nan"))))
                meta = torch.stack([tau_i, need[i].long(), torch.zeros_like(tau_i)], 1)
                buf.add(s[i], r[i], g[i], lab, ~greedy_succ[i], meta, wts)
            best_time = fv = None
        elif args.cross:
            # every sampled seeker worldline against every sampled recharger worldline, judged by the simulator
            cs = (ro["traj_s"][..., 0] * cfg.W + ro["traj_s"][..., 1]).reshape(N, M, cfg.T + 1)
            cr = (ro["traj_r"][..., 0] * cfg.W + ro["traj_r"][..., 1]).reshape(N, M, cfg.T + 1)
            arr = crossplay(cs, cr, gc, cfg)                                    # [N,M,M]
            lps = ro["cumlogp_d"][..., 0].reshape(N, M, cfg.T + 1)
            lpr = ro["cumlogp_d"][..., 1].reshape(N, M, cfg.T + 1)
            ta = arr.clamp(max=cfg.T)
            lp_i = torch.gather(lps[:, :, None, :].expand(-1, -1, M, -1), 3, ta[..., None]).squeeze(-1)
            lp_j = torch.gather(lpr[:, None, :, :].expand(-1, M, -1, -1), 3, ta[..., None]).squeeze(-1)
            score = torch.where(arr < BIG, -arr.float() * 1000.0 + (lp_i + lp_j).clamp(min=-900.0), torch.full((N, M, M), -1e9))
            flat = score.reshape(N, -1).argmax(1)
            bi, bj = flat // M, flat % M
            tau_x = arr[arN, bi, bj]
            any_succ = tau_x < BIG
            greedy_succ = arr[:, 0, 0] < BIG                                    # pair (0,0) is the greedy readout of both tubes
            diag = arr.diagonal(dim1=1, dim2=2) < BIG
            xstats = dict(search_diag_need=float(diag.any(1)[need].float().mean()) if need.any() else float("nan"),
                          uni_s=float(((arr[:, :, 0] < BIG).any(1) & ~greedy_succ)[need].float().mean()) if need.any() else float("nan"),
                          uni_r=float(((arr[:, 0, :] < BIG).any(1) & ~greedy_succ)[need].float().mean()) if need.any() else float("nan"))
            if any_succ.any():
                i = torch.nonzero(any_succ).squeeze(1)
                lab, wts, tau_i, stop, rel = credit_batch(cs[i, bi[i]], cr[i, bj[i]], tau_x[i], gc[i], cfg, args.null_k, rng, args.post_w,
                                                          edit=args.prune == 1)
                assert bool(verify_labels(s[i], r[i], g[i], lab, tau_i, cfg).all()), "cross-play label rejected by the simulator"
                pr_stats.append((tau_x[i].float().mean(), tau_i.float().mean(), (stop > 0).float().mean(),
                                 (stop[need[i]] > 0).float().mean() if need[i].any() else torch.tensor(float("nan")),
                                 rel.mean(), rel[need[i]].mean() if need[i].any() else torch.tensor(float("nan"))))
                meta = torch.stack([tau_i, need[i].long(), torch.zeros_like(tau_i)], 1)
                buf.add(s[i], r[i], g[i], lab, ~greedy_succ[i], meta, wts)
            best_time = fv = None
        else:
            best_time, best_roll, fv = hindsight_table(ro, N, M, cfg)
            any_succ = best_time[arN, gc] < BIG
            greedy_succ = fv[arN * M, gc] < BIG             # rollout 0 of each task is the greedy readout
        # (a) the posed task
        if not args.cross and not args.strict and any_succ.any():
            i = torch.nonzero(any_succ).squeeze(1)
            tau_i = best_time[i, gc[i]]
            wts = None
            if args.prune:
                lab, wts, tau_p, stop, rel = prune_batch(ro, best_roll[i, gc[i]], tau_i, gc[i], cfg, args.null_k, rng, args.post_w, edit=args.prune == 1)
                assert bool(verify_labels(s[i], r[i], g[i], lab, tau_p, cfg).all())
                pr_stats.append((tau_i.float().mean(), tau_p.float().mean(), (stop > 0).float().mean(),
                                 (stop[need[i]] > 0).float().mean() if need[i].any() else torch.tensor(float("nan")),
                                 rel.mean(), rel[need[i]].mean() if need[i].any() else torch.tensor(float("nan"))))
                tau_i = tau_p
            else:
                lab = make_labels(ro, best_roll[i, gc[i]], tau_i, cfg)
            meta = torch.stack([tau_i, need[i].long(), torch.zeros_like(tau_i)], 1)
            buf.add(s[i], r[i], g[i], lab, ~greedy_succ[i], meta, wts)
        # (b) hindsight: cells the seeker reached alive are rewarded targets for the same joint actions
        n_h = 0
        if args.hind > 0 and not args.cross and not args.strict:
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
                wts_h = None
                if args.prune:
                    lab, wts_h, tau_h, stop_h, rel_h = prune_batch(ro, best_roll[ti, ci], tau_h, ci, cfg, args.null_k, rng, args.post_w, edit=args.prune == 1)
                    assert bool(verify_labels(s[ti], r[ti], hg, lab, tau_h, cfg).all())
                else:
                    lab = make_labels(ro, best_roll[ti, ci], tau_h, cfg)
                h_need = manhattan(s[ti], hg) > cfg.B
                meta = torch.stack([tau_h, h_need.long(), torch.ones_like(tau_h)], 1)
                buf.add(s[ti], r[ti], hg, lab, torch.zeros(len(ti), dtype=torch.bool), meta, wts_h)
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
        stats.update(xstats)
        if pr_stats:
            stats.update(tau_raw=float(pr_stats[0][0]), tau_pruned=float(pr_stats[0][1]),
                         rech_moves=float(pr_stats[0][2]), rech_moves_need=float(pr_stats[0][3]),
                         rel=float(pr_stats[0][4]), rel_need=float(pr_stats[0][5]))

        # ---------------- learn ------------------
        losses = []
        if buf.n >= args.bs:
            for _ in range(args.G):
                bs_, br_, bg_, blab, bw = buf.sample(args.bs, gen, args.hard_frac if args.hard_frac >= 0 else None)
                if args.prefix > 0:
                    # burn a random-length prefix of the rewarded rollout into the past; learn to regrow the rest
                    w_any = (bw[:, 0] > 0).float()
                    tau_b = (w_any * torch.arange(cfg.T + 1)[None]).amax(1).long().clamp(min=1)     # last credited seeker slice
                    use = torch.rand(args.bs, generator=gen) < args.prefix
                    tp = (torch.rand(args.bs, generator=gen) * tau_b.float()).long()                # uniform in [0, tau-1]
                    tp = torch.where(use, tp, torch.zeros_like(tp))
                    cond = make_cond(bs_, br_, bg_, cfg, blab, tp)
                    bw = bw * (torch.arange(cfg.T + 1)[None, None] > tp[:, None, None]).float()
                else:
                    cond = make_cond(bs_, br_, bg_, cfg)
                K = args.K + int(torch.randint(-args.K_jitter, args.K_jitter + 1, (1,), generator=gen))
                taps = (K - args.taps,) if args.taps > 0 else ()
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(args.amp)):
                    res = model(cond, K, taps=taps)
                loss = tube_loss(res[0], blab, cfg, bw)
                for tl in (res[2] if taps else ()):
                    loss = loss + 0.3 * tube_loss(tl, blab, cfg, bw)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if args.ema > 0:
                    with torch.no_grad():
                        for pe, p in zip(ema_model.parameters(), model.parameters()):
                            pe.mul_(args.ema).add_(p, alpha=1 - args.ema)
                losses.append(loss.item())
        stats["loss"] = sum(losses) / len(losses) if losses else float("nan")
        stats["time"] = time.time() - t0
        for k in ("search_easy", "search_need", "greedy_easy", "greedy_need", "loss"):
            v = stats[k]
            if v == v:
                ema[k] = v if k not in ema else 0.95 * ema[k] + 0.05 * v

        if rnd % args.eval_every == 0 or rnd == args.rounds - 1:
            ev = evaluate(ema_model, val_need, cfg, args.K, args.amp)
            ee = evaluate(ema_model, val_easy, cfg, args.K, args.amp)
            stats["val_need"] = float(ev["success"].float().mean())
            stats["val_easy"] = float(ee["success"].float().mean())
            if args.eval_closed > 0:
                nc = args.eval_closed
                ec = evaluate_closed(ema_model, tuple(x[:nc] for x in val_need), cfg, args.K, args.amp)
                stats["val_need_closed"] = float(ec["success"].float().mean())
                stats["val_need_open_sub"] = float(ev["success"][:nc].float().mean())
            print(f"[{rnd:5d}] t={stats['time']:.0f}s loss={ema.get('loss', float('nan')):.3f} "
                  f"search e/n={ema.get('search_easy', 0):.2f}/{ema.get('search_need', 0):.2f} "
                  f"greedy e/n={ema.get('greedy_easy', 0):.2f}/{ema.get('greedy_need', 0):.2f} "
                  f"VAL easy={stats['val_easy']:.2f} need={stats['val_need']:.3f} closed={stats.get('val_need_closed', float('nan')):.3f} "
                  f"buf={buf.n} need={stats['buf_need']:.2f} hard={stats['buf_hard']:.2f} tau={stats['buf_tau']:.1f} alive={stats['alive_T']:.1f}",
                  flush=True)
            torch.save(dict(model=ema_model.state_dict(), raw=model.state_dict(), ema=ema_model.state_dict(), opt=opt.state_dict(),
                            round=rnd + 1, args=vars(args), posed=posed), os.path.join(out_dir, "ckpt.pt"))
            score = stats.get("val_need_closed", stats["val_need"])
            if score >= ema.get("best_val", -1):
                ema["best_val"] = score
                torch.save(dict(model=ema_model.state_dict(), round=rnd + 1, args=vars(args), val_need=stats["val_need"],
                                val_need_closed=stats.get("val_need_closed"), posed=posed.clone()), os.path.join(out_dir, "best.pt"))
            if rnd % 100 == 0:
                torch.save(dict(model=ema_model.state_dict(), round=rnd + 1, args=vars(args), posed=posed.clone()),
                           os.path.join(out_dir, f"ckpt_{rnd:05d}.pt"))
        logf.write(json.dumps(stats) + "\n")
        logf.flush()

    ev = evaluate(ema_model, test_need, cfg, args.K, args.amp)
    print(f"FINAL test_need success = {float(ev['success'].float().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
