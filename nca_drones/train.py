"""Reward-only training of the space-time NCA.

Loop (one "round"):
  1. sample tasks uniformly (no curriculum),
  2. grow the space-time volume with the current NCA,
  3. SEARCH: follow the two grown tubes stochastically in the simulator
     (M rollouts per task), collect the sparse reward,
  4. keep, per task, one rewarded joint rollout,
  5. LEARN: train the NCA to grow the tubes of rewarded rollouts.

There is no solver, no distance feature and no shaped reward anywhere: the
only teacher is `DroneEnv.step` returning 1 when the seeker reaches the target.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from env import Cfg, DroneEnv, manhattan, sample_tasks
from nca import SpaceTimeNCA, make_cond


# --------------------------------------------------------------------------
# rollouts: steer both drones from the voxel volume
# --------------------------------------------------------------------------
def rollout(logits, s, r, g, cfg, M=1, eps=None, temp=1.0, greedy=None, gen=None):
    """Run M simulator episodes per task, steering from the grown volume.

    logits: [N, 2, T+1, H, W]  tube logits (channel 0 seeker, 1 recharger).
    At time t a drone standing on cell p looks at the 5 cells it can reach and
    moves to one of them according to the tube logits in slice t+1.
    eps:    [N*M] probability of a uniformly random action (exploration)
    greedy: [N*M] bool, take the arg-max instead of sampling
    Returns dict with trajectories [N*M, T+1, 2] etc.
    """
    N = s.shape[0]
    tid = torch.arange(N).repeat_interleave(M)
    n = N * M
    env = DroneEnv(s[tid], r[tid], g[tid], cfg)
    if eps is None:
        eps = torch.zeros(n)
    if greedy is None:
        greedy = torch.ones(n, dtype=torch.bool)
    ar = torch.arange(n)
    traj_s = [env.ps.clone()]
    traj_r = [env.pr.clone()]
    logp = torch.zeros(n)
    ret = torch.zeros(n)
    for t in range(cfg.T):
        acts = []
        active = ~env.done
        for d, pos in ((0, env.ps), (1, env.pr)):
            c = env.cand(pos)                                        # [n,5,2]
            lg = logits[tid[:, None], d, t + 1, c[..., 0], c[..., 1]].float()  # [n,5]
            logq = F.log_softmax(lg / temp, -1)
            p = (1 - eps[:, None]) * logq.exp() + eps[:, None] / 5.0
            a = torch.multinomial(p, 1, generator=gen).squeeze(1)
            a = torch.where(greedy, lg.argmax(1), a)
            logp = logp + torch.where(active, logq[ar, a], torch.zeros(n))
            acts.append(a)
        ret = ret + env.step(acts[0], acts[1])
        traj_s.append(env.ps.clone())
        traj_r.append(env.pr.clone())
    return dict(
        traj_s=torch.stack(traj_s, 1), traj_r=torch.stack(traj_r, 1),
        success=ret > 0, tau=env.tau.clone(), logp=logp, tid=tid,
        n_recharge=env.n_recharge.clone(),
    )


def traj_to_labels(traj_s, traj_r, cfg):
    """[n,T+1,2] positions -> [n,2,T+1] cell index per time slice."""
    ls = traj_s[..., 0] * cfg.W + traj_s[..., 1]
    lr = traj_r[..., 0] * cfg.W + traj_r[..., 1]
    return torch.stack([ls, lr], 1)


def tube_loss(logits, labels, cfg, w=None):
    """Per-time-slice cross-entropy: each slice is a distribution over cells."""
    n = logits.shape[0]
    lg = logits.float().reshape(n, 2, cfg.T + 1, cfg.H * cfg.W)[:, :, 1:]
    ce = F.cross_entropy(lg.reshape(-1, cfg.H * cfg.W), labels[:, :, 1:].reshape(-1), reduction="none")
    ce = ce.reshape(n, 2, cfg.T)
    if w is not None:
        ce = ce * w
    return ce.mean()


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
        if amp:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                lg, _ = model(cond, K, noise=noise)
        else:
            lg, _ = model(cond, K, noise=noise)
        outs.append(lg.float())
    return torch.cat(outs)


@torch.no_grad()
def evaluate(model, tasks, cfg, K, amp=False):
    """Greedy open-loop readout of both tubes, executed in the simulator."""
    model.eval()
    s, r, g = tasks
    logits = grow(model, s, r, g, cfg, K, amp)
    out = rollout(logits, s, r, g, cfg, M=1)
    model.train()
    return out


class Buffer:
    def __init__(self, cap, cfg):
        self.cap = cap
        self.s = torch.zeros(cap, 2, dtype=torch.long)
        self.r = torch.zeros(cap, 2, dtype=torch.long)
        self.g = torch.zeros(cap, 2, dtype=torch.long)
        self.lab = torch.zeros(cap, 2, cfg.T + 1, dtype=torch.long)
        self.need = torch.zeros(cap, dtype=torch.bool)
        self.n = 0
        self.ptr = 0

    def add(self, s, r, g, lab, need):
        for i in range(s.shape[0]):
            j = self.ptr
            self.s[j], self.r[j], self.g[j], self.lab[j], self.need[j] = s[i], r[i], g[i], lab[i], need[i]
            self.ptr = (self.ptr + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, bs, gen, need_frac=None):
        if need_frac is not None and self.need[: self.n].any() and (~self.need[: self.n]).any():
            nn_ = int(round(bs * need_frac))
            idx_need = torch.nonzero(self.need[: self.n]).squeeze(1)
            idx_easy = torch.nonzero(~self.need[: self.n]).squeeze(1)
            a = idx_need[torch.randint(len(idx_need), (nn_,), generator=gen)]
            b = idx_easy[torch.randint(len(idx_easy), (bs - nn_,), generator=gen)]
            idx = torch.cat([a, b])
        else:
            idx = torch.randint(self.n, (bs,), generator=gen)
        return self.s[idx], self.r[idx], self.g[idx], self.lab[idx]


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
    ap.add_argument("--G", type=int, default=3, help="gradient steps per round")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--buf", type=int, default=20000)
    ap.add_argument("--eps_max", type=float, default=0.4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--B", type=int, default=5)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--need_frac", type=float, default=-1.0)
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
    held = set(task_key(*val_need, cfg).tolist()) | set(task_key(*test_need, cfg).tolist()) | set(task_key(*val_easy, cfg).tolist())

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
    eps_levels = torch.linspace(0, args.eps_max, args.M)
    eps = eps_levels.repeat(args.N)
    greedy = torch.zeros(args.N * args.M, dtype=torch.bool)
    greedy[:: args.M] = True

    t0 = time.time()
    ema = {}
    for rnd in range(start_round, args.rounds):
        # ---------------- search -----------------
        s, r, g = sample_tasks(args.N * 2, cfg, gen, "all")
        keys = task_key(s, r, g, cfg).tolist()
        keep = torch.tensor([k not in held for k in keys])
        s, r, g = s[keep][: args.N], r[keep][: args.N], g[keep][: args.N]
        need = manhattan(s, g) > cfg.B
        with torch.no_grad():
            logits = grow(model, s, r, g, cfg, args.K, args.amp)
            ro = rollout(logits, s, r, g, cfg, M=args.M, eps=eps, temp=args.temp, greedy=greedy, gen=gen)
        succ = ro["success"].reshape(args.N, args.M)
        tau = ro["tau"].reshape(args.N, args.M).float()
        logp = ro["logp"].reshape(args.N, args.M)
        # among rewarded rollouts: earliest reward first, then the one the tissue itself finds most likely
        score = torch.where(succ, -tau * 1000.0 + logp.clamp(min=-900), torch.full_like(tau, -1e9))
        best = score.argmax(1)
        any_succ = succ.any(1)
        sel = torch.arange(args.N) * args.M + best
        lab = traj_to_labels(ro["traj_s"][sel], ro["traj_r"][sel], cfg)
        if any_succ.any():
            buf.add(s[any_succ], r[any_succ], g[any_succ], lab[any_succ], need[any_succ])

        stats = dict(
            round=rnd,
            search_easy=float(any_succ[~need].float().mean()) if (~need).any() else float("nan"),
            search_need=float(any_succ[need].float().mean()) if need.any() else float("nan"),
            greedy_easy=float(succ[:, 0][~need].float().mean()) if (~need).any() else float("nan"),
            greedy_need=float(succ[:, 0][need].float().mean()) if need.any() else float("nan"),
            buf=buf.n, buf_need=float(buf.need[: buf.n].float().mean()) if buf.n else 0.0,
        )

        # ---------------- learn ------------------
        losses = []
        if buf.n >= args.bs:
            for _ in range(args.G):
                bs_, br_, bg_, blab = buf.sample(args.bs, gen, args.need_frac if args.need_frac >= 0 else None)
                cond = make_cond(bs_, br_, bg_, cfg)
                K = args.K + int(torch.randint(-args.K_jitter, args.K_jitter + 1, (1,), generator=gen))
                if args.amp:
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        lg, _ = model(cond, K)
                else:
                    lg, _ = model(cond, K)
                loss = tube_loss(lg, blab, cfg)
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
                  f"VAL easy={stats['val_easy']:.2f} need={stats['val_need']:.3f} buf={buf.n} need_frac={stats['buf_need']:.2f}",
                  flush=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), round=rnd + 1, args=vars(args)),
                       os.path.join(out_dir, "ckpt.pt"))
        logf.write(json.dumps(stats) + "\n")
        logf.flush()

    ev = evaluate(model, test_need, cfg, args.K, args.amp)
    print(f"FINAL test_need success = {float(ev['success'].float().mean()):.3f}", flush=True)


if __name__ == "__main__":
    main()
