"""Diagnostics: look at what the tissue grows and where the joint plan breaks."""
import argparse

import torch

from env import Cfg, manhattan, sample_tasks
from nca import SpaceTimeNCA
from train import evaluate, grow, rollout


def load(path):
    ck = torch.load(path)
    a = ck["args"]
    model = SpaceTimeNCA(C=a["C"], hid=a["hid"], update=a["update"], fire_rate=a.get("fire_rate", 1.0))
    model.load_state_dict(ck["model"])
    cfg = Cfg(T=a["T"], B=a["B"])
    return model, cfg, a


def show(traj_s, traj_r, g, tau, cfg):
    """ASCII space-time strip: one 8x8 board per time step."""
    rows = [[] for _ in range(cfg.H)]
    for t in range(min(int(tau) + 1, cfg.T + 1)):
        for x in range(cfg.H):
            line = ""
            for y in range(cfg.W):
                ch = "."
                if (x, y) == tuple(g.tolist()):
                    ch = "G"
                s_here = (x, y) == tuple(traj_s[t].tolist())
                r_here = (x, y) == tuple(traj_r[t].tolist())
                if s_here and r_here:
                    ch = "#"
                elif s_here:
                    ch = "S"
                elif r_here:
                    ch = "R"
                line += ch
            rows[x].append(line)
    for x in range(cfg.H):
        print(" ".join(rows[x]))


def classify(out, s, r, g, cfg):
    """Failure taxonomy for greedy rollouts."""
    succ = out["success"]
    met = out["n_recharge"] > 0
    tau = out["tau"]
    cats = {
        "success": succ,
        "never_met": ~succ & ~met,
        "met_then_died": ~succ & met & (tau < cfg.T),
        "timeout": ~succ & met & (tau >= cfg.T),
    }
    return cats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--K", type=int, default=0)
    args = ap.parse_args()
    model, cfg, a = load(args.ckpt)
    K = args.K or a["K"]
    gen = torch.Generator().manual_seed(args.seed)
    s, r, g = sample_tasks(args.n, cfg, gen, "need")
    out = evaluate(model, (s, r, g), cfg, K, bool(a.get("amp", 0)))
    cats = classify(out, s, r, g, cfg)
    print("greedy on recharge-needed tasks:", {k: round(float(v.float().mean()), 3) for k, v in cats.items()})
    d = manhattan(s, g)
    dsr = manhattan(s, r)
    print("success by d(seeker,target):", {int(k): round(float(out["success"][d == k].float().mean()), 2) for k in d.unique()})
    print("success by d(seeker,recharger):", {int(k): round(float(out["success"][dsr == k].float().mean()), 2) for k in dsr.unique()})
    for name in ("success", "never_met", "met_then_died", "timeout"):
        idx = torch.nonzero(cats[name]).squeeze(1)[: args.show]
        for i in idx.tolist():
            print(f"\n--- {name}: s={s[i].tolist()} r={r[i].tolist()} g={g[i].tolist()} tau={int(out['tau'][i])} recharges={int(out['n_recharge'][i])}")
            show(out["traj_s"][i], out["traj_r"][i], g[i], out["tau"][i], cfg)
