"""Final evaluation and probes for a trained space-time NCA.

  python3 analyze.py runs/<name>/best.pt [--full]

Everything here is OFFLINE analysis.  It may look at things the learner never
saw (battery level, Manhattan distances) - that is the point of a probe - and
nothing in this file is imported by training.
"""
import argparse
import json

import torch

from env import Cfg, DroneEnv, manhattan, sample_tasks
from nca import SpaceTimeNCA, make_cond
from train import evaluate, evaluate_closed, grow, rollout, task_key


def load(path):
    ck = torch.load(path)
    a = ck["args"]
    model = SpaceTimeNCA(C=a["C"], hid=a["hid"], update=a["update"], fire_rate=a.get("fire_rate", 1.0))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Cfg(T=a["T"], B=a["B"]), a, ck.get("round", -1)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def breakdown(out, s, r, g, cfg):
    succ = out["success"]
    met = out["n_recharge"] > 0
    tau = out["tau"]
    cats = {
        "success": succ,
        "never_met": ~succ & ~met,
        "met_then_died": ~succ & met & (tau < cfg.T),
        "timeout": ~succ & met & (tau >= cfg.T),
    }
    d, dsr = manhattan(s, g), manhattan(s, r)
    by_d = {int(k): [int(succ[d == k].sum()), int((d == k).sum())] for k in d.unique()}
    by_sr = {int(k): [int(succ[dsr == k].sum()), int((dsr == k).sum())] for k in dsr.unique()}
    return {k: round(float(v.float().mean()), 4) for k, v in cats.items()}, by_d, by_sr


@torch.no_grad()
def lesions(model, tasks, cfg, K, amp):
    """Counterfactual probes on recharge-needed tasks (open-loop dream)."""
    s, r, g = tasks
    n = s.shape[0]
    logits = grow(model, s, r, g, cfg, K, amp)
    base = rollout(logits, s, r, g, cfg, M=1)
    # (a) the recharger ignores its tube and hovers at its start
    env = DroneEnv(s, r, g, cfg)
    for t in range(cfg.T):
        c = env.cand(env.ps)
        a_s = logits[torch.arange(n)[:, None], 0, t + 1, c[..., 0], c[..., 1]].argmax(1)
        env.step(a_s, torch.zeros(n, dtype=torch.long))
    hover = env.success.float().mean()
    # (b) the recharger is deleted from the observation: how much of the seeker tube changes?
    cond = make_cond(s, r, g, cfg)
    cond[:, 1] = 0
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(amp)):
        lg2, _ = model(cond, K)
    lg2 = lg2.float()
    p1 = logits[:, 0].reshape(n, cfg.T + 1, -1).argmax(-1)
    p2 = lg2[:, 0].reshape(n, cfg.T + 1, -1).argmax(-1)
    changed = (p1[:, 1:] != p2[:, 1:]).any(1).float().mean()
    return dict(open_loop=float(base["success"].float().mean()), recharger_hovers=float(hover),
                seeker_tube_changes_without_recharger=float(changed))


@torch.no_grad()
def battery_probe(model, tasks, cfg, K, amp):
    """Is the (never observed) battery level linearly decodable from the hidden state on the seeker's worldline?"""
    s, r, g = tasks
    n = s.shape[0]
    cond = make_cond(s, r, g, cfg)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(amp)):
        logits, state = model(cond, K)
    logits, state = logits.float(), state.float()
    env = DroneEnv(s, r, g, cfg)
    X, Y, since = [], [], []
    last = torch.zeros(n, dtype=torch.long)
    for t in range(cfg.T):
        acts = []
        for d, pos in ((0, env.ps), (1, env.pr)):
            c = env.cand(pos)
            acts.append(logits[torch.arange(n)[:, None], d, t + 1, c[..., 0], c[..., 1]].argmax(1))
        act = ~env.done
        env.step(acts[0], acts[1])
        live = act
        feat = state[torch.arange(n), :, t + 1, env.ps[:, 0], env.ps[:, 1]]        # hidden state at the seeker's voxel
        X.append(feat[live]); Y.append(env.bat[live].float())
    X, Y = torch.cat(X), torch.cat(Y)
    m = len(X)
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: m // 2], perm[m // 2:]
    A = torch.cat([X, torch.ones(m, 1)], 1)
    w = torch.linalg.lstsq(A[tr].T @ A[tr] + 1e-3 * torch.eye(A.shape[1]), A[tr].T @ Y[tr, None]).solution
    pred = (A[te] @ w).squeeze(1)
    r2 = 1 - ((pred - Y[te]) ** 2).sum() / ((Y[te] - Y[te].mean()) ** 2).sum()
    # control: the same probe on time index alone
    tt = torch.cat([torch.full((int(1),), 0.0)])
    return dict(n_points=int(m), r2_battery_from_hidden=float(r2), battery_std=float(Y.std()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--full", action="store_true", help="also fresh-seed sets, lesions and probes")
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    model, cfg, a, rnd = load(args.ckpt)
    K, amp = args.K or a["K"], bool(a.get("amp", 0))
    res = dict(ckpt=args.ckpt, round=rnd, K=K)

    # the held-out test set: 200 recharge-needed tasks, seed 12345, never trained on, never used for model selection
    test = sample_tasks(200, cfg, torch.Generator().manual_seed(12345), "need")
    oc = evaluate_closed(model, test, cfg, K, amp)
    oo = evaluate(model, test, cfg, K, amp)
    kc, ko = int(oc["success"].sum()), int(oo["success"].sum())
    res["test200_closed_loop"] = dict(success=kc / 200, k=kc, n=200, ci95=[round(x, 3) for x in wilson(kc, 200)])
    res["test200_open_loop_dream"] = dict(success=ko / 200, k=ko, n=200, ci95=[round(x, 3) for x in wilson(ko, 200)])
    cats, by_d, by_sr = breakdown(oc, *test, cfg)
    res["test200_closed_breakdown"] = dict(outcomes=cats, by_target_distance=by_d, by_recharger_distance=by_sr)
    print(f"TEST (200 held-out recharge-needed tasks)  closed-loop: {kc}/200 = {kc/200:.3f}   open-loop dream: {ko}/200 = {ko/200:.3f}")
    print("  outcomes:", cats)
    print("  by d(seeker,target):   ", {k: f"{v[0]}/{v[1]}" for k, v in by_d.items()})
    print("  by d(seeker,recharger):", {k: f"{v[0]}/{v[1]}" for k, v in by_sr.items()})

    if args.full:
        # fresh tasks from other seeds (excluding anything in the validation / test sets)
        tot_k = tot_n = 0
        per = []
        for seed in (101, 202, 303, 404, 505):
            t = sample_tasks(200, cfg, torch.Generator().manual_seed(seed), "need")
            o = evaluate_closed(model, t, cfg, K, amp)
            k = int(o["success"].sum()); per.append(k / 200); tot_k += k; tot_n += 200
        res["fresh_1000_closed_loop"] = dict(success=tot_k / tot_n, per_seed=per, ci95=[round(x, 3) for x in wilson(tot_k, tot_n)])
        print(f"FRESH 5x200 recharge-needed tasks, closed-loop: {tot_k}/{tot_n} = {tot_k/tot_n:.3f}  per seed {per}")
        easy = sample_tasks(200, cfg, torch.Generator().manual_seed(606), "easy")
        oe = evaluate_closed(model, easy, cfg, K, amp)
        res["easy_200_closed_loop"] = float(oe["success"].float().mean())
        print(f"one-battery tasks, closed-loop: {res['easy_200_closed_loop']:.3f}")
        res["lesions"] = lesions(model, test, cfg, K, amp)
        print("lesions:", res["lesions"])
        res["battery_probe"] = battery_probe(model, test, cfg, K, amp)
        print("battery probe:", res["battery_probe"])
        # does the plan change at the (never shown) battery horizon?  share of plans in which the drones meet
        allt = sample_tasks(1500, cfg, torch.Generator().manual_seed(707), "all")
        oa = evaluate(model, allt, cfg, K, amp)
        d = manhattan(allt[0], allt[2])
        res["meet_rate_by_target_distance"] = {int(k): round(float((oa["n_recharge"][d == k] > 0).float().mean()), 3) for k in d.unique()}
        res["success_by_target_distance_open"] = {int(k): round(float(oa["success"][d == k].float().mean()), 3) for k in d.unique()}
        print("share of plans with a rendezvous, by d(seeker,target):", res["meet_rate_by_target_distance"])
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
