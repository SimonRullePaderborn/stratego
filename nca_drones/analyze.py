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
from nca import N_CODES, SpaceTimeNCA, cond_channels, make_cond, uses_battery, uses_codes
from train import evaluate, evaluate_closed, grow, rollout, task_key


def load(path):
    ck = torch.load(path)
    a = ck["args"]
    battery, codes = bool(a.get("battery", 0)), bool(a.get("codes", 0))      # switches of phase 2, steps 1 and 2
    model = SpaceTimeNCA(C=a["C"], hid=a["hid"], n_cond=cond_channels(battery, codes), n_out=N_CODES if codes else 2,
                         update=a["update"], fire_rate=a.get("fire_rate", 1.0))
    model.load_state_dict(ck["model"])
    model.eval()
    load.posed = ck.get("posed")          # tasks posed to the learner up to this checkpoint (None for old runs)
    return model, Cfg(T=a["T"], B=a["B"]), a, ck.get("round", -1)


def fresh_tasks(n, cfg, seed, mode="need"):
    """Fresh tasks that are disjoint from validation/test and, when the run logged it, from every task ever posed in training."""
    held = torch.cat([task_key(*sample_tasks(200, cfg, torch.Generator().manual_seed(999), "need"), cfg),
                      task_key(*sample_tasks(100, cfg, torch.Generator().manual_seed(998), "easy"), cfg),
                      task_key(*sample_tasks(200, cfg, torch.Generator().manual_seed(12345), "need"), cfg)])
    gen = torch.Generator().manual_seed(seed)
    out = [[], [], []]
    have = 0
    while have < n:
        s, r, g = sample_tasks(2 * n, cfg, gen, mode)
        k = task_key(s, r, g, cfg)
        ok = ~torch.isin(k, held)
        if load.posed is not None:
            ok &= ~load.posed[k]
        for o, x in zip(out, (s, r, g)):
            o.append(x[ok])
        have += int(ok.sum())
    return tuple(torch.cat(o)[:n] for o in out)


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
    cond = make_cond(s, r, g, cfg, battery=uses_battery(model))
    cond[:, 1] = 0
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(amp)):
        lg2, _ = model(cond, K)
    lg2 = lg2.float()
    p1 = logits[:, 0].reshape(n, cfg.T + 1, -1).argmax(-1)
    p2 = lg2[:, 0].reshape(n, cfg.T + 1, -1).argmax(-1)
    changed = (p1[:, 1:] != p2[:, 1:]).any(1).float().mean()
    return dict(open_loop=float(base["success"].float().mean()), recharger_hovers=float(hover),
                seeker_tube_changes_without_recharger=float(changed))


def _r2(X, Y, tr, te):
    A = torch.cat([X, torch.ones(len(X), 1)], 1)
    w = torch.linalg.lstsq(A[tr].T @ A[tr] + 1e-3 * torch.eye(A.shape[1]), A[tr].T @ Y[tr, None]).solution
    pred = (A[te] @ w).squeeze(1)
    return float(1 - ((pred - Y[te]) ** 2).sum() / ((Y[te] - Y[te].mean()) ** 2).sum())


@torch.no_grad()
def battery_probe(model, tasks, cfg, K, amp):
    """Is the (never observed) battery level linearly decodable from the hidden state on the seeker's worldline?

    Battery correlates with the time index (it is B - t until the first contact), and the tissue can
    know t from the seed slice, so the honest question is what the hidden state adds BEYOND time."""
    s, r, g = tasks
    n = s.shape[0]
    cond = make_cond(s, r, g, cfg, battery=uses_battery(model))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bool(amp)):
        logits, state = model(cond, K)
    logits, state = logits.float(), state.float()
    env = DroneEnv(s, r, g, cfg)
    X, Y, Tm = [], [], []
    for t in range(cfg.T):
        acts = []
        for d, pos in ((0, env.ps), (1, env.pr)):
            c = env.cand(pos)
            acts.append(logits[torch.arange(n)[:, None], d, t + 1, c[..., 0], c[..., 1]].argmax(1))
        live = ~env.done
        env.step(acts[0], acts[1])
        feat = state[torch.arange(n), :, t + 1, env.ps[:, 0], env.ps[:, 1]]        # hidden state at the seeker's voxel
        X.append(feat[live]); Y.append(env.bat[live].float()); Tm.append(torch.full((int(live.sum()),), float(t + 1)))
    X, Y, Tm = torch.cat(X), torch.cat(Y), torch.cat(Tm)
    m = len(X)
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: m // 2], perm[m // 2:]
    onehot_t = torch.nn.functional.one_hot(Tm.long(), cfg.T + 1).float()
    r2_time = _r2(onehot_t, Y, tr, te)
    r2_hidden = _r2(X, Y, tr, te)
    r2_both = _r2(torch.cat([X, onehot_t], 1), Y, tr, te)
    late = Tm > cfg.B                                                              # after B steps battery is no longer a function of t
    pl = torch.nonzero(late).squeeze(1)
    r2_late_time = r2_late_hidden = float("nan")
    if len(pl) > 200:
        pp = torch.randperm(len(pl), generator=torch.Generator().manual_seed(1))
        h = len(pl) // 2
        r2_late_time = _r2(onehot_t[pl], Y[pl], pp[:h], pp[h:])
        r2_late_hidden = _r2(X[pl], Y[pl], pp[:h], pp[h:])
    return dict(n_points=int(m), r2_from_time_index=r2_time, r2_from_hidden=r2_hidden, r2_from_hidden_plus_time=r2_both,
                after_step_B=dict(n=int(len(pl)), r2_from_time_index=r2_late_time, r2_from_hidden=r2_late_hidden))


@torch.no_grad()
def tube_diagnostics(model, tasks, cfg, K, amp):
    """Are the grown 'tubes' real worldlines, or a diffuse field that the reachable-cell arg-max rescues?"""
    s, r, g = tasks
    n = s.shape[0]
    logits = grow(model, s, r, g, cfg, K, amp)
    out = rollout(logits, s, r, g, cfg, M=1)
    p = torch.softmax(logits.reshape(n, 2, cfg.T + 1, -1), -1)
    gmax = p.argmax(-1)                                                 # slice-wide arg-max cell
    res = {}
    for d, name in ((0, "seeker"), (1, "recharger")):
        traj = out["traj_s"] if d == 0 else out["traj_r"]
        cell = traj[..., 0] * cfg.W + traj[..., 1]
        ts = torch.arange(cfg.T + 1)[None]
        live = (ts >= 1) & (ts <= out["tau"][:, None])
        same = (gmax[:, d] == cell)[live].float().mean()
        mass = p[:, d].gather(-1, cell[..., None]).squeeze(-1)[live].mean()
        gx, gy = gmax[:, d] // cfg.W, gmax[:, d] % cfg.W
        step = (gx[:, 1:] - gx[:, :-1]).abs() + (gy[:, 1:] - gy[:, :-1]).abs()
        legal = (step <= 1)[live[:, 1:]].float().mean()
        res[name] = dict(slice_argmax_is_executed_cell=round(float(same), 3), softmax_mass_on_executed_cell=round(float(mass), 3),
                         slice_argmax_moves_at_most_one_cell=round(float(legal), 3))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--full", action="store_true", help="also fresh-seed sets, lesions and probes")
    ap.add_argument("--prereg", action="store_true", help="the pre-registered final evaluation (EVAL_PROTOCOL.md)")
    ap.add_argument("--K", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    model, cfg, a, rnd = load(args.ckpt)
    if args.full and uses_codes(model):
        raise SystemExit("--full (lesions, probes, tube diagnostics) is not written for models with codes; use --prereg")
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

    if args.prereg:
        out = {}
        for prec, use_amp in (("bf16", True), ("fp32", False)):
            kc = ko = n = 0
            per_c, per_o = [], []
            dist_c = {}
            for seed in range(2001, 2011):
                t = fresh_tasks(200, cfg, seed)
                o = evaluate_closed(model, t, cfg, K, use_amp)
                o2 = evaluate(model, t, cfg, K, use_amp)
                kc += int(o["success"].sum()); ko += int(o2["success"].sum()); n += 200
                per_c.append(int(o["success"].sum())); per_o.append(int(o2["success"].sum()))
                dsr = manhattan(t[0], t[1])
                for d_ in dsr.unique().tolist():
                    a_, b_ = dist_c.get(d_, (0, 0))
                    dist_c[d_] = (a_ + int(o["success"][dsr == d_].sum()), b_ + int((dsr == d_).sum()))
            tc = int(evaluate_closed(model, test, cfg, K, use_amp)["success"].sum())
            to = int(evaluate(model, test, cfg, K, use_amp)["success"].sum())
            out[prec] = dict(
                new2000_closed=dict(k=kc, n=n, rate=kc / n, ci95=[round(x, 4) for x in wilson(kc, n)], per_seed=per_c),
                new2000_open=dict(k=ko, n=n, rate=ko / n, ci95=[round(x, 4) for x in wilson(ko, n)], per_seed=per_o),
                test200_closed=dict(k=tc, n=200, rate=tc / 200, ci95=[round(x, 4) for x in wilson(tc, 200)]),
                test200_open=dict(k=to, n=200, rate=to / 200, ci95=[round(x, 4) for x in wilson(to, 200)]),
                new2000_closed_by_recharger_distance={int(k_): f"{v[0]}/{v[1]}" for k_, v in sorted(dist_c.items())})
            print(f"[{prec}] NEW 2000 (never posed in training: {load.posed is not None}): closed {kc}/{n} = {kc/n:.4f} CI {wilson(kc, n)}   "
                  f"open {ko}/{n} = {ko/n:.4f} CI {wilson(ko, n)}")
            print(f"[{prec}] TEST 200: closed {tc}/200   open {to}/200")
            print(f"[{prec}] closed-loop by seeker-recharger distance:", out[prec]["new2000_closed_by_recharger_distance"])
        res["prereg"] = out
        res["disjoint_from_training"] = load.posed is not None
        res["n_posed_tasks"] = int(load.posed.sum()) if load.posed is not None else None

    if args.full:
        # fresh tasks from other seeds (excluding anything in the validation / test sets)
        tot_k = tot_n = tot_ko = 0
        per, per_o = [], []
        for seed in (101, 202, 303, 404, 505):
            t = fresh_tasks(200, cfg, seed)
            o = evaluate_closed(model, t, cfg, K, amp)
            oo2 = evaluate(model, t, cfg, K, amp)
            k = int(o["success"].sum()); per.append(k / 200); tot_k += k; tot_n += 200
            k2 = int(oo2["success"].sum()); per_o.append(k2 / 200); tot_ko += k2
        res["fresh_1000"] = dict(disjoint_from_training=load.posed is not None,
                                 closed_loop=dict(success=tot_k / tot_n, per_seed=per, ci95=[round(x, 3) for x in wilson(tot_k, tot_n)]),
                                 open_loop=dict(success=tot_ko / tot_n, per_seed=per_o, ci95=[round(x, 3) for x in wilson(tot_ko, tot_n)]))
        print(f"FRESH 5x200 recharge-needed tasks (never posed in training: {load.posed is not None})  "
              f"closed-loop {tot_k}/{tot_n} = {tot_k/tot_n:.3f} {per}   open-loop {tot_ko}/{tot_n} = {tot_ko/tot_n:.3f} {per_o}")
        # robustness of the open-loop dream to the number of iterations and to full precision
        sweep = {}
        for k_it in (24, 28, 32, 36, 40, 48):
            sweep[f"K={k_it}"] = float(evaluate(model, test, cfg, k_it, amp)["success"].float().mean())
        sweep["K=32_fp32"] = float(evaluate(model, test, cfg, K, False)["success"].float().mean())
        res["open_loop_test_vs_iterations"] = sweep
        print("open-loop test success vs iterations / precision:", sweep)
        res["tube_diagnostics"] = tube_diagnostics(model, test, cfg, K, amp)
        print("tube diagnostics:", res["tube_diagnostics"])
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
