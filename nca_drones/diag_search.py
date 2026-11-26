"""Diagnostic: are the remaining failures a SEARCH problem or a READOUT problem?

For a trained checkpoint and 2000 fresh recharge-needed tasks (seeds 5001..5010,
never posed in training where the run logged it), broken down by the distance
at which the recharger starts:

  greedy    the tissue's own plan (open-loop dream; closed-loop on the far tasks)
  joint64   training-time search: does any of the 64 joint rollouts earn the reward?
  xplay64   cross-play: any of the 64 x 64 (seeker worldline, recharger worldline) pairs?
  xplay256  the same with 256 x 256 pairs (far tasks only)

If search finds rewards where the greedy plan fails, the tissue has something to
learn from and more training helps.  If search finds nothing either, no amount of
imitation can help and the missing ingredient is exploration.

Offline analysis only (uses Manhattan distance to stratify); never imported by training.

  python3 diag_search.py runs/v10/best_prereg.pt [out.json]
"""
import json
import sys

import torch

from analyze import fresh_tasks, load
from env import manhattan
from train import BIG, crossplay, evaluate_closed, grow, rollout


def mix(N, M, shared=0.5, eps_max=0.3):
    """The exploration mix used in training: 1 greedy, the rest half independent / half shared noise."""
    n_sh = int(round(shared * (M - 1)))
    n_in = M - 1 - n_sh
    eps_m = torch.cat([torch.zeros(1), torch.linspace(0.02, eps_max, n_in), torch.linspace(0.0, eps_max, n_sh)])
    temp_m = torch.cat([torch.ones(1 + n_in), torch.tensor([1.0, 2.0, 3.0]).repeat(n_sh // 3 + 1)[:n_sh]])
    shared_m = torch.cat([torch.zeros(1 + n_in, dtype=torch.bool), torch.ones(n_sh, dtype=torch.bool)])
    greedy = torch.zeros(N * M, dtype=torch.bool)
    greedy[::M] = True
    return eps_m.repeat(N), temp_m.repeat(N), shared_m.repeat(N), greedy


def cells(traj, cfg):
    return traj[..., 0] * cfg.W + traj[..., 1]


@torch.no_grad()
def xplay(logits, s, r, g, cfg, M, gen):
    N = s.shape[0]
    eps, temp, sh, gr = mix(N, M)
    rk = rollout(logits, s, r, None, cfg, M=M, eps=eps, temp=temp, greedy=gr, gen=gen, ez=4, shared=sh, kinematic=True)
    cs = cells(rk["traj_s"], cfg).reshape(N, M, cfg.T + 1)
    cr = cells(rk["traj_r"], cfg).reshape(N, M, cfg.T + 1)
    arr = crossplay(cs, cr, cells(g, cfg), cfg)
    ok = arr < BIG
    return dict(any=ok.flatten(1).any(1), diag=ok.diagonal(dim1=1, dim2=2).any(1),
                uni_s=ok[:, :, 0].any(1), uni_r=ok[:, 0, :].any(1))        # only one drone deviates from the greedy plan


@torch.no_grad()
def main(path, out=None):
    model, cfg, a, rnd = load(path)
    K, amp = a["K"], bool(a.get("amp", 0))
    parts = [fresh_tasks(200, cfg, seed) for seed in range(5001, 5011)]
    s, r, g = (torch.cat([p[i] for p in parts]) for i in range(3))
    N = s.shape[0]
    dsr, dsg = manhattan(s, r), manhattan(s, g)
    gen = torch.Generator().manual_seed(0)

    logits = torch.cat([grow(model, s[i:i + 500], r[i:i + 500], g[i:i + 500], cfg, K, amp) for i in range(0, N, 500)])
    ro0 = rollout(logits, s, r, g, cfg, M=1)
    greedy_ok = ro0["success"]

    eps, temp, sh, gr = mix(N, 64)
    ro = rollout(logits, s, r, g, cfg, M=64, eps=eps, temp=temp, greedy=gr, gen=gen, ez=4, shared=sh)
    joint64 = ro["success"].reshape(N, 64).any(1)
    x64 = xplay(logits, s, r, g, cfg, 64, gen)

    far = dsr >= 8
    fi = torch.nonzero(far).squeeze(1)
    x256 = xplay(logits[fi], s[fi], r[fi], g[fi], cfg, 256, gen)
    x256_any = torch.zeros(N, dtype=torch.bool); x256_any[fi] = x256["any"]
    x256_diag = torch.zeros(N, dtype=torch.bool); x256_diag[fi] = x256["diag"]
    oc = evaluate_closed(model, (s[fi], r[fi], g[fi]), cfg, K, amp)
    closed_ok = torch.zeros(N, dtype=torch.bool); closed_ok[fi] = oc["success"]

    res = dict(ckpt=path, round=rnd, n=N, by_recharger_distance={})
    print(f"{path} (round {rnd}), {N} fresh recharge-needed tasks")
    print(f"{'d(s,r)':>6} {'n':>5} {'greedy':>7} {'closed':>7} {'joint64':>8} {'xplay64':>8} {'only-S':>7} {'only-R':>7} {'joint256':>9} {'xplay256':>9}")
    f = lambda m, v: float(v[m].float().mean())
    for d in sorted(dsr.unique().tolist()):
        m = dsr == d
        row = dict(n=int(m.sum()), greedy_open=f(m, greedy_ok), joint64=f(m, joint64), xplay64=f(m, x64["any"]),
                   unilateral_seeker=f(m, x64["uni_s"]), unilateral_recharger=f(m, x64["uni_r"]))
        if d >= 8:
            row.update(greedy_closed=f(m, closed_ok), joint256=f(m, x256_diag), xplay256=f(m, x256_any))
        res["by_recharger_distance"][d] = row
        g2 = lambda k: f"{row[k]:.2f}" if k in row else "  -  "
        print(f"{d:>6} {row['n']:>5} {g2('greedy_open'):>7} {g2('greedy_closed'):>7} {g2('joint64'):>8} {g2('xplay64'):>8} "
              f"{g2('unilateral_seeker'):>7} {g2('unilateral_recharger'):>7} {g2('joint256'):>9} {g2('xplay256'):>9}")
    tot = dict(greedy_open=float(greedy_ok.float().mean()), joint64=float(joint64.float().mean()), xplay64=float(x64["any"].float().mean()))
    res["all"] = tot
    print("all tasks:", {k: round(v, 3) for k, v in tot.items()})

    # anatomy of the far failures: who is not approaching?  (positions at t=5, when an unmet seeker dies)
    ts, tr = ro0["traj_s"], ro0["traj_r"]
    anat = {}
    for d in (8, 9, 10):
        m = (dsr == d) & ~greedy_ok & (ro0["n_recharge"] == 0)
        if int(m.sum()) < 5:
            continue
        gap5 = manhattan(ts[m, 5], tr[m, 5]).float()
        app_s = (d - manhattan(ts[m, 5], r[m])).float()            # how much the seeker closed the gap to the recharger's start
        app_r = (d - manhattan(tr[m, 5], s[m])).float()
        tgt_s = (dsg[m] - manhattan(ts[m, 5], g[m])).float()       # how much the seeker advanced towards the target
        anat[d] = dict(n=int(m.sum()), gap_at_t5=float(gap5.mean()), seeker_closed=float(app_s.mean()),
                       recharger_closed=float(app_r.mean()), seeker_towards_target=float(tgt_s.mean()),
                       gap_le_2=float((gap5 <= 2).float().mean()))
    res["never_met_anatomy"] = anat
    print("never-met failures (greedy, open-loop): mean after 5 steps")
    for d, v in anat.items():
        print(f"  d(s,r)={d}: n={v['n']}  remaining gap {v['gap_at_t5']:.2f}  (gap<=2 in {v['gap_le_2']:.0%})  seeker closed {v['seeker_closed']:.2f} of 5, "
              f"recharger closed {v['recharger_closed']:.2f} of 5, seeker advanced {v['seeker_towards_target']:.2f} towards the target")

    # adjacent starts: is it the swap problem (each steps onto the other's cell)?
    m1 = (dsr == 1)
    swap = ((ts[:, 1] == tr[:, 0]).all(-1) & (tr[:, 1] == ts[:, 0]).all(-1))
    cross_any = torch.zeros(N, dtype=torch.bool)
    for t in range(0, 5):
        cross_any |= ((ts[:, t + 1] == tr[:, t]).all(-1) & (tr[:, t + 1] == ts[:, t]).all(-1))
    res["adjacent_start"] = dict(n=int(m1.sum()), greedy_open=f(m1, greedy_ok), fail_n=int((m1 & ~greedy_ok).sum()),
                                 fail_with_swap_first_step=float(swap[m1 & ~greedy_ok].float().mean()) if (m1 & ~greedy_ok).any() else None,
                                 fail_with_swap_in_5=float(cross_any[m1 & ~greedy_ok].float().mean()) if (m1 & ~greedy_ok).any() else None,
                                 fail_never_met=float((ro0["n_recharge"][m1 & ~greedy_ok] == 0).float().mean()) if (m1 & ~greedy_ok).any() else None,
                                 swap_rate_all_failures=float(cross_any[~greedy_ok].float().mean()))
    print("adjacent start (d=1):", res["adjacent_start"])
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
