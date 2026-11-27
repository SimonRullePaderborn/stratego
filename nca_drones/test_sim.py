"""Consistency checks for the simulator and for hindsight relabelling.

Run:  python3 test_sim.py
"""
import torch

from env import Cfg, DroneEnv, manhattan, replay, sample_tasks
from train import BIG, hindsight_table, make_labels, rollout


def test_basic():
    cfg = Cfg()
    s = torch.tensor([[0, 0]]); r = torch.tensor([[7, 7]]); g = torch.tensor([[0, 6]])
    env = DroneEnv(s, r, g, cfg)
    for _ in range(6):
        env.step(torch.tensor([4]), torch.tensor([0]))
    assert not bool(env.success) and int(env.tau) == 5, "seeker must die after B=5 steps without recharge"
    # recharger waits on the path -> seeker survives
    r = torch.tensor([[0, 3]])
    env = DroneEnv(s, r, g, cfg)
    for _ in range(6):
        env.step(torch.tensor([4]), torch.tensor([0]))
    assert bool(env.success) and int(env.tau) == 6 and int(env.n_recharge) == 1
    # arriving with an empty battery still counts
    g = torch.tensor([[0, 5]]); r = torch.tensor([[7, 7]])
    env = DroneEnv(s, r, g, cfg)
    for _ in range(5):
        env.step(torch.tensor([4]), torch.tensor([0]))
    assert bool(env.success) and int(env.tau) == 5


def test_hindsight_matches_simulator():
    """Every hindsight-rewarded (rollout, cell) pair must be rewarded by a true replay, and vice versa."""
    cfg = Cfg()
    gen = torch.Generator().manual_seed(0)
    N, M = 48, 32
    s, r, g = sample_tasks(N, cfg, gen)
    logits = torch.randn(N, 2, cfg.T + 1, cfg.H, cfg.W, generator=gen) * 2.0
    eps = torch.linspace(0, 0.5, M).repeat(N)
    greedy = torch.zeros(N * M, dtype=torch.bool); greedy[::M] = True
    ro = rollout(logits, s, r, None, cfg, M=M, eps=eps, greedy=greedy, gen=gen)
    best_time, best_roll, fv = hindsight_table(ro, N, M, cfg)
    tid = ro["tid"]
    checked = 0
    for c in range(cfg.H * cfg.W):
        gc = torch.tensor([[c // cfg.W, c % cfg.W]]).expand(N * M, -1)
        succ, tau = replay(s[tid], r[tid], gc, ro["acts"], cfg)
        hs = fv[:, c] < BIG
        start_here = (s[tid] == gc).all(-1)   # target == start cell is not a task (first-visit ignores t=0)
        assert bool((succ == hs)[~start_here].all()), f"cell {c}: hindsight success disagrees with simulator"
        assert bool((tau[succ & ~start_here] == fv[:, c][succ & ~start_here]).all()), f"cell {c}: arrival time disagrees"
        checked += int((succ & ~start_here).sum())
    # labels: replaying must visit exactly the labelled cells up to tau
    i = torch.nonzero((best_time < BIG).any(1)).squeeze(1)
    ci = (best_time[i] < BIG).float().argmax(1)
    lab = make_labels(ro, best_roll[i, ci], best_time[i, ci], cfg)
    roll = best_roll[i, ci]
    cells = ro["traj_s"][roll][..., 0] * cfg.W + ro["traj_s"][roll][..., 1]
    for j in range(len(i)):
        t = int(best_time[i[j], ci[j]])
        assert bool((lab[j, 0, : t + 1] == cells[j, : t + 1]).all())
        assert bool((lab[j, 0, t:] == ci[j]).all()), "after the reward the seeker stays on the target"
    print(f"hindsight == simulator on {checked} rewarded (rollout, target) pairs")


def test_task_filter():
    cfg = Cfg()
    gen = torch.Generator().manual_seed(1)
    s, r, g = sample_tasks(5000, cfg, gen, "need")
    assert bool((manhattan(s, g) > cfg.B).all()) and bool((manhattan(s, r) <= 2 * cfg.B).all())
    assert bool(((s != r).any(-1) & (s != g).any(-1) & (r != g).any(-1)).all())


if __name__ == "__main__":
    test_basic(); test_task_filter(); test_hindsight_matches_simulator()
    print("all simulator checks passed")


def test_prune():
    """arrival_time mirrors DroneEnv; pruned rollouts are real rewarded rollouts and never slower."""
    from prune import arrival_time, prune_batch, verify_labels
    cfg = Cfg()
    gen = torch.Generator().manual_seed(3)
    N, M = 64, 32
    s, r, g = sample_tasks(N, cfg, gen)
    logits = torch.randn(N, 2, cfg.T + 1, cfg.H, cfg.W, generator=gen) * 1.5
    eps = torch.linspace(0, 0.5, M).repeat(N)
    greedy = torch.zeros(N * M, dtype=torch.bool)
    ro = rollout(logits, s, r, None, cfg, M=M, eps=eps, greedy=greedy, gen=gen, ez=4)
    best_time, best_roll, fv = hindsight_table(ro, N, M, cfg)
    tid = ro["tid"]
    cs = (ro["traj_s"][..., 0] * cfg.W + ro["traj_s"][..., 1])
    cr = (ro["traj_r"][..., 0] * cfg.W + ro["traj_r"][..., 1])
    # 1. mirror check on every (rollout, cell)
    for c in range(0, cfg.H * cfg.W, 7):
        gc = torch.tensor([[c // cfg.W, c % cfg.W]]).expand(N * M, -1)
        succ, tau = replay(s[tid], r[tid], gc, ro["acts"], cfg)
        for i in range(0, N * M, 5):
            if int(cs[i, 0]) == c:
                continue
            a = arrival_time(cs[i].tolist(), cr[i].tolist(), c, cfg.B, cfg.T)
            assert (a > 0) == bool(succ[i]) and (a < 0 or a == int(tau[i])), (i, c, a, bool(succ[i]), int(tau[i]))
    # 2. pruning keeps the reward (authoritative replay) and never delays it
    reach = best_time < BIG
    reach[torch.arange(N), s[:, 0] * cfg.W + s[:, 1]] = False
    ti, ci = torch.nonzero(reach, as_tuple=True)
    sel = torch.randperm(len(ti), generator=gen)[:400]
    ti, ci = ti[sel], ci[sel]
    tau = best_time[ti, ci]
    import random
    lab, wts, tau_p, stop, rel = prune_batch(ro, best_roll[ti, ci], tau, ci, cfg, 8, random.Random(0))
    hg = torch.stack([ci // cfg.W, ci % cfg.W], -1)
    assert bool(verify_labels(s[ti], r[ti], hg, lab, tau_p, cfg).all())
    assert bool((tau_p <= tau).all())
    easy = tau_p <= cfg.B
    # a seeker that never needed a recharge must be judged independent of its partner
    solo = torch.tensor([all(int(lab[j, 0, t]) != int(lab[j, 1, t]) for t in range(1, int(tau_p[j]) + 1)) for j in range(len(tau_p))])
    assert bool((rel[solo] == 0).all()), "reward without any contact must not credit the recharger"
    assert bool((wts[:, 0, 1] == 1).all()) and bool((wts[solo][:, 1] [:, 1:] == 0).all())
    print(f"partner relevance: {float((rel > 0).float().mean()):.2%} of rewarded rollouts needed the recharger")
    print(f"pruning: mean arrival {tau.float().mean():.2f} -> {tau_p.float().mean():.2f}; recharger still moves in {(stop > 0).float().mean():.2%} of rollouts")


if __name__ == "__main__":
    test_prune()


def test_crossplay():
    """The batched pair judge must agree with a true simulator replay for every pair."""
    from train import crossplay
    from prune import labels_to_actions
    cfg = Cfg()
    gen = torch.Generator().manual_seed(5)
    N, M = 24, 12
    s, r, g = sample_tasks(N, cfg, gen)
    logits = torch.randn(N, 2, cfg.T + 1, cfg.H, cfg.W, generator=gen) * 1.5
    eps = torch.linspace(0, 0.4, M).repeat(N)
    shared = (torch.arange(M) % 2 == 0).repeat(N)
    ro = rollout(logits, s, r, None, cfg, M=M, eps=eps, greedy=torch.zeros(N * M, dtype=torch.bool), gen=gen, ez=3, shared=shared, kinematic=True)
    cs = (ro["traj_s"][..., 0] * cfg.W + ro["traj_s"][..., 1]).reshape(N, M, cfg.T + 1)
    cr = (ro["traj_r"][..., 0] * cfg.W + ro["traj_r"][..., 1]).reshape(N, M, cfg.T + 1)
    gc = g[:, 0] * cfg.W + g[:, 1]
    arr = crossplay(cs, cr, gc, cfg)
    acts = ro["acts"].reshape(N, M, cfg.T, 2)
    n_ok = 0
    for i in range(M):
        for j in range(M):
            a = torch.stack([acts[:, i, :, 0], acts[:, j, :, 1]], -1)
            succ, tau = replay(s, r, g, a, cfg)
            assert bool((succ == (arr[:, i, j] < BIG)).all()), (i, j)
            assert bool((tau[succ] == arr[:, i, j][succ]).all()), (i, j)
            n_ok += int(succ.sum())
    print(f"cross-play judge == simulator replay on all {N * M * M} pairs ({n_ok} rewarded)")


if __name__ == "__main__":
    test_crossplay()


def test_null_partner_credit():
    """Credit is 0 for a recharger that was never needed and ~1 where the reward needed it."""
    from credit import null_partner_credit
    cfg = Cfg()
    gen = torch.Generator().manual_seed(11)
    # seeker walks 3 steps east to the target: recharger irrelevant
    s = torch.tensor([[0, 0], [0, 0]]); r = torch.tensor([[7, 7], [0, 3]]); g = torch.tensor([[0, 3], [0, 7]])
    acts = torch.zeros(2, cfg.T, 2, dtype=torch.long)
    acts[0, :3, 0] = 4                       # task 0: east x3, recharger idle
    acts[1, :7, 0] = 4                       # task 1: east x7 through the recharger waiting at (0,3)
    succ, tau = replay(s, r, g, acts, cfg)
    assert bool(succ.all()) and tau.tolist() == [3, 7]
    w_s, w_r = null_partner_credit(s, r, g, acts, cfg, 2000, gen)
    assert float(w_r[0]) == 0.0, "recharger gets no credit when the seeker did it alone"
    assert 0.45 < float(w_r[1]) < 0.7, "recharger on the seeker's path: a random partner rescues it about 45% of the time"
    assert float(w_s[1]) == 1.0, "a random seeker cannot reach a far target"
    print(f"null-partner credit: easy task recharger {float(w_r[0]):.2f}, needed-recharge task recharger {float(w_r[1]):.2f}, seekers {w_s.tolist()}")


if __name__ == "__main__":
    test_null_partner_credit()


def test_battery_input():
    """Phase 2, step 1: the battery map holds the simulator's own level, on the seeker's cell, for the realised past only."""
    from nca import make_cond
    cfg = Cfg()
    # seeker walks east from (0,0) through the recharger waiting at (0,3) to the target at (0,6)
    s = torch.tensor([[0, 0]]); r = torch.tensor([[0, 3]]); g = torch.tensor([[0, 6]])
    env = DroneEnv(s, r, g, cfg)
    hist = torch.zeros(1, 2, cfg.T + 1, dtype=torch.long)
    bat = torch.zeros(1, cfg.T + 1, dtype=torch.long)
    hist[:, 0, 0] = 0; hist[:, 1, 0] = 3; bat[:, 0] = env.bat
    for t in range(6):
        env.step(torch.tensor([4]), torch.tensor([0]))
        hist[:, 0, t + 1] = env.ps[:, 0] * cfg.W + env.ps[:, 1]
        hist[:, 1, t + 1] = env.pr[:, 0] * cfg.W + env.pr[:, 1]
        bat[:, t + 1] = env.bat
    assert bat[0, :7].tolist() == [5, 4, 3, 5, 4, 3, 2] and bool(env.success), "one unit per step, full again at the meeting"
    t0 = torch.tensor([4])                                   # the first four steps are history
    plain = make_cond(s, r, g, cfg, hist, t0)
    cond = make_cond(s, r, g, cfg, hist, t0, bat=bat, battery=True)
    assert cond.shape[1] == 4 and bool((cond[:, :3] == plain).all()), "the three position maps are unchanged"
    want = {(0, 0): 1.0, (1, 1): 0.8, (2, 2): 0.6, (3, 3): 1.0, (4, 4): 0.8}          # (slice, column) -> level / B
    ch = cond[0, 3]
    for (t, col), v in want.items():
        assert abs(float(ch[t, 0, col]) - v) < 1e-6, f"battery map at slice {t}"
    assert int((ch != 0).sum()) == len(want), "nothing beyond the realised past, nothing off the seeker's cell"
    start = make_cond(s, r, g, cfg, battery=True)
    assert float(start[0, 3, 0, 0, 0]) == 1.0 and int((start[0, 3] != 0).sum()) == 1, "a full battery at t = 0"
    # the trace that rollout records is the simulator's own
    gen = torch.Generator().manual_seed(3)
    s2, r2, g2 = sample_tasks(16, cfg, gen)
    logits = torch.randn(16, 2, cfg.T + 1, cfg.H, cfg.W, generator=gen)
    ro = rollout(logits, s2, r2, g2, cfg, M=1)
    env = DroneEnv(s2, r2, g2, cfg)
    assert bool((ro["bat"][:, 0] == cfg.B).all())
    for t in range(cfg.T):
        env.step(ro["acts"][:, t, 0], ro["acts"][:, t, 1])
        assert bool((ro["bat"][:, t + 1] == env.bat).all()), f"battery trace differs at step {t + 1}"
    print("battery input: the map holds the simulator's level on the seeker's cell, for the realised past only")


if __name__ == "__main__":
    test_battery_input()


def test_codes():
    """Phase 2, step 2: each drone is drawn on the map its code names, and its tube is the output map of that code."""
    from nca import N_CODES, cond_channels, draw_codes, make_cond, pick_tubes
    cfg = Cfg()
    codes = draw_codes(1000, torch.Generator().manual_seed(5))
    assert codes.shape == (1000, 2) and bool((codes[:, 0] != codes[:, 1]).all()), "two different codes per episode"
    assert int(codes.min()) == 0 and int(codes.max()) == N_CODES - 1
    assert len(codes[:, 0].unique()) == N_CODES and len(codes[:, 1].unique()) == N_CODES, "every code occurs for both drones"
    # the episode of test_battery_input: seeker east from (0,0) through the recharger at (0,3); four steps are history
    s = torch.tensor([[0, 0]]); r = torch.tensor([[0, 3]]); g = torch.tensor([[0, 6]])
    env = DroneEnv(s, r, g, cfg)
    hist = torch.zeros(1, 2, cfg.T + 1, dtype=torch.long)
    bat = torch.zeros(1, cfg.T + 1, dtype=torch.long)
    hist[:, 0, 0] = 0; hist[:, 1, 0] = 3; bat[:, 0] = env.bat
    for t in range(6):
        env.step(torch.tensor([4]), torch.tensor([0]))
        hist[:, 0, t + 1] = env.ps[:, 0] * cfg.W + env.ps[:, 1]
        hist[:, 1, t + 1] = env.pr[:, 0] * cfg.W + env.pr[:, 1]
        bat[:, t + 1] = env.bat
    c = torch.tensor([[5, 11]])                              # seeker code 5, recharger code 11
    cond = make_cond(s, r, g, cfg, hist, torch.tensor([4]), bat=bat, battery=True, codes=c)
    assert cond.shape[1] == cond_channels(True, True) == 1 + 3 * N_CODES
    assert float(cond[0, 0, 0, 0, 6]) == 1.0 and int((cond[0, 0] != 0).sum()) == 1, "map 0: the target, in slice 0"
    ch = cond[0, 1 + 5]
    assert [float(ch[t, 0, t]) for t in range(5)] == [1.0] * 5 and int((ch != 0).sum()) == 5, "the seeker on the map of its code"
    ch = cond[0, 1 + N_CODES + 11]
    assert [float(ch[t, 0, 3]) for t in range(5)] == [1.0] * 5 and int((ch != 0).sum()) == 5, "the recharger on the map of its code"
    ch = cond[0, 1 + 2 * N_CODES + 5]
    for t, v in enumerate([1.0, 0.8, 0.6, 1.0, 0.8]):
        assert abs(float(ch[t, 0, t]) - v) < 1e-6, f"battery level at slice {t}"
    assert int((ch != 0).sum()) == 5
    assert int((cond != 0).sum()) == 1 + 5 + 5 + 5, "every other map is empty"
    start = make_cond(s, r, g, cfg, battery=True, codes=c)
    assert int((start != 0).sum()) == 4 and float(start[0, 1 + 5, 0, 0, 0]) == 1.0 and float(start[0, 1 + N_CODES + 11, 0, 0, 3]) == 1.0 \
        and float(start[0, 1 + 2 * N_CODES + 5, 0, 0, 0]) == 1.0, "without a past: slice 0 only"
    nobat = make_cond(s, r, g, cfg, hist, torch.tensor([4]), codes=c)
    assert nobat.shape[1] == 1 + 2 * N_CODES and bool((nobat == cond[:, :1 + 2 * N_CODES]).all()), "the battery maps come last"
    y = torch.arange(2 * N_CODES * 3.0).reshape(2, N_CODES, 3)
    p = pick_tubes(y, torch.tensor([[5, 11], [0, 15]]))
    assert p.shape == (2, 2, 3) and bool((p[0, 0] == y[0, 5]).all()) and bool((p[0, 1] == y[0, 11]).all()) \
        and bool((p[1, 0] == y[1, 0]).all()) and bool((p[1, 1] == y[1, 15]).all()), "a drone's tube is the output map of its code"
    print("codes: each drone on the map its code names, its tube read from the output map of the same code")


if __name__ == "__main__":
    test_codes()


def test_closed_loop_search():
    """Phase 2, step 6: a greedy closed-loop rollout is the closed-loop readout step for step, and it is rollout 0 of a search."""
    from nca import N_CODES, SpaceTimeNCA, cond_channels
    from train import eval_codes, evaluate_closed, rollout_closed
    cfg = Cfg()
    torch.manual_seed(0)
    model = SpaceTimeNCA(C=16, hid=32, n_cond=cond_channels(True, True), n_out=N_CODES)
    torch.nn.init.normal_(model.out.weight, std=0.05)        # the update starts at zero; with weights the tubes are not flat
    gen = torch.Generator().manual_seed(1)
    n, M = 24, 4
    s, r, g = sample_tasks(n, cfg, gen)
    ref = evaluate_closed(model, (s, r, g), cfg, K=6)
    codes = eval_codes(model, n)                             # the codes evaluate_closed drew
    one = rollout_closed(model, s, r, g, cfg, K=6, codes=codes)
    assert bool((one["success"] == ref["success"]).all()) and bool((one["tau"] == ref["tau"]).all())
    live = torch.arange(cfg.T + 1)[None] <= ref["tau"][:, None]
    for d, traj in ((0, one["traj_s"]), (1, one["traj_r"])):
        assert bool(((traj[..., 0] * cfg.W + traj[..., 1]) == ref["hist"][:, d])[live].all()), "the same cells until the episode ends"
    eps = torch.tensor([0.0, 0.1, 0.2, 0.3]).repeat(n)
    greedy = torch.zeros(n * M, dtype=torch.bool); greedy[::M] = True
    shared = torch.zeros(n * M, dtype=torch.bool); shared[3::M] = True
    ro = rollout_closed(model, s, r, g, cfg, K=6, M=M, eps=eps, greedy=greedy, gen=gen, ez=4, shared=shared, codes=codes)
    assert ro["traj_s"].shape == (n * M, cfg.T + 1, 2) and ro["acts"].shape == (n * M, cfg.T, 2) and ro["bat"].shape == (n * M, cfg.T + 1)
    tid = ro["tid"]
    assert bool((tid == torch.arange(n).repeat_interleave(M)).all())
    assert bool((ro["traj_s"][::M] == one["traj_s"]).all()) and bool((ro["traj_r"][::M] == one["traj_r"]).all()) \
        and bool((ro["tau"][::M] == one["tau"]).all()), "rollout 0 of every task is the greedy one"
    succ, tau = replay(s[tid], r[tid], g[tid], ro["acts"], cfg)
    assert bool((succ == ro["success"]).all()) and bool((tau == ro["tau"]).all()), "the recorded actions replay to the same outcome"
    differ = int((ro["traj_s"].reshape(n, M, -1) != ro["traj_s"].reshape(n, M, -1)[:, :1]).any(-1).any(-1).sum())
    print(f"closed-loop search: the greedy rollout equals the closed-loop readout on {n} tasks; exploration changed the seeker's path in {differ} of them")


if __name__ == "__main__":
    test_closed_loop_search()


def test_bound():
    """Phase 2, step 7: a state inside the unit ball is left alone, a larger one is scaled onto it, and the bounded
    rule keeps every voxel there however long it is iterated."""
    from nca import SpaceTimeNCA, bound
    s = torch.zeros(1, 4, 1, 1, 2)
    s[0, :, 0, 0, 0] = torch.tensor([0.5, -0.5, 0.5, -0.5])          # root-mean-square 0.5
    s[0, :, 0, 0, 1] = torch.tensor([3.0, -4.0, 0.0, 0.0])           # root-mean-square 2.5
    b = bound(s)
    assert bool((b[0, :, 0, 0, 0] == s[0, :, 0, 0, 0]).all()), "a small state is unchanged"
    assert torch.allclose(b[0, :, 0, 0, 1], torch.tensor([1.2, -1.6, 0.0, 0.0])), "a large state is divided by its root-mean-square"
    torch.manual_seed(0)
    model = SpaceTimeNCA(C=16, hid=32, n_cond=3, n_out=2, update="bound")
    torch.nn.init.normal_(model.out.weight, std=0.5)                # large updates, so that the bound is reached
    cond = torch.randn(2, 3, 5, 4, 4)
    with torch.no_grad():
        _, state = model(cond, 60)
    rms = state.pow(2).mean(1).sqrt()
    assert float(rms.max()) <= 1.0 + 1e-5 and float(rms.max()) > 0.99, "after 60 iterations the largest voxel sits on the bound"
    print(f"bound: small states unchanged, large ones scaled to 1; after 60 iterations the largest root-mean-square is {float(rms.max()):.4f}")


if __name__ == "__main__":
    test_bound()


def test_kept_state():
    """Phase 2, step 8: learning rebuilds the kept state exactly as closed loop builds it, and a greedy kept rollout
    is the kept closed-loop readout."""
    from nca import N_CODES, SpaceTimeNCA, cond_channels, make_cond, pick_tubes
    from train import eval_codes, evaluate_closed, replay_kept, rollout_closed
    cfg = Cfg()
    torch.manual_seed(0)
    model = SpaceTimeNCA(C=16, hid=32, n_cond=cond_channels(True, True), n_out=N_CODES, update="bound")
    torch.nn.init.normal_(model.out.weight, std=0.05)
    model.kept, model.ks = True, 3
    gen = torch.Generator().manual_seed(2)
    n, K0, ks = 6, 7, 3
    s, r, g = sample_tasks(n, cfg, gen)
    codes = eval_codes(model, n)
    one = rollout_closed(model, s, r, g, cfg, K=K0, codes=codes)                 # greedy, kept: gives realised pasts
    ref = evaluate_closed(model, (s, r, g), cfg, K=K0)
    assert bool((one["success"] == ref["success"]).all()) and bool((one["tau"] == ref["tau"]).all()), "kept rollout = kept readout"
    lab = torch.stack([one["traj_s"][..., 0] * cfg.W + one["traj_s"][..., 1], one["traj_r"][..., 0] * cfg.W + one["traj_r"][..., 1]], 1)
    bat = one["bat"]
    tp = torch.minimum(torch.tensor([0, 1, 2, 3, 4, 2]), (one["tau"] - 1).clamp(min=0))   # pasts of different lengths
    with torch.no_grad():
        got = replay_kept(model, s, r, g, cfg, lab, bat, tp, codes, K0, ks, grad_its=0, amp=False)
        for b in range(n):                                # the same state, built one sample and one step at a time
            sb, rb, gb, cb = s[b:b + 1], r[b:b + 1], g[b:b + 1], codes[b:b + 1]
            state = None
            for t in range(int(tp[b]) + 1):
                cond = make_cond(sb, rb, gb, cfg, lab[b:b + 1], torch.tensor([t]), bat=bat[b:b + 1], battery=True, codes=cb)
                lg, state = model(cond, K0 if t == 0 else ks, s=state)
            assert torch.allclose(got[b], lg[0], atol=1e-5), f"sample {b}: replay differs from stepping, past of {int(tp[b])} steps"
    # gradient reaches the weights through the last iterations
    out = replay_kept(model, s, r, g, cfg, lab, bat, tp, codes, K0, ks, grad_its=4, amp=False)
    pick_tubes(out, codes).sum().backward()
    assert model.perc.weight.grad is not None and float(model.perc.weight.grad.abs().sum()) > 0
    print(f"kept state: learning's replay equals step-by-step growth for pasts of {tp.tolist()} steps; a greedy kept rollout equals the kept readout")


if __name__ == "__main__":
    test_kept_state()
