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
