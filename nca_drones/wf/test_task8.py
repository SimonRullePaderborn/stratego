"""The 8x8 task in the new simulator against the old simulator: equal in every field after every step.

RUN ON THE CLUSTER ONLY, from nca_drones/:   python3 -m wf.test_task8

Four sets of episodes: random actions against a target; a scripted pair that walks towards target and partner
with some random steps, so that captures and recharges are frequent (the script is test code and nothing learns
from it); free flight without a target; and free flight with a battery that never runs out.
"""
import torch

import env as old
from wf.task8 import SimEnv

FIELDS = ("ps", "pr", "bat", "done", "success", "tau", "n_recharge")


def _same(a, b, t, what):
    assert a.dtype == b.dtype and a.shape == b.shape, f"{what}: {a.dtype} {tuple(a.shape)} against {b.dtype} {tuple(b.shape)}"
    assert bool((a == b).all()), f"{what} differs at step {t} in {int((a != b).reshape(a.shape[0], -1).any(1).sum())} episodes"


def _run(cfg, s, r, g, policy, what):
    A, B = old.DroneEnv(s, r, g, cfg), SimEnv(s, r, g, cfg)
    for t in range(cfg.T + 1):
        for name in FIELDS:
            _same(getattr(A, name), getattr(B, name), t, f"{what}: {name}")
        assert A.t == B.t == t
        _same(A.cand(A.ps), B.cand(B.ps), t, f"{what}: cells the seeker can reach")
        _same(A.cand(A.pr), B.cand(B.pr), t, f"{what}: cells the recharger can reach")
        if t < cfg.T:
            a_s, a_r = policy(A, t)
            _same(A.step(a_s, a_r), B.step(a_s, a_r), t + 1, f"{what}: reward")
    return A


def _toward(p, q):
    """An action that moves p one cell towards q (rows first), or stays when it is there."""
    dr, dc = q[:, 0] - p[:, 0], q[:, 1] - p[:, 1]
    a = torch.zeros(p.shape[0], dtype=torch.long)
    a = torch.where(dc > 0, torch.full_like(a, 4), a)
    a = torch.where(dc < 0, torch.full_like(a, 3), a)
    a = torch.where(dr > 0, torch.full_like(a, 2), a)
    a = torch.where(dr < 0, torch.full_like(a, 1), a)
    return a


def test_old_and_new_simulator_agree():
    cfg = old.Cfg()
    gen = torch.Generator().manual_seed(0)
    n = 4000

    def random_pair(A, t):
        return torch.randint(0, 5, (n,), generator=gen), torch.randint(0, 5, (n,), generator=gen)

    s, r, g = old.sample_tasks(n, cfg, gen)
    A = _run(cfg, s, r, g, random_pair, "random actions")
    died = int((A.done & ~A.success & (A.tau < cfg.T)).sum())
    assert died >= 100, "the random set must contain empty batteries"

    first_to_partner = torch.rand(n, generator=gen) < 0.5

    def scripted_pair(A, t):
        goal = torch.where((first_to_partner & (A.n_recharge == 0))[:, None], A.pr, g)
        a_s, a_r = _toward(A.ps, goal), _toward(A.pr, A.ps)
        rs, rr = random_pair(A, t)
        return (torch.where(torch.rand(n, generator=gen) < 0.15, rs, a_s),
                torch.where(torch.rand(n, generator=gen) < 0.15, rr, a_r))

    s, r, g = old.sample_tasks(n, cfg, gen)
    B = _run(cfg, s, r, g, scripted_pair, "scripted pair")
    won, met = int(B.success.sum()), int((B.n_recharge > 0).sum())
    assert won >= 500 and met >= 100, f"the scripted set must contain captures and recharges, got {won} and {met}"

    s, r, g = old.sample_tasks(n, cfg, gen)
    _run(cfg, s, r, None, random_pair, "free flight")
    _run(old.Cfg(B=10 ** 6), s, r, None, random_pair, "free flight, battery without end")
    print(f"old and new simulator agree in every field after every step: 4 sets of {n} episodes of {cfg.T} steps "
          f"({died} empty batteries in the random set; {won} captures and {met} episodes with a recharge in the scripted set)")


if __name__ == "__main__":
    test_old_and_new_simulator_agree()
