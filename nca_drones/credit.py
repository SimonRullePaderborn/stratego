"""Interventional credit assignment with the real simulator.

A sparse reward says "this joint rollout worked".  It does not say whose
behaviour made it work.  We ask the simulator: replay the rollout, but replace
ONE drone's actions by uniformly random ones.  If the reward still arrives,
that drone's behaviour was not what earned it.

    credit(drone) = 1 - P(reward | this drone acts at random, the other as recorded)

This is a difference reward estimated by counterfactual replays.  It uses
nothing but `env.replay` (the simulator) and its 0/1 reward: no battery
variable, no distance, no co-location test appears in this file.
"""
import torch

from env import replay


def null_partner_credit(s, r, g, acts, cfg, K, gen):
    """acts: [k, T, 2] recorded joint actions of k rewarded rollouts (column 0 seeker, 1 recharger).

    Returns (credit_seeker [k], credit_recharger [k]) in [0, 1]."""
    k, T, _ = acts.shape
    sK, rK, gK = (x.repeat_interleave(K, 0) for x in (s, r, g))
    out = []
    for drone in (0, 1):
        a = acts.repeat_interleave(K, 0).clone()
        a[:, :, drone] = torch.randint(0, 5, (k * K, T), generator=gen)
        succ, _ = replay(sK, rK, gK, a, cfg)
        out.append(1.0 - succ.float().reshape(k, K).mean(1))
    return out[0], out[1]
