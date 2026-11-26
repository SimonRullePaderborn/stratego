"""Interventional credit assignment: which parts of a rewarded joint rollout mattered?

A sparse reward only says "this joint rollout worked".  It does not say which
of the two drones' movements caused it.  We ask the simulator: remove a piece
of the rollout and replay it; if the reward still arrives (and no later), that
piece was not needed and is dropped.  What survives is the causally necessary
skeleton of the joint plan.

Interventions tried (all judged only by the simulator's 0/1 reward):
  * recharger freeze: the recharger stops moving after time t* (smallest t* that keeps the reward),
  * seeker loop cut: if the seeker returns to a cell it was on before, drop the loop
    (the rest of its path happens earlier),
  * recharger loop cut: same for the recharger (it arrives earlier and waits).

No distances, no battery knowledge, no solver: an intervention is accepted iff
`arrival_time` (a line-by-line mirror of DroneEnv.step, tested against it)
returns a reward.
"""
import torch

from env import MOVES, DroneEnv


def arrival_time(ps, pr, g, B, T):
    """Mirror of DroneEnv for one episode given cell sequences ps, pr (t = 0..L).

    Returns the time the seeker is rewarded at target g, or -1."""
    bat = B
    for t in range(1, len(ps)):
        bat -= 1
        if ps[t] == pr[t]:
            bat = B
        if ps[t] == g:
            return t
        if bat <= 0 or t >= T:
            return -1
    return -1


def _freeze_recharger(ps, pr, g, B, T):
    tau = len(ps) - 1
    for ts in range(0, tau):
        cand = pr[: ts + 1] + [pr[ts]] * (tau - ts)
        if arrival_time(ps, cand, g, B, T) == tau:
            return cand, ts
    return pr, tau


def _cut_seeker_loops(ps, pr, g, B, T):
    changed = True
    while changed:
        changed = False
        L = len(ps) - 1
        loops = [(t2 - t1, t1, t2) for t1 in range(L) for t2 in range(t1 + 1, L + 1) if ps[t1] == ps[t2]]
        loops.sort(reverse=True)
        for _, t1, t2 in loops:
            nps = ps[: t1 + 1] + ps[t2 + 1:]
            if len(nps) < 2:
                continue
            npr = pr[: len(nps)]
            if arrival_time(nps, npr, g, B, T) == len(nps) - 1:
                ps, pr, changed = nps, npr, True
                break
    return ps, pr


def _cut_recharger_loops(ps, pr, g, B, T):
    tau = len(ps) - 1
    changed = True
    while changed:
        changed = False
        loops = [(t2 - t1, t1, t2) for t1 in range(tau) for t2 in range(t1 + 1, tau + 1) if pr[t1] == pr[t2]]
        loops.sort(reverse=True)
        for _, t1, t2 in loops:
            if all(pr[t] == pr[t1] for t in range(t1, t2 + 1)):
                continue                                   # already a hover
            npr = pr[: t1 + 1] + pr[t2 + 1:]
            npr = npr + [npr[-1]] * (tau + 1 - len(npr))   # arrive earlier, then wait
            if arrival_time(ps, npr, g, B, T) == tau:
                pr, changed = npr, True
                break
    return pr


def prune(ps, pr, g, B, T):
    """ps, pr: python lists of cell ids for t = 0..tau (seeker rewarded at tau on cell g)."""
    assert arrival_time(ps, pr, g, B, T) == len(ps) - 1
    pr, _ = _freeze_recharger(ps, pr, g, B, T)
    ps, pr = _cut_seeker_loops(ps, pr, g, B, T)
    pr, ts = _freeze_recharger(ps, pr, g, B, T)
    pr = _cut_recharger_loops(ps, pr, g, B, T)
    pr, ts = _freeze_recharger(ps, pr, g, B, T)
    return ps, pr, ts


def prune_batch(ro, roll, tau, gcell, cfg):
    """Prune selected rollouts.  Returns labels [k, 2, T+1], new tau [k], recharger stop time [k]."""
    W = cfg.W
    cs = (ro["traj_s"][roll][..., 0] * W + ro["traj_s"][roll][..., 1]).tolist()
    cr = (ro["traj_r"][roll][..., 0] * W + ro["traj_r"][roll][..., 1]).tolist()
    taus = tau.tolist()
    gs = gcell.tolist()
    lab = torch.zeros(len(roll), 2, cfg.T + 1, dtype=torch.long)
    new_tau = torch.zeros(len(roll), dtype=torch.long)
    stop = torch.zeros(len(roll), dtype=torch.long)
    for i in range(len(roll)):
        ps, pr, ts = prune(cs[i][: taus[i] + 1], cr[i][: taus[i] + 1], gs[i], cfg.B, cfg.T)
        L = len(ps) - 1
        lab[i, 0] = torch.tensor(ps + [ps[-1]] * (cfg.T - L))
        lab[i, 1] = torch.tensor(pr + [pr[-1]] * (cfg.T - L))
        new_tau[i] = L
        stop[i] = ts
    return lab, new_tau, stop


def labels_to_actions(lab, cfg):
    """Cell-index labels [k, 2, T+1] -> joint actions [k, T, 2] (for an authoritative replay)."""
    pos = torch.stack([lab // cfg.W, lab % cfg.W], -1)            # [k,2,T+1,2]
    delta = pos[:, :, 1:] - pos[:, :, :-1]                        # [k,2,T,2]
    match = (delta[..., None, :] == MOVES).all(-1)                # [k,2,T,5]
    assert bool(match.any(-1).all()), "illegal move in label"
    return match.float().argmax(-1).permute(0, 2, 1)              # [k,T,2]


def verify_labels(s, r, g, lab, tau, cfg):
    """Authoritative check with the real simulator: the labelled joint plan earns the reward at tau."""
    acts = labels_to_actions(lab, cfg)
    env = DroneEnv(s, r, g, cfg)
    for t in range(cfg.T):
        env.step(acts[:, t, 0], acts[:, t, 1])
    return env.success & (env.tau == tau)
