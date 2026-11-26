"""Two-drone rendezvous simulator (vectorised, torch, CPU).

World
-----
* H x W grid (default 8x8), discrete time, horizon T.
* SEEKER drone must reach a static TARGET cell.
* RECHARGER drone is a second, independently steered drone.
* The seeker's battery holds B units and loses one unit EVERY time step
  (hovering costs energy as well).  When seeker and recharger occupy the same
  cell at the same time step the seeker's battery is refilled to B.
* Reward is sparse: 1 when the seeker stands on the target, else 0.
  The episode ends on success, on an empty battery, or at t == T.

What the learner takes from this file: `step` (positions after the move, the
0/1 reward, and whether the episode is over), `cand` (which cell each of the
5 actions leads to) and `replay`.  The battery level is never observed.
The task sampler uses Manhattan distance and B to guarantee feasible tasks and
to define the "recharge-needed" evaluation set; that is part of the
environment, not of the learner.
"""
from dataclasses import dataclass

import torch

# action -> displacement (stay, north, south, west, east)
MOVES = torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=torch.long)


@dataclass
class Cfg:
    H: int = 8
    W: int = 8
    T: int = 20   # horizon (number of steps); the space-time lattice has T+1 slices
    B: int = 5    # seeker battery capacity (steps)


def manhattan(a, b):
    return (a - b).abs().sum(-1)


def sample_tasks(n, cfg, gen, mode="all"):
    """Uniformly random distinct cells for seeker / recharger / target.

    Rejection-filtered so that every task is feasible: if the seeker cannot
    reach the target on one battery (d(s,g) > B) the recharger must start
    within rendezvous range (d(s,r) <= 2B).  With T >= B + (H-1) + (W-1) this
    simple condition is exact, no search is needed to state it.

    mode: "all"  -> every feasible task (training distribution)
          "need" -> only tasks where recharging is REQUIRED (d(s,g) > B)
          "easy" -> only tasks solvable on one battery
    """
    out_s, out_r, out_g = [], [], []
    have = 0
    while have < n:
        m = max(4 * (n - have), 64)
        idx = torch.rand(m, cfg.H * cfg.W, generator=gen).argsort(1)[:, :3]
        cells = torch.stack([idx // cfg.W, idx % cfg.W], -1)  # [m,3,2]
        s, r, g = cells[:, 0], cells[:, 1], cells[:, 2]
        need = manhattan(s, g) > cfg.B
        feas = (~need) | (manhattan(s, r) <= 2 * cfg.B)
        keep = feas
        if mode == "need":
            keep = feas & need
        elif mode == "easy":
            keep = ~need
        out_s.append(s[keep]); out_r.append(r[keep]); out_g.append(g[keep])
        have += int(keep.sum())
    s = torch.cat(out_s)[:n]; r = torch.cat(out_r)[:n]; g = torch.cat(out_g)[:n]
    return s, r, g


class DroneEnv:
    """Batched simulator.  All tensors are [n, ...]."""

    def __init__(self, s, r, g, cfg):
        """g = None runs the same physics without a target ("free flight"): the
        episode then ends only on an empty battery or at t == T."""
        self.cfg = cfg
        self.ps = s.clone()
        self.pr = r.clone()
        self.g = g.clone() if g is not None else None
        n = s.shape[0]
        self.n = n
        self.bat = torch.full((n,), cfg.B, dtype=torch.long)
        self.t = 0
        self.done = torch.zeros(n, dtype=torch.bool)
        self.success = torch.zeros(n, dtype=torch.bool)
        self.tau = torch.zeros(n, dtype=torch.long)       # time of termination
        self.n_recharge = torch.zeros(n, dtype=torch.long)
        self._lim = torch.tensor([cfg.H - 1, cfg.W - 1])

    def cand(self, pos):
        """Cells reached from `pos` by each of the 5 actions (walls block)."""
        c = pos[:, None, :] + MOVES[None]
        return torch.minimum(torch.maximum(c, torch.zeros_like(c)), self._lim.expand_as(c))

    def step(self, a_s, a_r):
        """Advance one time step.  Returns reward [n] (1.0 exactly once on success)."""
        cfg = self.cfg
        act = ~self.done
        ar = torch.arange(self.n)
        new_s = self.cand(self.ps)[ar, a_s]
        new_r = self.cand(self.pr)[ar, a_r]
        self.ps = torch.where(act[:, None], new_s, self.ps)
        self.pr = torch.where(act[:, None], new_r, self.pr)
        self.t += 1
        self.bat = torch.where(act, self.bat - 1, self.bat)
        meet = act & (self.ps == self.pr).all(-1)
        self.bat = torch.where(meet, torch.full_like(self.bat, cfg.B), self.bat)
        self.n_recharge += meet.long()
        at_goal = (self.ps == self.g).all(-1) if self.g is not None else torch.zeros(self.n, dtype=torch.bool)
        succ = act & at_goal
        fail = act & ~at_goal & ((self.bat <= 0) | (self.t >= cfg.T))
        self.success |= succ
        ended = succ | fail
        self.tau = torch.where(ended, torch.full_like(self.tau, self.t), self.tau)
        self.done |= ended
        return succ.float()


def replay(s, r, g, acts, cfg):
    """Replay recorded joint actions [n, T, 2] against target g.  Returns (success, tau).

    This is the reference for hindsight relabelling: a rollout counts as a
    rewarded episode for target g exactly if this function says so."""
    env = DroneEnv(s, r, g, cfg)
    for t in range(acts.shape[1]):
        env.step(acts[:, t, 0], acts[:, t, 1])
    return env.success.clone(), env.tau.clone()
