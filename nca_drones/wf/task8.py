"""The measured 8x8 task, played in the new simulator (PROGRAMME.md, sections 12.2 and 12.4).

RUN ON THE CLUSTER ONLY: by the PI's rule no project code is executed anywhere else.

SimEnv has the interface of env.DroneEnv, the simulator of the 8x8 work: one seeker with a battery of cfg.B steps,
one recharger without a battery, a target that does not move, a reward when the seeker stands on the target, and
an episode that is over on the reward, on an empty battery, or at t == cfg.T.  Here every rule of the world is
executed by wf/sim.py: the seeker is a capturer, every cell is a road cell so that the recharger may go anywhere,
the target never acts (p_move = 0), and a drone whose battery is empty is removed.  What sim.py does not have is
the end of an episode.  SimEnv keeps that part: an episode that is over is frozen, as in DroneEnv.

wf/test_task8.py checks that the two simulators agree in every field after every step.
"""
from dataclasses import fields, replace

import torch

from . import sim


def _keep(mask, new, old):
    """Per episode: `new` where mask is set, `old` elsewhere, for every field of the state."""
    out = {}
    for f in fields(old):
        a, b = getattr(new, f.name), getattr(old, f.name)
        out[f.name] = torch.where(mask.reshape(-1, *([1] * (a.dim() - 1))), a, b)
    return sim.State(**out)


class SimEnv:
    """Batched 8x8 task on wf/sim.py.  All tensors are [n, ...]; fields and methods as in env.DroneEnv."""

    def __init__(self, s, r, g, cfg):
        """g = None runs the same physics without a target: the episode then ends only on an empty battery or at t == T."""
        n = s.shape[0]
        self.cfg, self.n = cfg, n
        self.scfg = sim.Cfg(H=cfg.H, W=cfg.W, capacity=(cfg.B, 0, 0), sense=(cfg.H + cfg.W,) * 3,
                            empty_battery="removed", p_move=0.0, T=cfg.T, max_spawns=1)
        none = torch.zeros(n, cfg.H, cfg.W, dtype=torch.bool)
        kind = torch.tensor([sim.CAPTURER, sim.RECHARGER]).expand(n, 2).clone()
        battery = torch.tensor([cfg.B, 0]).expand(n, 2).clone()
        st = sim.reset(self.scfg, obstacle=none, road=~none, pos=torch.stack([s, r], 1).clone(), kind=kind,
                       exists=torch.ones(n, 2, dtype=torch.bool), battery=battery, seed=0)
        if g is not None:                       # the target is placed, not drawn: the task names its cell
            st = replace(st, tpos=g.clone(), tactive=torch.ones(n, dtype=torch.bool))
        self.st = st
        self.g = g.clone() if g is not None else None
        self.ps, self.pr = s.clone(), r.clone()
        self.bat = torch.full((n,), cfg.B, dtype=torch.long)
        self.t = 0
        self.done = torch.zeros(n, dtype=torch.bool)
        self.success = torch.zeros(n, dtype=torch.bool)
        self.tau = torch.zeros(n, dtype=torch.long)       # time of termination
        self.n_recharge = torch.zeros(n, dtype=torch.long)

    def cand(self, pos):
        """Cells reached from `pos` by each of the 5 actions, by the simulator's rule for a drone with power."""
        n = pos.shape[0]
        probe = replace(self.st, pos=torch.stack([pos, pos], 1), exists=torch.ones(n, 2, dtype=torch.bool),
                        battery=torch.full((n, 2), max(self.cfg.B, 1), dtype=torch.long))
        return sim.peek(self.scfg, probe)[:, 0]

    def step(self, a_s, a_r):
        """Advance one time step.  Returns reward [n] (1.0 exactly once on success)."""
        cfg = self.cfg
        act = ~self.done
        nxt, captured = sim.step(self.scfg, self.st, torch.stack([a_s, a_r], 1))
        self.st = _keep(act, nxt, self.st)                 # an episode that is over does not move on
        self.ps, self.pr = self.st.pos[:, 0].clone(), self.st.pos[:, 1].clone()
        self.t += 1
        self.bat = self.st.battery[:, 0].clone()
        meet = act & (self.ps == self.pr).all(-1)
        self.n_recharge += meet.long()
        succ = act & captured
        fail = act & ~succ & ((self.bat <= 0) | (self.t >= cfg.T))
        self.success |= succ
        ended = succ | fail
        self.tau = torch.where(ended, torch.full_like(self.tau, self.t), self.tau)
        self.done |= ended
        return succ.float()
