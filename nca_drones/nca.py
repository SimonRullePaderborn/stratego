"""Space-time Neural Cellular Automaton.

One local update rule, shared by every voxel of a (t, x, y) lattice, iterated
K times.  The lattice is told the observation at t = 0 (three one-hot maps in
slice 0: seeker, recharger, target) and, in closed-loop mode only, where the two
drones have actually been in the slices of the past.  Everything else - the
target's persistence through time, any notion of distance, the battery, the
recharge coupling - has to be invented by the rule in its hidden channels.
(From phase 2, step 1 on, the seeker's battery level can be shown as a fourth
map; see make_cond.  The measured runs v10 and v11_noshared did not have it.)

Perception is a plain 3x3x3 convolution with zero padding, followed by a
per-voxel MLP.  No coordinate channels, no pooling, no global operations.
Note that zero padding on a bounded lattice does let a voxel work out where the
walls and the first and last time slice are; position and time are therefore
recoverable even though they are never given.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def voxel_norm(s, eps=1e-3):
    """Per-voxel normalisation across channels (strictly local)."""
    mu = s.mean(1, keepdim=True)
    var = s.var(1, keepdim=True, unbiased=False)
    return (s - mu) / (var + eps).sqrt()


class SpaceTimeNCA(nn.Module):
    def __init__(self, C=16, hid=64, n_cond=3, n_out=2, update="res", fire_rate=1.0):
        super().__init__()
        self.C, self.hid, self.n_cond, self.n_out = C, hid, n_cond, n_out
        self.update = update
        self.fire_rate = fire_rate
        self.perc = nn.Conv3d(C + n_cond, hid, 3, padding=1)
        self.mid = nn.Conv3d(hid, hid, 1)
        if update == "gru":
            self.out = nn.Conv3d(hid, 2 * C, 1)
        else:
            self.out = nn.Conv3d(hid, C, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        self.head = nn.Conv3d(C, n_out, 1)

    def init_state(self, cond, noise=0.0, gen=None):
        n, _, T1, H, W = cond.shape
        s = torch.zeros(n, self.C, T1, H, W)
        if noise > 0:
            s = s + noise * torch.randn(s.shape, generator=gen)
        return s

    def step(self, s, cond):
        if self.update == "gru":
            x = torch.cat([s, cond], 1)
            h = F.relu(self.perc(x))
            h = F.relu(self.mid(h))
            zc = self.out(h)
            z = torch.sigmoid(zc[:, : self.C] - 1.0)
            c = torch.tanh(zc[:, self.C:])
            new = (1 - z) * s + z * c
        else:
            x = torch.cat([voxel_norm(s), cond], 1)
            h = F.relu(self.perc(x))
            h = F.relu(self.mid(h))
            new = s + self.out(h)
        if self.fire_rate < 1.0 and self.training:
            m = (torch.rand(s.shape[0], 1, *s.shape[2:]) < self.fire_rate).float()
            new = m * new + (1 - m) * s
        return new

    def readout(self, s):
        return self.head(s)

    def forward(self, cond, K, s=None, noise=0.0, taps=()):
        """Grow for K iterations.  Returns (logits, state[, tapped logits])."""
        if s is None:
            s = self.init_state(cond, noise)
        tapped = []
        for k in range(K):
            s = self.step(s, cond)
            if (k + 1) in taps:
                tapped.append(self.readout(s))
        logits = self.readout(s)
        if taps:
            return logits, s, tapped
        return logits, s


def make_cond(s, r, g, cfg, hist=None, t0=None, bat=None, battery=False):
    """Observation seed.

    Slice t=0 always holds three one-hot maps: seeker, recharger, target.
    Optionally the REALISED PAST is burned in as well: for every slice
    t <= t0 the seeker / recharger maps hold where the drones actually were
    (hist [n, 2, T+1] cell indices, t0 [n]).  The tissue then has to grow the
    future around a fixed past - nothing else about the world is told to it.

    battery=True adds a fourth map (phase 2, step 1): on the seeker's cell, its
    battery level as a share of the capacity.  Slice 0 holds a full battery,
    which is how every episode starts; the slices of the realised past hold
    the levels the simulator reported (bat [n, T+1], in units of charge).
    """
    n = s.shape[0]
    nc = 4 if battery else 3
    cond = torch.zeros(n, nc, cfg.T + 1, cfg.H, cfg.W)
    ar = torch.arange(n)
    cond[ar, 0, 0, s[:, 0], s[:, 1]] = 1.0
    cond[ar, 1, 0, r[:, 0], r[:, 1]] = 1.0
    cond[ar, 2, 0, g[:, 0], g[:, 1]] = 1.0
    if battery:
        cond[ar, 3, 0, s[:, 0], s[:, 1]] = 1.0
    if hist is not None:
        ts = torch.arange(cfg.T + 1)[None].expand(n, -1)
        m = ts <= t0[:, None]                                  # [n,T+1] slices that are already history
        flat = cond.reshape(n, nc, cfg.T + 1, cfg.H * cfg.W)
        for d in (0, 1):
            oh = F.one_hot(hist[:, d], cfg.H * cfg.W).float() * m[..., None]
            flat[:, d] = torch.maximum(flat[:, d], oh)
        if battery:
            assert bat is not None, "a realised past needs the battery levels the simulator reported"
            level = bat.float() / cfg.B * m                    # [n,T+1], zero outside the realised past
            level[:, 0] = 1.0
            flat[:, 3] = F.one_hot(hist[:, 0], cfg.H * cfg.W).float() * level[..., None]
        cond = flat.reshape(n, nc, cfg.T + 1, cfg.H, cfg.W)
    return cond
