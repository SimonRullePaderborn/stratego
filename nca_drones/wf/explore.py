"""Exploration noise on the readout (design page, section 7). RUN ON THE CLUSTER ONLY.

A strategy is the greedy rule plus one stream of this noise. Per strategy three numbers are drawn: a
temperature, an excursion rate and a longest hold. At every step a drone either starts an excursion, which is
one random action repeated for up to the hold length, or samples its action from its scores, standardised over
its five options and divided by the temperature. Noise is independent between drones. The greedy strategy
takes the best score and never starts an excursion.

The rule never sees any of this: it only changes which action is taken.

Random numbers are drawn on the generator's own device and then moved, so a generator on the CPU gives the
same noise whether the scores live on a CPU or on a GPU, and a run can be replayed on either.
"""
import math
from dataclasses import dataclass, replace

import torch


@dataclass
class Strategy:
    greedy: torch.Tensor    # [B] bool
    tau: torch.Tensor       # [B] float: temperature
    rate: torch.Tensor      # [B] float: chance per drone and step of starting an excursion
    hold: torch.Tensor      # [B] long: longest excursion in steps
    left: torch.Tensor      # [B,N] long: steps left in the running excursion
    held: torch.Tensor      # [B,N] long: the action of the running excursion


def _rand(shape, gen, device):
    return torch.rand(*shape, generator=gen, device=gen.device).to(device)


def draw_strategies(B, N, gen, device, greedy=None, tau=(0.25, 2.0), rate=(0.0, 0.3), hold=(1, 4)):
    """One strategy per clone. `greedy`: [B] bool; default: none is."""
    u = _rand((B, 3), gen, device)
    t = torch.exp(math.log(tau[0]) + u[:, 0] * (math.log(tau[1]) - math.log(tau[0])))       # log-uniform
    r = rate[0] + u[:, 1] * (rate[1] - rate[0])
    span = hold[1] - hold[0]
    h = hold[0] + (u[:, 2] * (span + 1)).long().clamp(max=span)
    zero = torch.zeros(B, N, dtype=torch.long, device=device)
    g = torch.zeros(B, dtype=torch.bool, device=device) if greedy is None else greedy
    return Strategy(greedy=g, tau=t, rate=r, hold=h, left=zero, held=zero.clone())


def choose(strategy, scores, gen):
    """Actions [B,N] for scores [B,N,5], and the strategy with its excursions advanced by one step."""
    B, N, A = scores.shape
    dev = scores.device
    z = (scores - scores.mean(-1, keepdim=True)) / (scores.std(-1, keepdim=True) + 1e-6)
    u = _rand((B, N, A), gen, dev).clamp(1e-9, 1 - 1e-7)
    sampled = (z / strategy.tau[:, None, None] - torch.log(-torch.log(u))).argmax(-1)       # Gumbel-max sampling

    v = _rand((B, N, 3), gen, dev)
    longest = strategy.hold[:, None].expand(B, N)
    start = (strategy.left == 0) & (v[..., 0] < strategy.rate[:, None]) & ~strategy.greedy[:, None]
    left = torch.where(start, torch.minimum(1 + (v[..., 1] * longest).long(), longest), strategy.left)
    held = torch.where(start, (v[..., 2] * A).long().clamp(max=A - 1), strategy.held)

    action = torch.where(left > 0, held, sampled)
    action = torch.where(strategy.greedy[:, None], scores.argmax(-1), action)
    return action, replace(strategy, left=(left - 1).clamp(min=0), held=held)
