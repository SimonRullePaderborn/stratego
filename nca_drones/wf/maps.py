"""Maps and start layouts for the simulator (benchmark version 0, not frozen). RUN ON THE CLUSTER ONLY.

One generator family so far: a road network with loops and dead-end spurs, and rectangular obstacles.
Its guarantees hold by construction, without any check:

* roads are connected: a tree is grown over the junctions, one junction at a time, and every later segment
  and spur starts on an existing road cell;
* free space is connected and no free cell has fewer than two free neighbours: every obstacle is a rectangle
  that keeps one cell of distance, diagonals included, from every other obstacle and from the border, and is
  smaller than the map;
* no obstacle covers a road cell.

Not guaranteed: how far a free cell lies from the nearest road. That is measured per family in phase 1.

Further families (city blocks with alleys, braided mazes) need connectivity checks and wait for the PI's
decision on benchmark-side tools.
"""
import math

import torch

from .sim import CAPTURER, MOVES, OBSERVER, RECHARGER


def make_map(H, W, gen, spacing=5, loops=0.35, spurs=2, obstacle_share=0.18, block=(2, 4), tries=400):
    """One map as (obstacle, road), two [H,W] bool tensors. `gen` is a torch.Generator on the CPU."""

    def ri(lo, hi):                                   # a random integer, both ends included
        return int(torch.randint(lo, hi + 1, (1,), generator=gen))

    def lines(n):                                     # junction lines spread from 2 to n - 3, at most `spacing` apart
        count = max(2, math.ceil((n - 5) / spacing) + 1)
        return [round(2 + i * (n - 5) / (count - 1)) for i in range(count)]

    road = torch.zeros(H, W, dtype=torch.bool)
    ys, xs = lines(H), lines(W)
    junction = {(i, j): (min(H - 2, max(1, y + ri(-1, 1))), min(W - 2, max(1, x + ri(-1, 1))))
                for i, y in enumerate(ys) for j, x in enumerate(xs)}
    edges = [((i, j), (i, j + 1)) for i in range(len(ys)) for j in range(len(xs) - 1)]
    edges += [((i, j), (i + 1, j)) for i in range(len(ys) - 1) for j in range(len(xs))]

    def lay(a, b):                                    # a one-cell-wide L between two junctions
        (ya, xa), (yb, xb) = junction[a], junction[b]
        if ri(0, 1):
            road[ya, min(xa, xb):max(xa, xb) + 1] = True
            road[min(ya, yb):max(ya, yb) + 1, xb] = True
        else:
            road[min(ya, yb):max(ya, yb) + 1, xa] = True
            road[yb, min(xa, xb):max(xa, xb) + 1] = True

    # grow a tree: attach one junction that is not yet connected, until all are
    keys = list(junction)
    inside, tree = {keys[ri(0, len(keys) - 1)]}, []
    road[junction[next(iter(inside))]] = True
    while len(inside) < len(keys):
        frontier = [e for e in edges if (e[0] in inside) != (e[1] in inside)]
        e = frontier[ri(0, len(frontier) - 1)]
        tree.append(e)
        inside.update(e)
    for e in edges:                                   # the tree, plus a share of the other segments as loops
        if e in tree or float(torch.rand(1, generator=gen)) < loops:
            lay(*e)
    for _ in range(spurs):                            # dead-end spurs, each starting on a road cell
        cells = road.nonzero()
        y, x = cells[ri(0, len(cells) - 1)].tolist()
        dy, dx = MOVES[ri(1, 4)]
        for _ in range(ri(2, 5)):
            y, x = y + dy, x + dx
            if not (1 <= y < H - 1 and 1 <= x < W - 1):
                break
            road[y, x] = True

    obstacle = torch.zeros(H, W, dtype=torch.bool)
    for _ in range(tries):
        if int(obstacle.sum()) >= obstacle_share * H * W:
            break
        h, w = ri(*block), ri(*block)
        y, x = ri(1, H - 1 - h), ri(1, W - 1 - w)     # at least one free cell to every border
        if road[y:y + h, x:x + w].any() or obstacle[y - 1:y + h + 1, x - 1:x + w + 1].any():
            continue
        obstacle[y:y + h, x:x + w] = True
    return obstacle, road


def make_team(cfg, obstacle, road, gen, n_cap=(1, 5), n_rech=(0, 3), n_obs=(0, 2), n_slots=10, full=False):
    """Kinds, start cells and battery levels of one team: (pos [N,2], kind [N], exists [N], battery [N]).
    Capturers start on free cells, rechargers on road cells, observers anywhere. Batteries are drawn
    uniformly between 2 units and full, or are full. Unused slots do not exist."""

    def ri(lo, hi):
        return int(torch.randint(lo, hi + 1, (1,), generator=gen))

    kinds = [CAPTURER] * ri(*n_cap) + [RECHARGER] * ri(*n_rech) + [OBSERVER] * ri(*n_obs)
    assert len(kinds) <= n_slots
    cells = {CAPTURER: (~obstacle).nonzero(), RECHARGER: road.nonzero(),
             OBSERVER: torch.ones_like(obstacle).nonzero()}
    pos = torch.zeros(n_slots, 2, dtype=torch.long)
    kind = torch.zeros(n_slots, dtype=torch.long)
    exists = torch.zeros(n_slots, dtype=torch.bool)
    battery = torch.zeros(n_slots, dtype=torch.long)
    for i, k in enumerate(kinds):
        pos[i] = cells[k][ri(0, len(cells[k]) - 1)]
        kind[i], exists[i] = k, True
        cap = cfg.capacity[k]
        battery[i] = cap if (full or cap == 0) else ri(min(2, cap), cap)
    return pos, kind, exists, battery


def make_batch(cfg, B, seed, device="cpu", **team):
    """B independent episodes as the arguments of sim.reset, except the world seed: fresh maps and teams."""
    gen = torch.Generator().manual_seed(int(seed))
    rows = []
    for _ in range(B):
        obstacle, road = make_map(cfg.H, cfg.W, gen)
        rows.append((obstacle, road) + make_team(cfg, obstacle, road, gen, **team))
    return tuple(torch.stack(col).to(device) for col in zip(*rows))
