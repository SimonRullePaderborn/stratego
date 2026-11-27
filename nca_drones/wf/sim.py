"""Batched simulator for the 24x24 / ten-drone programme (benchmark version 0, not frozen).

RUN ON THE CLUSTER ONLY: by the PI's rule no project code is executed anywhere else.

Every field of the state is a tensor with the batch first, so thousands of clones step at once on a GPU.
The rules are those of section 3 of the design page (docs/wavefront_v2_design.html). Values the PI has not
decided yet are settings in Cfg. Not a setting yet: sight and sensing are never blocked by obstacles.

What the learner may use: reset, draw_target, step, peek, observe; for search and credit also select, branch,
new_world and the fields sense_on and ghost. It must not import anything from the benchmark side.
"""
from dataclasses import dataclass, fields, replace

import torch

CAPTURER, RECHARGER, OBSERVER = 0, 1, 2
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))      # stay, north, south, west, east as (row, column)


@dataclass(frozen=True)
class Cfg:
    H: int = 24
    W: int = 24
    capacity: tuple = (12, 0, 24)      # battery in steps per kind (capturer, recharger, observer); 0 means unlimited
    sense: tuple = (3, 2, 5)           # sensing radius per kind, Manhattan distance
    target_sight: int = 2              # the target sees drones within this Manhattan distance
    capture_radius: int = 0            # decided: exact capture; kept as a setting
    swap_captures: bool = True         # decided: exchanging cells with the target is a capture
    empty_battery: str = "grounded"    # decided: an empty drone cannot fly but can be refilled ("removed": gone for good)
    empty_senses: bool = False         # proposal, not decided: an empty drone senses nothing
    empty_captures: bool = False       # proposal, not decided: an empty capturer cannot capture
    empty_is_seen: bool = False        # proposal, not decided: the target ignores an empty drone
    p_move: float = 1.0                # probability that the target acts in a step (decided: as fast as a drone)
    relocate_after: int = 0            # training: an uncaptured target moves to a new cell after this many steps; 0 = never
    T: int = 96                        # the world's tie-break table covers steps 0 .. T
    max_spawns: int = 32               # the world's spawn table covers this many targets


@dataclass
class State:
    obstacle: torch.Tensor     # [B,H,W] bool
    road: torch.Tensor         # [B,H,W] bool; a road cell is never an obstacle
    pos: torch.Tensor          # [B,N,2] long, (row, column)
    kind: torch.Tensor         # [B,N] long
    exists: torch.Tensor       # [B,N] bool: the slot holds a drone
    battery: torch.Tensor      # [B,N] long: steps left; ignored for kinds with unlimited energy
    sense_on: torch.Tensor     # [B,N] bool: False withholds this drone's sensing from the team (credit test)
    ghost: torch.Tensor        # [B,N] bool: a drone the world ignores (credit audit); it moves and nothing else
    tpos: torch.Tensor         # [B,2] long
    tactive: torch.Tensor      # [B] bool: a target exists
    tage: torch.Tensor         # [B] long: steps since the target was drawn
    t: torch.Tensor            # [B] long
    captures: torch.Tensor     # [B] long
    n_spawn: torch.Tensor      # [B] long: how many targets have been drawn (index into spawn_u)
    spawn_u: torch.Tensor      # [B,S] float in [0,1): the world's spawn draws
    tnoise: torch.Tensor       # [B,T+1,6] float in [0,1): the world's tie-breaks for 5 options, and its move draw
    seen: torch.Tensor         # [B,H,W] bool: cells inside some sensing radius after the last step
    target_seen: torch.Tensor  # [B] bool: the target was inside one of them


def _per_kind(values, kind):
    return torch.tensor(values, device=kind.device)[kind]


def _flat(cfg, p):
    return p[..., 0] * cfg.W + p[..., 1]


def _lookup(grid, flat):
    """grid [B,H,W], flat [B,...] long cell indices -> the grid's values at those cells, shaped like flat."""
    B = grid.shape[0]
    return grid.reshape(B, -1).gather(1, flat.reshape(B, -1)).reshape(flat.shape)


def _tables(cfg, rows, seed):
    """One row of random numbers per world, so that a world does not depend on how many others are drawn with it.
    Drawn on the CPU, so that a seed gives the same worlds on every device."""
    S, T1 = cfg.max_spawns, cfg.T + 1
    r = torch.rand(rows, S + T1 * 6, generator=torch.Generator().manual_seed(int(seed)))
    return r[:, :S], r[:, S:].reshape(rows, T1, 6)


def powered(cfg, st):
    """Drones that can fly: they exist and have battery left, or need none. A ghost is always powered."""
    cap = _per_kind(cfg.capacity, st.kind)
    return st.exists & ((cap == 0) | (st.battery > 0) | st.ghost)


def select(st, idx):
    """The episodes idx (a long tensor; repeats allowed) as a new batch."""
    return State(**{f.name: getattr(st, f.name)[idx] for f in fields(st)})


def new_world(cfg, st, seed):
    """Draw afresh everything the team cannot know: where targets appear and how the target breaks ties."""
    spawn_u, tnoise = _tables(cfg, st.kind.shape[0], 2 * int(seed))
    return replace(st, spawn_u=spawn_u.to(st.pos.device), tnoise=tnoise.to(st.pos.device))


def branch(cfg, st, M, W, seed):
    """M x W clones of every episode. Clone (m, w) of episode b sits at index (b * M + m) * W + w.
    The M strategies of an episode all meet the same W worlds; worlds differ between episodes.
    Its random numbers never coincide with those of new_world, whatever the two seeds are."""
    B = st.kind.shape[0]
    dev = st.pos.device
    out = select(st, torch.arange(B, device=dev).repeat_interleave(M * W))
    S, T1 = cfg.max_spawns, cfg.T + 1
    spawn_u, tnoise = _tables(cfg, B * W, 2 * int(seed) + 1)
    spawn_u = spawn_u.to(dev).reshape(B, 1, W, S).expand(B, M, W, S).reshape(B * M * W, S)
    tnoise = tnoise.to(dev).reshape(B, 1, W, T1, 6).expand(B, M, W, T1, 6).reshape(B * M * W, T1, 6)
    return replace(out, spawn_u=spawn_u, tnoise=tnoise)


def moves(cfg, st):
    """For every drone and action: the cell it leads to [B,N,5,2] and whether the move is possible [B,N,5].
    An impossible move leads to the drone's own cell. Capturers may enter free cells, rechargers road cells,
    observers every cell; a drone without power stays where it is."""
    d = torch.tensor(MOVES, device=st.pos.device)                 # [5,2]
    here = st.pos[:, :, None, :]                                  # [B,N,1,2]
    cand = here + d                                               # [B,N,5,2]
    inb = (cand[..., 0] >= 0) & (cand[..., 0] < cfg.H) & (cand[..., 1] >= 0) & (cand[..., 1] < cfg.W)
    safe = torch.where(inb[..., None], cand, here.expand_as(cand))
    flat = _flat(cfg, safe)
    obst, road = _lookup(st.obstacle, flat), _lookup(st.road, flat)
    k = st.kind[:, :, None]
    ok = inb & torch.where(k == OBSERVER, torch.ones_like(inb), torch.where(k == RECHARGER, road, ~obst))
    ok = ok & powered(cfg, st)[:, :, None]
    ok[..., 0] = True                                             # staying is always possible
    return torch.where(ok[..., None], cand, here.expand_as(cand)), ok


def peek(cfg, st):
    """The cell each action of each drone leads to, [B,N,5,2]. This is all the readout needs to know about moving."""
    return moves(cfg, st)[0]


def sense(cfg, st):
    """Cells inside the sensing radius of a drone that has power and whose sensing is not withheld, and whether
    the target is in one of them."""
    B = st.kind.shape[0]
    dev = st.pos.device
    small = torch.int16                                           # [B,N,H,W] is the largest tensor of a step
    r = _per_kind(cfg.sense, st.kind).to(small)[:, :, None, None]
    on = (st.exists & ~st.ghost & st.sense_on & (powered(cfg, st) | cfg.empty_senses))[:, :, None, None]
    yy = torch.arange(cfg.H, device=dev, dtype=small)[None, None, :, None]
    xx = torch.arange(cfg.W, device=dev, dtype=small)[None, None, None, :]
    p = st.pos.to(small)
    man = (yy - p[:, :, 0, None, None]).abs() + (xx - p[:, :, 1, None, None]).abs()                       # [B,N,H,W]
    seen = ((man <= r) & on).any(1)
    ar = torch.arange(B, device=dev)
    return seen, st.tactive & seen[ar, st.tpos[:, 0], st.tpos[:, 1]]


def draw_target(cfg, st, where=None):
    """Draw a new target on a random free cell that holds no drone, in the episodes selected by `where`
    (default: those without a target). The cell comes from the world's spawn table, indexed by how many
    targets have been drawn, so clones of one world that capture at different times use the same draws.
    A target that has just been drawn is not sensed before the next step."""
    B = st.kind.shape[0]
    dev = st.pos.device
    ar = torch.arange(B, device=dev)
    need = ~st.tactive if where is None else where
    present = (st.exists & ~st.ghost).long()
    occupied = torch.zeros(B, cfg.H * cfg.W, dtype=torch.long, device=dev).scatter_add_(1, _flat(cfg, st.pos), present) > 0
    free = ~st.obstacle.reshape(B, -1) & ~occupied
    count = free.sum(1)
    u = st.spawn_u[ar, st.n_spawn.clamp(max=st.spawn_u.shape[1] - 1)]
    k = torch.minimum((u * count).long(), (count - 1).clamp(min=0))
    idx = (free.long().cumsum(1) > k[:, None]).long().argmax(1)   # the (k+1)-th free cell in reading order
    cell = torch.stack([idx // cfg.W, idx % cfg.W], 1)
    ok = need & (count > 0)
    if bool((ok & (st.n_spawn >= st.spawn_u.shape[1])).any()):
        raise RuntimeError("more targets drawn in one episode than cfg.max_spawns")
    return replace(st, tpos=torch.where(ok[:, None], cell, st.tpos), tactive=st.tactive | ok,
                   tage=torch.where(ok, torch.zeros_like(st.tage), st.tage), n_spawn=st.n_spawn + ok.long(),
                   target_seen=st.target_seen & ~ok)


def reset(cfg, obstacle, road, pos, kind, exists, battery, seed):
    """A batch of episodes without a target. `seed` fixes the world. Call draw_target to place the first target."""
    B, N = kind.shape
    dev = pos.device

    def zl(*shape):
        return torch.zeros(*shape, dtype=torch.long, device=dev)

    def zb(*shape):
        return torch.zeros(*shape, dtype=torch.bool, device=dev)

    st = State(obstacle=obstacle, road=road, pos=pos, kind=kind, exists=exists, battery=battery,
               sense_on=~zb(B, N), ghost=zb(B, N), tpos=zl(B, 2), tactive=zb(B), tage=zl(B), t=zl(B),
               captures=zl(B), n_spawn=zl(B), spawn_u=torch.zeros(B, cfg.max_spawns, device=dev),
               tnoise=torch.zeros(B, cfg.T + 1, 6, device=dev), seen=zb(B, cfg.H, cfg.W), target_seen=zb(B))
    st = new_world(cfg, st, seed)
    seen, tseen = sense(cfg, st)
    return replace(st, seen=seen, target_seen=tseen)


def step(cfg, st, actions, respawn=False):
    """One step for the whole batch. actions: [B,N] long in 0..4 (see MOVES).
    Returns the next state and, per episode, whether the target was captured in this step.

    Order: (1) drones and target choose from the same state; (2) all move at once; (3) capture is tested;
    (4) every powered battery loses one unit, a drone on a recharger's cell is refilled, an empty drone is
    grounded; (5) sensing; (6) the target moves to a new cell if it has not been captured for
    cfg.relocate_after steps, and with respawn=True a new target is drawn where there is none."""
    B, N = st.kind.shape
    dev = st.pos.device
    ar = torch.arange(B, device=dev)
    if bool((st.t >= st.tnoise.shape[1]).any()):
        raise RuntimeError("episode longer than cfg.T + 1 steps")
    pw = powered(cfg, st)
    real = st.exists & ~st.ghost

    # (1, 2) drones
    cand, _ = moves(cfg, st)
    new = cand.gather(2, actions[:, :, None, None].expand(B, N, 1, 2)).squeeze(2)               # [B,N,2]

    # (1, 2) target: it chooses from the state before anyone moved
    d = torch.tensor(MOVES, device=dev)
    opt = st.tpos[:, None, :] + d                                                                # [B,5,2]
    inb = (opt[..., 0] >= 0) & (opt[..., 0] < cfg.H) & (opt[..., 1] >= 0) & (opt[..., 1] < cfg.W)
    safe = torch.where(inb[..., None], opt, st.tpos[:, None, :].expand_as(opt))
    legal = inb & ~_lookup(st.obstacle, _flat(cfg, safe))
    legal[:, 0] = True
    vis = real & (pw | cfg.empty_is_seen) & ((st.pos - st.tpos[:, None, :]).abs().sum(-1) <= cfg.target_sight)   # [B,N]
    dist = (opt[:, :, None, :] - st.pos[:, None, :, :]).abs().sum(-1)                            # [B,5,N]
    nearest = torch.where(vis[:, None, :], dist, torch.full_like(dist, cfg.H + cfg.W)).min(-1).values   # [B,5]
    noise = st.tnoise[ar, st.t]                                                                  # [B,6]
    flee = vis.any(1)[:, None]
    score = torch.where(flee, nearest.float(), torch.zeros_like(noise[:, :5])) + 0.5 * noise[:, :5]
    choice = score.masked_fill(~legal, -1e9).argmax(1)       # ties between equal distances go to the larger noise
    acts = st.tactive & (noise[:, 5] < cfg.p_move)
    choice = torch.where(acts, choice, torch.zeros_like(choice))
    tnew = opt[ar, choice]                                                                       # [B,2]

    # (3) capture: same cell after the move, or the two exchanged cells
    capturer = real & (pw | cfg.empty_captures) & (st.kind == CAPTURER)
    hit = (new - tnew[:, None, :]).abs().sum(-1) <= cfg.capture_radius
    if cfg.swap_captures:
        hit = hit | ((new == st.tpos[:, None, :]).all(-1) & (st.pos == tnew[:, None, :]).all(-1) & (new != st.pos).any(-1))
    captured = st.tactive & (capturer & hit).any(1)

    # (4) batteries
    cap = _per_kind(cfg.capacity, st.kind)
    finite = real & (cap > 0)
    battery = torch.where(finite & pw, st.battery - 1, st.battery)
    recharger = real & pw & (st.kind == RECHARGER)
    other = ~torch.eye(N, dtype=torch.bool, device=dev)
    same_cell = (new[:, :, None, :] == new[:, None, :, :]).all(-1) & other                       # [B,N,N]
    refilled = finite & (same_cell & recharger[:, None, :]).any(2)
    battery = torch.where(refilled, cap, battery)
    exists = st.exists
    if cfg.empty_battery == "removed":
        exists = exists & ~(finite & (battery <= 0))
    elif cfg.empty_battery != "grounded":
        raise ValueError(cfg.empty_battery)

    tactive = st.tactive & ~captured
    out = replace(st, pos=new, battery=battery, exists=exists, tpos=tnew, tactive=tactive,
                  tage=st.tage + tactive.long(), t=st.t + 1, captures=st.captures + captured.long())

    # (5) sensing
    seen, tseen = sense(cfg, out)
    out = replace(out, seen=seen, target_seen=tseen)

    # (6) relocation and respawn
    if cfg.relocate_after > 0:
        out = draw_target(cfg, out, out.tactive & (out.tage >= cfg.relocate_after))
    if respawn:
        out = draw_target(cfg, out)
    return out, captured


def observe(cfg, st):
    """What the team knows after a step. Its own drones are always known; the target only if it was sensed.
    A ghost drone is reported like any other: only the world ignores it."""
    B = st.kind.shape[0]
    cap = _per_kind(cfg.capacity, st.kind)
    largest = max(max(cfg.capacity), 1)
    level = torch.where(cap == 0, torch.ones_like(st.battery, dtype=torch.float), st.battery.float() / largest)
    target_map = torch.zeros(B, cfg.H * cfg.W, dtype=torch.bool, device=st.pos.device)
    target_map.scatter_(1, _flat(cfg, st.tpos)[:, None], st.target_seen[:, None])
    return {"pos": st.pos, "kind": st.kind, "exists": st.exists, "powered": powered(cfg, st),
            "battery": level, "seen": st.seen, "target_seen": st.target_seen,
            "target": target_map.view(B, cfg.H, cfg.W)}
