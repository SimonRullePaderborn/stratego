"""Hand-worked cases for the simulator and the map generator. RUN ON THE CLUSTER ONLY.

    cd nca_drones && python3 -m wf.test_sim

Every expected value below was worked out by hand from the rules in section 3 of the design page.
These tests have not been executed yet; their first run is the first step of phase 1.
"""
from dataclasses import replace

import torch

from . import explore, maps, sim
from .sim import CAPTURER, OBSERVER, RECHARGER

STAY, NORTH, SOUTH, WEST, EAST = 0, 1, 2, 3, 4


def world(cfg, drones, obstacles=(), roads=(), target=None, seed=0):
    """One hand-made episode. drones: (kind, (row, column), battery) each."""
    n = len(drones)
    obstacle = torch.zeros(1, cfg.H, cfg.W, dtype=torch.bool)
    road = torch.zeros(1, cfg.H, cfg.W, dtype=torch.bool)
    for y, x in obstacles:
        obstacle[0, y, x] = True
    for y, x in roads:
        road[0, y, x] = True
    pos = torch.tensor([[list(p) for _, p, _ in drones]])
    kind = torch.tensor([[k for k, _, _ in drones]])
    battery = torch.tensor([[b for _, _, b in drones]])
    st = sim.reset(cfg, obstacle, road, pos, kind, torch.ones(1, n, dtype=torch.bool), battery, seed)
    if target is not None:
        st = replace(st, tpos=torch.tensor([list(target)]), tactive=torch.ones(1, dtype=torch.bool))
        seen, tseen = sim.sense(cfg, st)
        st = replace(st, seen=seen, target_seen=tseen)
    return st


def act(*a):
    return torch.tensor([list(a)])


def cells(st):
    return st.pos[0].tolist()


def prefer(st, option, t=0):
    """Make the target's tie-break at step t favour one option, and let it act."""
    row = [0.0] * 6
    row[option] = 0.9
    st.tnoise[0, t] = torch.tensor(row)
    return st


def test_moves_by_kind():
    cfg = sim.Cfg(H=6, W=6)
    st = world(cfg, [(CAPTURER, (2, 2), 5), (OBSERVER, (2, 2), 5), (RECHARGER, (5, 0), 0), (CAPTURER, (0, 0), 5)],
               obstacles=[(2, 3)], roads=[(5, x) for x in range(6)])
    assert sim.peek(cfg, st)[0, 0, EAST].tolist() == [2, 2]            # the capturer's way east is an obstacle
    st, _ = sim.step(cfg, st, act(EAST, EAST, NORTH, NORTH))
    # capturer blocked by the obstacle, observer flies onto it, recharger cannot leave the road, nobody leaves the map
    assert cells(st) == [[2, 2], [2, 3], [5, 0], [0, 0]]
    st, _ = sim.step(cfg, st, act(SOUTH, STAY, EAST, SOUTH))
    assert cells(st) == [[3, 2], [2, 3], [5, 1], [1, 0]]


def test_battery_grounding_and_refill():
    cfg = sim.Cfg(H=6, W=6)
    roads = [(5, x) for x in range(6)]
    st = world(cfg, [(CAPTURER, (5, 2), 1), (RECHARGER, (5, 0), 0)], roads=roads)
    st, _ = sim.step(cfg, st, act(STAY, EAST))                         # hovering costs the last unit
    assert st.battery[0, 0] == 0 and st.exists[0, 0] and not sim.powered(cfg, st)[0, 0]
    st, _ = sim.step(cfg, st, act(NORTH, EAST))                        # grounded: cannot fly; the recharger arrives
    assert cells(st) == [[5, 2], [5, 2]] and st.battery[0, 0] == 12
    st, _ = sim.step(cfg, st, act(NORTH, STAY))                        # refilled: flies again
    assert cells(st)[0] == [4, 2] and st.battery[0, 0] == 11

    st = world(cfg, [(CAPTURER, (5, 1), 1), (RECHARGER, (5, 2), 0)], roads=roads)
    st, _ = sim.step(cfg, st, act(EAST, STAY))                         # arrives on the recharger with its last unit
    assert st.battery[0, 0] == 12 and sim.powered(cfg, st)[0, 0]

    gone = replace(cfg, empty_battery="removed")
    st = world(gone, [(CAPTURER, (3, 3), 1), (RECHARGER, (5, 0), 0)], roads=roads)
    st, _ = sim.step(gone, st, act(STAY, STAY))
    assert not st.exists[0, 0]


def test_capture():
    still = sim.Cfg(H=6, W=6, p_move=0.0)                              # a target that never acts
    st = world(still, [(CAPTURER, (3, 2), 5)], target=(3, 3))
    st, captured = sim.step(still, st, act(EAST))
    assert captured[0] and st.captures[0] == 1 and not st.tactive[0]

    st = world(still, [(OBSERVER, (3, 2), 5), (RECHARGER, (3, 4), 0)], roads=[(3, 3), (3, 4)], target=(3, 3))
    st, captured = sim.step(still, st, act(EAST, WEST))                # both land on the target; neither can capture
    assert cells(st) == [[3, 3], [3, 3]] and not captured[0] and st.tactive[0]

    st = world(still, [(CAPTURER, (3, 2), 0)], target=(3, 3))          # an empty capturer cannot fly, so cannot capture
    st, captured = sim.step(still, st, act(EAST))
    assert cells(st) == [[3, 2]] and not captured[0]

    st = world(still, [(CAPTURER, (3, 2), 1)], target=(3, 3))          # a capture on the last unit counts
    st, captured = sim.step(still, st, act(EAST))
    assert captured[0] and st.battery[0, 0] == 0

    blind = sim.Cfg(H=6, W=6, target_sight=0)                          # a target that does not see the capturer next to it
    st = prefer(world(blind, [(CAPTURER, (3, 2), 5)], target=(3, 3)), WEST)
    st, captured = sim.step(blind, st, act(EAST))                      # they exchange cells
    assert captured[0] and cells(st) == [[3, 3]]
    st = prefer(world(replace(blind, swap_captures=False), [(CAPTURER, (3, 2), 5)], target=(3, 3)), WEST)
    st, captured = sim.step(replace(blind, swap_captures=False), st, act(EAST))
    assert not captured[0] and st.tpos[0].tolist() == [3, 2] and cells(st) == [[3, 3]]


def test_target_flees_what_it_sees():
    cfg = sim.Cfg(H=8, W=8)
    # a capturer two cells to the west: staying keeps distance 2, west gives 1, the other three give 3
    st = prefer(world(cfg, [(CAPTURER, (4, 2), 5)], target=(4, 4)), EAST)
    st, _ = sim.step(cfg, st, act(STAY))
    assert st.tpos[0].tolist() == [4, 5]
    # three cells away it is not seen: the target wanders, here towards the drone
    st = prefer(world(cfg, [(CAPTURER, (4, 1), 5)], target=(4, 4)), WEST)
    st, _ = sim.step(cfg, st, act(STAY))
    assert st.tpos[0].tolist() == [4, 3]
    # an empty drone is ignored: the target walks onto its cell and nothing happens
    st = prefer(world(cfg, [(CAPTURER, (4, 3), 0)], target=(4, 4)), WEST)
    st, captured = sim.step(cfg, st, act(STAY))
    assert st.tpos[0].tolist() == [4, 3] and not captured[0]
    # it never enters an obstacle: with obstacles north, south and east of it, only staying keeps it away
    st = prefer(world(cfg, [(CAPTURER, (4, 2), 5)], obstacles=[(3, 4), (5, 4), (4, 5)], target=(4, 4)), EAST)
    st, _ = sim.step(cfg, st, act(STAY))
    assert st.tpos[0].tolist() == [4, 4]


def test_lone_capturer_never_captures():
    """On a map where every free cell has two free neighbours, a single drone in sight cannot capture."""
    cfg = sim.Cfg(H=8, W=8, capacity=(0, 0, 0), T=300)
    B = 64
    g = torch.Generator().manual_seed(1)
    empty = torch.zeros(B, 8, 8, dtype=torch.bool)
    pos = torch.randint(0, 8, (B, 1, 2), generator=g)
    st = sim.reset(cfg, empty, empty.clone(), pos, torch.zeros(B, 1, dtype=torch.long),
                   torch.ones(B, 1, dtype=torch.bool), torch.zeros(B, 1, dtype=torch.long), seed=2)
    st = sim.draw_target(cfg, st)
    assert st.tactive.all()
    for _ in range(300):
        cand = sim.peek(cfg, st)                                               # [B,1,5,2]
        chase = (cand - st.tpos[:, None, None, :]).abs().sum(-1).argmin(-1)    # step towards the target's cell
        st, captured = sim.step(cfg, st, chase)
        assert not captured.any()


def test_two_capturers_close_a_corner():
    cfg = sim.Cfg(H=8, W=8, capacity=(0, 0, 0))
    st = world(cfg, [(CAPTURER, (0, 2), 0), (CAPTURER, (2, 0), 0)], target=(0, 0))
    st, captured = sim.step(cfg, st, act(WEST, NORTH))     # staying keeps distance 2 to both; either move gives 1
    assert not captured[0] and st.tpos[0].tolist() == [0, 0] and cells(st) == [[0, 1], [1, 0]]
    st, captured = sim.step(cfg, st, act(WEST, STAY))      # staying keeps distance 1; either move lands on a drone
    assert captured[0]


def test_sensing():
    cfg = sim.Cfg(H=12, W=12)
    for kind, n in ((CAPTURER, 25), (RECHARGER, 13), (OBSERVER, 61)):          # 2 r (r + 1) + 1 cells for radius r
        st = world(cfg, [(kind, (6, 6), 5)])
        assert int(st.seen.sum()) == n
    assert world(cfg, [(CAPTURER, (6, 6), 5)], target=(6, 9)).target_seen[0]
    assert not world(cfg, [(CAPTURER, (6, 6), 5)], target=(6, 10)).target_seen[0]
    assert world(cfg, [(CAPTURER, (6, 6), 5)], target=(4, 7)).target_seen[0]                 # distance 2 + 1
    st = world(cfg, [(CAPTURER, (6, 6), 5)], target=(6, 9))
    seen, tseen = sim.sense(cfg, replace(st, sense_on=torch.zeros(1, 1, dtype=torch.bool)))  # sensing withheld
    assert int(seen.sum()) == 0 and not tseen[0]
    assert int(world(cfg, [(CAPTURER, (6, 6), 0)]).seen.sum()) == 0                          # an empty drone senses nothing
    obs = sim.observe(cfg, st)
    assert obs["target"][0, 6, 9] and int(obs["target"].sum()) == 1 and abs(float(obs["battery"][0, 0]) - 5 / 24) < 1e-6


def test_draw_target():
    cfg = sim.Cfg(H=6, W=6)
    st = world(cfg, [(CAPTURER, (0, 2), 5)], obstacles=[(0, 0), (0, 1)])
    st.spawn_u[0, 0] = 0.0                                             # the first free cell in reading order
    a = sim.draw_target(cfg, st)
    assert a.tactive[0] and a.tpos[0].tolist() == [0, 3] and a.n_spawn[0] == 1 and a.tage[0] == 0
    assert sim.draw_target(cfg, st).tpos[0].tolist() == [0, 3]         # the same state gives the same draw
    assert sim.draw_target(cfg, a).tpos[0].tolist() == [0, 3] and sim.draw_target(cfg, a).n_spawn[0] == 1   # none needed
    st.spawn_u[0, 0] = 0.999                                           # the last of the 33 free cells
    assert sim.draw_target(cfg, st).tpos[0].tolist() == [5, 5]


def test_relocation_and_respawn():
    cfg = sim.Cfg(H=6, W=6, p_move=0.0, relocate_after=3)
    st = world(cfg, [(CAPTURER, (5, 5), 9)], target=(0, 0))
    for _ in range(2):
        st, _ = sim.step(cfg, st, act(STAY))
    assert st.n_spawn[0] == 0 and st.tage[0] == 2
    st, _ = sim.step(cfg, st, act(STAY))                               # not captured for 3 steps: it is drawn anew
    assert st.n_spawn[0] == 1 and st.tage[0] == 0 and st.tactive[0]

    still = sim.Cfg(H=6, W=6, p_move=0.0)
    st = world(still, [(CAPTURER, (3, 2), 5)], target=(3, 3))
    st.spawn_u[0, 0] = 0.61                                            # the 22nd of the 35 free cells
    st, captured = sim.step(still, st, act(EAST), respawn=True)
    assert captured[0] and st.tactive[0] and st.captures[0] == 1 and st.n_spawn[0] == 1 and not st.target_seen[0]
    assert st.tpos[0].tolist() == [3, 4]                               # the count skips (3, 3), where the drone now stands


def test_ghost_is_ignored():
    still = sim.Cfg(H=6, W=6, p_move=0.0)
    st = world(still, [(CAPTURER, (3, 2), 0), (RECHARGER, (3, 4), 0)], roads=[(3, 3), (3, 4)], target=(3, 3))
    st = replace(st, ghost=torch.ones(1, 2, dtype=torch.bool))
    st, captured = sim.step(still, st, act(EAST, WEST))
    # both ghosts move onto the target: no capture, no refill, no sensing, and a ghost needs no battery to move
    assert cells(st) == [[3, 3], [3, 3]] and not captured[0] and st.battery[0, 0] == 0 and int(st.seen.sum()) == 0

    cfg = sim.Cfg(H=8, W=8)
    st = world(cfg, [(CAPTURER, (4, 3), 5)], target=(4, 4))
    st = prefer(replace(st, ghost=torch.ones(1, 1, dtype=torch.bool)), WEST)
    st, captured = sim.step(cfg, st, act(STAY))                        # the target does not see a ghost next to it
    assert st.tpos[0].tolist() == [4, 3] and not captured[0]


def test_rectangular_grid():
    cfg = sim.Cfg(H=4, W=7)
    st = world(cfg, [(CAPTURER, (3, 6), 5), (CAPTURER, (0, 0), 5)])
    assert st.seen.shape == (1, 4, 7) and int(st.seen.sum()) == 20     # two corner footprints of 4 + 3 + 2 + 1 cells
    st, _ = sim.step(cfg, st, act(SOUTH, NORTH))
    st, _ = sim.step(cfg, st, act(EAST, WEST))
    assert cells(st) == [[3, 6], [0, 0]]                               # nobody leaves the map
    st, _ = sim.step(cfg, st, act(WEST, EAST))
    assert cells(st) == [[3, 5], [0, 1]]
    st.spawn_u[0, 0] = 0.999
    assert sim.draw_target(cfg, st).tpos[0].tolist() == [3, 6]         # the last free cell in reading order


def fails(fn):
    try:
        fn()
    except RuntimeError:
        return True
    return False


def test_tables_do_not_run_out_silently():
    cfg = sim.Cfg(H=6, W=6, T=2)                                       # the world covers steps 0, 1 and 2
    st = world(cfg, [(CAPTURER, (0, 0), 9)])
    for _ in range(3):
        st, _ = sim.step(cfg, st, act(STAY))
    assert fails(lambda: sim.step(cfg, st, act(STAY)))
    one = sim.Cfg(H=6, W=6, max_spawns=1)                              # a world with a single target
    drawn = sim.draw_target(one, world(one, [(CAPTURER, (0, 0), 9)]))
    assert fails(lambda: sim.draw_target(one, replace(drawn, tactive=torch.zeros(1, dtype=torch.bool))))


def test_settings_for_an_empty_drone():
    """The three proposals that are not decided yet, each switched the other way."""
    cfg = sim.Cfg(H=8, W=8)
    empty = [(CAPTURER, (4, 3), 0)]
    can_capture = replace(cfg, empty_captures=True)
    st = prefer(world(can_capture, empty, target=(4, 4)), WEST)
    st, captured = sim.step(can_capture, st, act(STAY))
    assert captured[0]                                                 # the target walks onto it and is caught
    is_seen = replace(cfg, empty_is_seen=True)
    st = prefer(world(is_seen, empty, target=(4, 4)), WEST)
    st, _ = sim.step(is_seen, st, act(STAY))
    assert st.tpos[0].tolist() == [3, 4]                               # it steps away: north is the first of three equals
    assert int(world(sim.Cfg(H=12, W=12, empty_senses=True), [(CAPTURER, (6, 6), 0)]).seen.sum()) == 25


def test_a_recharger_does_not_refill_itself():
    cfg = sim.Cfg(H=6, W=6, capacity=(12, 3, 24))                      # a recharger with a battery, for this test only
    st = world(cfg, [(RECHARGER, (5, 0), 1)], roads=[(5, x) for x in range(6)])
    st, _ = sim.step(cfg, st, act(STAY))
    assert st.battery[0, 0] == 0


def test_exploration_noise():
    gen = torch.Generator().manual_seed(8)
    scores = torch.randn(4, 3, 5, generator=torch.Generator().manual_seed(9))
    greedy = explore.draw_strategies(4, 3, gen, "cpu", greedy=torch.ones(4, dtype=torch.bool))
    action, after = explore.choose(greedy, scores, gen)
    assert torch.equal(action, scores.argmax(-1)) and int(after.left.sum()) == 0     # greedy: best score, no excursion

    always = explore.draw_strategies(64, 3, gen, "cpu", rate=(1.0, 1.0), hold=(3, 3))
    assert (always.hold == 3).all()
    flat = torch.zeros(64, 3, 5)
    first, s1 = explore.choose(always, flat, gen)                      # every drone starts an excursion of 1 to 3 steps
    assert (s1.left >= 0).all() and (s1.left <= 2).all() and torch.equal(first, s1.held)
    second, _ = explore.choose(s1, flat, gen)
    assert (second[s1.left > 0] == first[s1.left > 0]).all()           # a running excursion repeats its action


def test_branch_shares_worlds_and_steps_are_repeatable():
    cfg = sim.Cfg(H=12, W=12)
    st = sim.reset(cfg, *maps.make_batch(cfg, 2, seed=3), seed=4)
    M, W = 3, 2
    br = sim.branch(cfg, st, M, W, seed=5)
    assert br.pos.shape[0] == 2 * M * W
    at = lambda b, m, w: (b * M + m) * W + w
    for b in range(2):
        for w in range(W):
            for m in range(1, M):
                assert torch.equal(br.tnoise[at(b, 0, w)], br.tnoise[at(b, m, w)])
                assert torch.equal(br.spawn_u[at(b, 0, w)], br.spawn_u[at(b, m, w)])
                assert torch.equal(br.pos[at(b, 0, w)], st.pos[b])
        assert not torch.equal(br.tnoise[at(b, 0, 0)], br.tnoise[at(b, 0, 1)])
    assert not torch.equal(br.tnoise[at(0, 0, 0)], br.tnoise[at(1, 0, 0)])
    assert not torch.equal(sim.branch(cfg, st, 1, 1, seed=4).tnoise, st.tnoise)    # a branch never repeats the reset's world

    br = sim.draw_target(cfg, br)
    actions = torch.randint(0, 5, br.kind.shape, generator=torch.Generator().manual_seed(6))
    one, _ = sim.step(cfg, br, actions, respawn=True)
    two, _ = sim.step(cfg, br, actions, respawn=True)
    assert all(torch.equal(getattr(one, f), getattr(two, f)) for f in ("pos", "battery", "tpos", "seen", "captures"))


def connected(mask):
    """Flood fill from one true cell; true if it reaches all of them. Test-side only."""
    todo = [tuple(mask.nonzero()[0].tolist())]
    found = set(todo)
    while todo:
        y, x = todo.pop()
        for dy, dx in sim.MOVES[1:]:
            c = (y + dy, x + dx)
            if 0 <= c[0] < mask.shape[0] and 0 <= c[1] < mask.shape[1] and bool(mask[c]) and c not in found:
                found.add(c)
                todo.append(c)
    return len(found) == int(mask.sum())


def test_maps_and_teams():
    for side in (12, 16, 24):
        cfg = sim.Cfg(H=side, W=side)
        gen = torch.Generator().manual_seed(side)
        for _ in range(20):
            obstacle, road = maps.make_map(side, side, gen)
            free = ~obstacle
            assert not (obstacle & road).any() and road.any()
            assert connected(free) and connected(road)
            padded = torch.zeros(side + 2, side + 2, dtype=torch.long)
            padded[1:-1, 1:-1] = free.long()
            neighbours = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            assert (neighbours[free] >= 2).all()                       # no dead end anywhere
            pos, kind, exists, battery = maps.make_team(cfg, obstacle, road, gen)
            for i in range(len(kind)):
                if not exists[i]:
                    continue
                y, x = pos[i].tolist()
                if kind[i] == CAPTURER:
                    assert free[y, x] and 2 <= battery[i] <= 12
                if kind[i] == RECHARGER:
                    assert road[y, x]
                if kind[i] == OBSERVER:
                    assert 2 <= battery[i] <= 24
            assert 1 <= int((exists & (kind == CAPTURER)).sum()) <= 5
        if side == 24:
            assert obstacle.any()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
