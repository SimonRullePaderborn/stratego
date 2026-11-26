"""Builds unified-wavefront-design.html (figures are generated here so every coordinate is exact)."""
import math
import sys

# usage: python3 build_design_page.py [out.html] [--standalone]
#   --standalone wraps the page in a full HTML document (the published page is a fragment that the host wraps itself)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
STANDALONE = "--standalone" in sys.argv
OUT = ARGS[0] if ARGS else "unified_wavefront_design.html"
HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<style>:root{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pre(lines, col):
    """Pseudo-code block; (code, comment) pairs get their comments aligned in one column."""
    out = []
    for ln in lines:
        if isinstance(ln, tuple):
            out.append(esc(ln[0].ljust(col)) + '<span class="c">' + esc(ln[1]) + "</span>")
        else:
            out.append(esc(ln))
    return "<pre>" + "\n".join(out) + "</pre>"


PRE_RULE = pre([
    ("x  = [ normalise(S), X ]", "per voxel; normalise acts across the C state channels"),
    ("h  = relu( conv 3×3×3 (x) )", "zero padding; the only place neighbours are seen"),
    ("S  ← S + W3 · relu( W2 · h )", "W2 and W3 act on one voxel at a time"),
    ("tube, forecast = linear readouts of S", "one voxel at a time"),
], 41)

PRE_ACT = pre([
    "start of an episode",
    "    S ← 0, X ← 0",
    "    clamp the first observation into the present slice",
    "    run the rule K0 times",
    "",
    "every step t",
    "    for each controlled drone i, in a freshly shuffled order",
    "        for each action a of drone i",
    ("            u[a]     ← sim.peek(i, a)", "the drone's own next state"),
    "            score[a] ← tube[type(i)][present + 1][u[a]]",
    ("        a_i ← argmax over a of  score[a] / τ + noise(i, t, a)", "τ = 0: plain argmax, no noise"),
    ("        clamp fact[type(i)][present + 1][u[a_i]] ← 1", "the commitment"),
    "        run the rule k_c times",
    "    observation, reward, done ← sim.step(a_1 … a_n)",
    "    slide S and X one slice towards the past",
    ("    clamp the observation into the present slice", "replaces the commitments"),
    "    run the rule k_s times",
], 65)

PRE_SEARCH = pre([
    "for each task, drawn uniformly",
    "    worlds     ← W seeds",
    "    strategies ← the greedy one and M − 1 sampled ones",
    "    for each strategy σ and world ω",
    "        G[σ, ω] ← return of the clone (σ, ω)",
    "    recombination, simulator only",
    "        mix the drones' recorded action lists across clones",
    "        replay every mix in all W worlds; each mix is one more strategy",
    "    V[σ] ← mean of G[σ, ω] over ω",
    "    best ← argmax V",
    ("    if V[best] > V[greedy]       keep best", "search beat the current rule"),
    ("    else if V[greedy] > 0        keep greedy", "consolidate what already works"),
    "    else                         keep nothing",
], 48)

PRE_LEARN = pre([
    "repeat",
    "    take a batch of stored episodes",
    "    replay each one: same task, same world, stored actions and commit order,",
    "                     lattice run by the protocol of §5 with K0, k_c, k_s redrawn",
    "    every n_bp steps",
    ("        loss ← tube loss + λ · forecast loss", "on the slices ahead of the present"),
    "        back-propagate through the rule iterations of these n_bp steps only",
    "        update the weights, detach the lattice state, continue the replay",
], 48)


# ----------------------------------------------------------------------------- figure 1: the lattice around the front
def fig1():
    W, H = 920, 216
    tile, gap = 64, 12
    pitch = tile + gap
    x0, y0 = 46, 86
    cell = tile / 4
    labels = ["t−4", "t−3", "t−2", "t−1", "t", "t+1", "t+2", "t+3", "t+4", "t+5", "t+6"]
    S = [(0, 3), (0, 2), (1, 2), (1, 1), (2, 1)]
    R = [(3, 3), (3, 2), (3, 2), (3, 1), (3, 1)]
    G = (3, 0)
    fused = {5: ((2, 1), .9), 6: ((2, 0), .7), 7: ((3, 0), .55)}
    faint = {5: [("a", (2, 0))], 6: [("a", (3, 0)), ("b", (3, 1))]}
    xe = x0 + 11 * pitch - gap

    def ctr(i, c):
        return x0 + i * pitch + (c[0] + .5) * cell, y0 + (c[1] + .5) * cell

    def diamond(cx, cy, dashed):
        d = f"M{cx:.1f} {cy-7:.1f}L{cx+7:.1f} {cy:.1f}L{cx:.1f} {cy+7:.1f}L{cx-7:.1f} {cy:.1f}Z"
        extra = ' stroke-dasharray="3 2.5"' if dashed else ""
        return f'<path d="{d}" fill="var(--surface)" stroke="currentColor" stroke-width="1.6"{extra}/>'

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Eleven time slices in a row. The four slices behind the present hold facts: a seeker and a recharger approaching each other and a fixed target. The present slice is the front. The six slices ahead hold the dream: the two tubes merge at t+1 and reach the target at t+3, and the target is shown as a forecast.">']
    o.append('<defs><marker id="f1arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>')
    o.append(f'<text x="{x0}" y="15" font-size="12.5" fill="currentColor">time: at every step the window moves one slice to the right</text>')
    o.append(f'<line x1="{x0}" y1="27" x2="{xe}" y2="27" stroke="currentColor" stroke-width="1.2" opacity=".65" marker-end="url(#f1arr)"/>')

    def bracket(i0, i1, text):
        xa, xb = x0 + i0 * pitch, x0 + i1 * pitch + tile
        o.append(f'<path d="M{xa} 72V64H{xb}V72" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".55"/>')
        o.append(f'<text x="{(xa+xb)/2}" y="56" font-size="12.5" text-anchor="middle" fill="currentColor">{text}</text>')

    bracket(0, 3, "facts: clamped")
    bracket(5, 10, "dream: grown by the rule")
    xp = x0 + 4 * pitch + tile / 2
    o.append(f'<text x="{xp}" y="56" font-size="12.5" font-weight="600" text-anchor="middle" style="fill:var(--accent)">front</text>')
    o.append(f'<path d="M{xp-5} 63L{xp+5} 63L{xp} 72Z" style="fill:var(--accent)"/>')

    for i in range(11):
        x = x0 + i * pitch
        if i < 4:
            o.append(f'<rect x="{x}" y="{y0}" width="{tile}" height="{tile}" rx="4" style="fill:var(--tile)" stroke="currentColor" stroke-opacity=".4" stroke-width="1.2"/>')
        elif i == 4:
            o.append(f'<rect x="{x}" y="{y0}" width="{tile}" height="{tile}" rx="4" style="fill:var(--tile);stroke:var(--accent)" stroke-width="2.6"/>')
        else:
            o.append(f'<rect x="{x}" y="{y0}" width="{tile}" height="{tile}" rx="4" fill="none" stroke="currentColor" stroke-opacity=".45" stroke-width="1.2" stroke-dasharray="4 3"/>')
        for k in (1, 2, 3):
            o.append(f'<line x1="{x+k*cell}" y1="{y0+3}" x2="{x+k*cell}" y2="{y0+tile-3}" stroke="currentColor" stroke-opacity=".11"/>')
            o.append(f'<line x1="{x+3}" y1="{y0+k*cell}" x2="{x+tile-3}" y2="{y0+k*cell}" stroke="currentColor" stroke-opacity=".11"/>')
        gx, gy = ctr(i, G)
        o.append(diamond(gx, gy, dashed=i > 4))
        if i <= 4:
            sx, sy = ctr(i, S[i]); rx, ry = ctr(i, R[i])
            o.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6.5" style="fill:var(--a);stroke:var(--surface)" stroke-width="2"/>')
            o.append(f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="6.5" style="fill:var(--b);stroke:var(--surface)" stroke-width="2"/>')
        for which, c in faint.get(i, []):
            fx, fy = ctr(i, c)
            if which == "a":
                o.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="6" style="fill:var(--a)" opacity=".22"/>')
            else:
                o.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="6" fill="none" style="stroke:var(--b)" stroke-width="2.5" opacity=".3"/>')
        if i in fused:
            c, op = fused[i]
            fx, fy = ctr(i, c)
            o.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="5.5" style="fill:var(--a)" opacity="{op}"/>')
            o.append(f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="9" fill="none" style="stroke:var(--b)" stroke-width="2.6" opacity="{op}"/>')
        o.append(f'<text x="{x+tile/2}" y="{y0+tile+18}" font-size="11.5" text-anchor="middle" fill="currentColor" class="mono" opacity=".8">{labels[i]}</text>')

    ly = y0 + tile + 52
    o.append(f'<circle cx="{x0+7}" cy="{ly-4}" r="6" style="fill:var(--a)"/><text x="{x0+20}" y="{ly}" font-size="12" fill="currentColor">seeker</text>')
    o.append(f'<circle cx="{x0+97}" cy="{ly-4}" r="6" style="fill:var(--b)"/><text x="{x0+110}" y="{ly}" font-size="12" fill="currentColor">recharger</text>')
    o.append(diamond(x0 + 207, ly - 4, False) + f'<text x="{x0+220}" y="{ly}" font-size="12" fill="currentColor">target</text>')
    o.append(f'<circle cx="{x0+317}" cy="{ly-4}" r="6" fill="currentColor"/><text x="{x0+330}" y="{ly}" font-size="12" fill="currentColor">solid: a fact</text>')
    o.append(f'<circle cx="{x0+447}" cy="{ly-4}" r="4.5" fill="currentColor" opacity=".6"/><circle cx="{x0+447}" cy="{ly-4}" r="8" fill="none" stroke="currentColor" stroke-width="2.2" opacity=".6"/>'
             f'<text x="{x0+463}" y="{ly}" font-size="12" fill="currentColor">disc in a ring: both tubes on one cell</text>')
    o.append(diamond(x0 + 717, ly - 4, True) + f'<text x="{x0+730}" y="{ly}" font-size="12" fill="currentColor">dashed: a forecast</text>')
    o.append("</svg>")
    return "\n".join(o)


# ----------------------------------------------------------------------------- figure 2: strategies x worlds, then lesions
def fig2():
    W, H = 920, 292
    lx, ly, cp, rp = 172, 86, 40, 34
    rows = [("greedy", [1, 0, 0, 1]), ("sampled 1", [0, 0, 0, 0]), ("sampled 2", [1, 1, 0, 1]),
            ("sampled 3", [1, 0, 0, 0]), ("sampled 4", [0, 1, 0, 1]), ("recombined", [0, 0, 1, 0])]
    kept = 2
    les = [("seeker random", [0, 0, 0, 0]), ("recharger random", [0, 0, 0, 1]), ("observer random", [1, 0, 0, 1])]
    rx0 = 668
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Left: six strategies are each run in the same four worlds. The strategy with the highest average return, 0.75, is kept. Right: the kept strategy is run again three times, each time with one drone acting at random. The drop in average return gives that drone its credit: 1.00 for the seeker, 0.67 for the recharger, 0.33 for the observer.">']
    o.append('<defs><marker id="f2arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>')

    def cellmark(cx, cy, on):
        if on:
            return f'<circle cx="{cx}" cy="{cy}" r="8" fill="currentColor"/>'
        return f'<circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="currentColor" stroke-opacity=".4" stroke-width="1.4"/>'

    def header(x_first, x_val, extra=None):
        for j in range(4):
            o.append(f'<text x="{x_first+j*cp}" y="{ly-24}" font-size="11.5" text-anchor="middle" fill="currentColor" class="mono" opacity=".8">w{j+1}</text>')
        o.append(f'<text x="{x_val}" y="{ly-24}" font-size="11.5" text-anchor="middle" fill="currentColor" opacity=".8">value</text>')
        if extra:
            o.append(f'<text x="{extra}" y="{ly-24}" font-size="11.5" text-anchor="middle" fill="currentColor" opacity=".8">credit</text>')

    o.append(f'<text x="64" y="22" font-size="13" font-weight="600" fill="currentColor">Search: every strategy meets the same worlds</text>')
    o.append(f'<text x="536" y="22" font-size="13" font-weight="600" fill="currentColor">Credit: the kept strategy with one drone lesioned</text>')
    header(lx, lx + 4 * cp + 14)
    header(rx0, rx0 + 4 * cp + 10, rx0 + 4 * cp + 66)

    ky = ly + kept * rp
    o.append(f'<rect x="{lx-22}" y="{ky-15}" width="{4*cp+58}" height="30" rx="6" style="fill:var(--accent-soft);stroke:var(--accent)" stroke-width="1.8"/>')
    for i, (name, vals) in enumerate(rows):
        cy = ly + i * rp
        o.append(f'<text x="{lx-32}" y="{cy+4}" font-size="12.5" text-anchor="end" fill="currentColor">{name}</text>')
        for j, v in enumerate(vals):
            o.append(cellmark(lx + j * cp, cy, v))
        o.append(f'<text x="{lx+4*cp+14}" y="{cy+4}" font-size="12.5" text-anchor="middle" fill="currentColor" class="mono">{sum(vals)/4:.2f}</text>')
    o.append(f'<text x="{lx+4*cp+48}" y="{ky+4}" font-size="12.5" font-weight="600" style="fill:var(--accent)">kept</text>')
    ax0, ax1 = lx + 4 * cp + 86, 528
    o.append(f'<line x1="{ax0}" y1="{ky}" x2="{ax1}" y2="{ky}" stroke="currentColor" stroke-width="1.4" marker-end="url(#f2arr)"/>')
    o.append(f'<text x="{(ax0+ax1)/2}" y="{ky-9}" font-size="12" text-anchor="middle" fill="currentColor">run again</text>')

    base = 0.75
    for i, (name, vals) in enumerate(les):
        cy = ly + (i + 1) * rp
        v = sum(vals) / 4
        o.append(f'<text x="{rx0-24}" y="{cy+4}" font-size="12.5" text-anchor="end" fill="currentColor">{name}</text>')
        for j, x in enumerate(vals):
            o.append(cellmark(rx0 + j * cp, cy, x))
        o.append(f'<text x="{rx0+4*cp+10}" y="{cy+4}" font-size="12.5" text-anchor="middle" fill="currentColor" class="mono">{v:.2f}</text>')
        o.append(f'<text x="{rx0+4*cp+66}" y="{cy+4}" font-size="12.5" font-weight="600" text-anchor="middle" fill="currentColor" class="mono">{(base-v)/base:.2f}</text>')
    yl = ly + 5 * rp + 4
    o.append(cellmark(544, yl - 4, 1) + f'<text x="558" y="{yl}" font-size="12" fill="currentColor">reward reached in this world</text>')
    o.append(cellmark(744, yl - 4, 0) + f'<text x="758" y="{yl}" font-size="12" fill="currentColor">no reward</text>')
    o.append("</svg>")
    return "\n".join(o)


# ----------------------------------------------------------------------------- chart: search yield vs budget (measured)
XS = [16, 64, 256, 1024, 4096]
SERIES = [("a", "Independent noise, joint rollouts", [0.75, 1.75, 7.0, 17.25, 37.5], "independent noise"),
          ("b", "Shared noise, joint rollouts", [2.5, 9.0, 29.0, 45.25, 65.5], "shared noise"),
          ("c", "Independent noise, all pairs recombined", [6.0, 22.5, 58.5], "recombined pairs")]
PL, PR, PT, PB = 64, 520, 28, 308


def X(v):
    return PL + (math.log2(v) - 4) / 8 * (PR - PL)


def Y(p):
    return PB - p / 70 * (PB - PT)


def chart():
    W, H = 760, 356
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Line chart. Share of 400 recharge-needed tasks for which search finds at least one rewarded plan before any training, against the number of plans sampled per drone, 16 to 4,096 on a log scale. Independent noise rises from 0.8 to 37.5 percent, shared noise from 2.5 to 65.5 percent, and recombining all pairs of independently sampled plans from 6.0 percent at 16 to 58.5 percent at 256.">']
    for p in (0, 20, 40, 60):
        o.append(f'<line x1="{PL}" y1="{Y(p):.1f}" x2="{PR}" y2="{Y(p):.1f}" style="stroke:var(--{"line-2" if p == 0 else "line"})" stroke-width="1"/>')
        o.append(f'<text x="{PL-10}" y="{Y(p)+4:.1f}" font-size="12" text-anchor="end" style="fill:var(--ink-2)" class="num">{p}%</text>')
    for v in XS:
        o.append(f'<text x="{X(v):.1f}" y="{PB+19}" font-size="12" text-anchor="middle" style="fill:var(--ink-2)" class="num">{v:,}</text>')
    o.append(f'<text x="{(PL+PR)/2}" y="{PB+41}" font-size="12" text-anchor="middle" style="fill:var(--ink-2)">plans sampled per drone and task (log scale)</text>')
    o.append(f'<line class="hair" x1="{PL}" y1="{PT}" x2="{PL}" y2="{PB}" style="stroke:var(--line-2)" stroke-width="1" opacity="0"/>')
    ends = {"a": (PR + 12, Y(37.5) + 4), "b": (PR + 12, Y(65.5) + 4), "c": (X(256) + 12, Y(58.5) - 4)}
    for key, name, vals, short in SERIES:
        pts = " ".join(f"{X(XS[i]):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        o.append(f'<polyline points="{pts}" fill="none" style="stroke:var(--{key})" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(vals):
            o.append(f'<circle cx="{X(XS[i]):.1f}" cy="{Y(v):.1f}" r="4.5" style="fill:var(--{key});stroke:var(--surface)" stroke-width="2"/>')
        ex, ey = ends[key]
        o.append(f'<text x="{ex:.1f}" y="{ey:.1f}" font-size="12.5" style="fill:var(--ink)">{short}</text>')
    half = (X(64) - X(16)) / 2
    for i, v in enumerate(XS):
        vals = "; ".join(f"{name}: {vv[i]:.1f}%" for _, name, vv, _ in SERIES if i < len(vv))
        o.append(f'<rect class="band" data-i="{i}" tabindex="0" x="{X(v)-half:.1f}" y="{PT}" width="{2*half:.1f}" height="{PB-PT}" fill="transparent" aria-label="{v:,} plans per drone. {vals}"/>')
    o.append("</svg>")
    return "\n".join(o)


PAGE = r'''<title>Unified Wavefront Design</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
/* Layout: one reading column (68ch); tables, code and figures break out to the full 1040px wrapper. */
:root{
  --bg:#f5f6f8; --surface:#ffffff; --ink:#161a22; --ink-2:#55606f; --line:#dce1e8; --line-2:#b8c0cc;
  --accent:#2a52be; --accent-soft:#e7ecfa; --code:#eceff4; --tile:#eceff4;
  --a:#257de0; --b:#f47230; --c:#00b894;
  --font-body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0e1116; --surface:#161a21; --ink:#e6e9ef; --ink-2:#98a2b3; --line:#262c36; --line-2:#3b4452;
    --accent:#8fb0ff; --accent-soft:#1a2440; --code:#1a1f28; --tile:#1d222b;
    --a:#338ced; --b:#e76624; --c:#00a888; color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#0e1116; --surface:#161a21; --ink:#e6e9ef; --ink-2:#98a2b3; --line:#262c36; --line-2:#3b4452;
  --accent:#8fb0ff; --accent-soft:#1a2440; --code:#1a1f28; --tile:#1d222b;
  --a:#338ced; --b:#e76624; --c:#00a888; color-scheme:dark;
}
body{background:var(--bg);color:var(--ink);font-family:var(--font-body);font-size:16px;line-height:1.6}
.wrap{max-width:1040px;margin:0 auto;padding-block:44px 88px;padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px))}
h1,h2,h3{font-weight:600;line-height:1.2;text-wrap:balance;margin:0}
h1{font-size:clamp(1.9rem,4.4vw,2.6rem);letter-spacing:-.01em}
h2{font-size:1.4rem;margin-top:3.4rem;padding-top:1.1rem;border-top:1px solid var(--line-2)}
h3{font-size:1.02rem;margin-top:1.9rem}
h2 .sec{font-family:var(--font-mono);font-weight:500;font-size:.95rem;color:var(--ink-2);margin-right:.6rem}
p,ul,ol{max-width:68ch;margin:.85rem 0}
ul,ol{padding-left:1.3rem}
li{margin:.4rem 0}
a{color:var(--accent)}
a:focus-visible,.band:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.eyebrow{font-family:var(--font-mono);font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2);margin:0 0 .9rem}
.lede{font-size:1.14rem;line-height:1.55;margin-top:1.1rem}
.status{display:flex;gap:14px;align-items:baseline;border:1px solid var(--line-2);background:var(--surface);border-radius:6px;padding:12px 16px;margin:1.6rem 0;max-width:68ch}
.status .label{font-family:var(--font-mono);font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-2);flex:none}
.status p{margin:0;min-width:0}
.tag{display:inline-block;font-family:var(--font-mono);font-size:.7rem;font-weight:500;letter-spacing:.05em;text-transform:uppercase;padding:1px 7px;border-radius:3px;border:1px solid transparent;white-space:nowrap;vertical-align:.08em}
.tag.m{background:var(--accent-soft);color:var(--accent)}
.tag.u{color:var(--ink-2);border-color:var(--line-2)}
.toc{margin-top:1.8rem}
.toc ol{list-style:none;padding:0;margin:0;max-width:none;display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:2px 28px}
.toc li{margin:0}
.toc a{display:flex;gap:.7rem;padding:5px 0;text-decoration:none;color:var(--ink);border-bottom:1px solid var(--line)}
.toc a span{font-family:var(--font-mono);font-size:.82rem;color:var(--ink-2);flex:none;width:2.2rem}
.toc a:hover{color:var(--accent)}
code{font-family:var(--font-mono);font-size:.88em;background:var(--code);padding:1px 5px;border-radius:3px}
pre{background:var(--code);border:1px solid var(--line);border-radius:6px;padding:16px 18px;margin:1.1rem 0;overflow-x:auto;font-family:var(--font-mono);font-size:13.5px;line-height:1.6;tab-size:4}
pre .c{color:var(--ink-2)}
.scroll{overflow-x:auto;margin:1.1rem 0}
table{border-collapse:collapse;width:100%;font-size:.93rem;line-height:1.45}
th,td{text-align:left;vertical-align:top;padding:9px 14px 9px 0;border-bottom:1px solid var(--line)}
th{font-weight:600;border-bottom:1px solid var(--line-2);white-space:nowrap}
td:first-child{font-weight:500}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;padding-right:22px}
table.wide{min-width:720px}
figure{margin:1.6rem 0}
figure .scroll{margin:0}
figure svg{display:block;width:100%;height:auto}
figure .diagram svg{min-width:720px}
figure svg text{font-family:var(--font-body)}
figure svg text.mono{font-family:var(--font-mono)}
figure svg text.num{font-variant-numeric:tabular-nums}
figcaption{font-size:.9rem;color:var(--ink-2);max-width:68ch;margin-top:.8rem}
.rules{padding-left:1.4rem}
.rules li{margin:.75rem 0;padding-left:.2rem}
.rules em{font-style:normal;font-family:var(--font-mono);font-size:.78rem;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-2)}
.formula{font-family:var(--font-mono);font-size:.92rem;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:12px 16px;margin:1.1rem 0;max-width:68ch;overflow-x:auto;white-space:nowrap}
.viz{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:18px 20px 14px;max-width:820px}
.viz h3{margin:0;font-size:1rem}
.viz .sub{margin:.25rem 0 .7rem;color:var(--ink-2);font-size:.9rem;max-width:none}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:.86rem;color:var(--ink-2);margin:0 0 .4rem;padding:0;list-style:none;max-width:none}
.legend li{display:flex;align-items:center;gap:8px;margin:0}
.key{display:inline-block;width:18px;height:2px;border-radius:1px;flex:none}
.plot{position:relative;max-width:760px}
.plot .scroll{margin:0}
.plot svg{min-width:560px}
table.narrow{width:auto;min-width:min(100%,460px)}
.tip{position:absolute;top:0;left:0;pointer-events:none;background:var(--surface);border:1px solid var(--line-2);border-radius:6px;padding:8px 11px;font-size:.8rem;line-height:1.5;box-shadow:0 6px 18px rgba(0,0,0,.14);white-space:nowrap;z-index:2}
.tip .t{color:var(--ink-2);margin-bottom:2px}
.tip .r{display:flex;align-items:center;gap:8px}
.tip .r b{font-weight:600;min-width:3.4em;text-align:right;font-variant-numeric:tabular-nums}
.tip .r span.n{color:var(--ink-2)}
.band{cursor:crosshair}
footer{margin-top:3.4rem;padding-top:1.1rem;border-top:1px solid var(--line-2);color:var(--ink-2);font-size:.88rem}
footer p{max-width:none}
@media (max-width:560px){ body{font-size:15.5px} .status{flex-direction:column;gap:4px} pre{font-size:12.5px} }
@media (prefers-reduced-motion:no-preference){ .hair{transition:opacity .12s} }
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Specification · space-time cellular automaton for drone coordination</p>
  <h1>Unified Wavefront Design</h1>
  <p class="lede">The design has one lattice, one local rule and one operation, which is to clamp what is known or already decided and grow the rest. The same learner covers larger grids, moving targets, hidden targets with an observer, directed sensors and several drones to recharge, because each of those enters only as a declaration made by the environment.</p>
  <div class="status"><span class="label">Status</span><p>This is a specification. Parts marked <span class="tag m">measured</span> rest on the 8×8 experiments in this repository. Parts marked <span class="tag u">not yet tested</span> are proposals. The design has not been built or run as a whole.</p></div>
  <nav class="toc" aria-label="Sections"><ol>
    <li><a href="#idea"><span>§1</span>The idea</a></li>
    <li><a href="#contract"><span>§2</span>The bitter-lesson contract</a></li>
    <li><a href="#interface"><span>§3</span>What an environment declares</a></li>
    <li><a href="#lattice"><span>§4</span>The lattice</a></li>
    <li><a href="#acting"><span>§5</span>Acting</a></li>
    <li><a href="#search"><span>§6</span>Search</a></li>
    <li><a href="#credit"><span>§7</span>Credit</a></li>
    <li><a href="#learning"><span>§8</span>Learning</a></li>
    <li><a href="#checks"><span>§9</span>Evaluation and checks</a></li>
    <li><a href="#dials"><span>§10</span>Compute dials</a></li>
    <li><a href="#priors"><span>§11</span>What was removed, what remains</a></li>
    <li><a href="#evidence"><span>§12</span>Evidence so far</a></li>
    <li><a href="#build"><span>§13</span>Build order</a></li>
  </ol></nav>
</header>

<h2 id="idea"><span class="sec">§1</span>The idea</h2>
<p>The lattice has time as one of its three axes. At any moment it holds a window of time slices around the present. Slices behind the present hold <strong>facts</strong>, meaning what the sensors reported and where the drones actually went. Slices ahead hold the <strong>dream</strong>. For each kind of drone the system controls, the dream contains a <strong>tube</strong> that marks where those drones will be. For everything the system only observes, it contains a <strong>forecast</strong>. The present is the <strong>front</strong> between facts and dream.</p>
<figure>
  <div class="scroll diagram">@@FIG1@@</div>
  <figcaption>The window at step t on a 4×4 excerpt of the grid. Behind the front, two drones have approached each other. Ahead of it, the rule has grown both tubes onto the same cells: they meet at t+1 and reach the target at t+3. Faint marks are places the tubes still give some weight to.</figcaption>
</figure>
<p>Every step repeats four actions.</p>
<ol>
  <li>Clamp the newest facts into the lattice.</li>
  <li>Run the rule for a few iterations.</li>
  <li>Move each controlled drone to the reachable position where its tube is strongest, and clamp that choice as a commitment.</li>
  <li>Advance the simulator and slide the window by one slice.</li>
</ol>
<p>The lattice keeps its state from step to step. A plan made at one step is still there at the next unless new facts contradict it. <span class="tag u">not yet tested</span></p>
<p>Two rules carry over from the earlier work. Drones are steered by tubes, which are positions in space-time, and never by a field of arrows. Forecasts describe things the system does not control, and no action is ever read from a forecast.</p>

<h2 id="contract"><span class="sec">§2</span>The bitter-lesson contract</h2>
<p>Sutton's bitter lesson says that methods which scale with computation, namely search and learning, end up ahead of methods that build in what their designers know. For this design that becomes five rules. Each one has a check.</p>
<ol class="rules">
  <li><strong>No domain concept in the learner.</strong> Battery, distance, meeting, escort, cone and heading are not named anywhere in it. Environments differ only through the declaration in §3. <em>Check</em> a text search of the learner's source.</li>
  <li><strong>Every ability comes from search or learning.</strong> Each has a compute dial, listed in §10.</li>
  <li><strong>More compute never hurts.</strong> More iterations, more clones or a longer window must not lower success. <em>Check</em> the three sweeps in §9. The rule trained so far fails this check: twice the iterations cost it 5 to 18 points <span class="tag m">measured</span>. The persistent lattice and the randomised iteration counts exist to repair that.</li>
  <li><strong>No exploration prior, curriculum, reward shaping or solution heuristic.</strong> Tasks are drawn uniformly. Selection uses return and nothing else.</li>
  <li><strong>Every prior that remains is written down</strong> together with the test that could remove it (§11).</li>
</ol>

<h2 id="interface"><span class="sec">§3</span>What an environment declares</h2>
<p>The learner knows an environment only through this declaration and four simulator calls.</p>
<div class="scroll"><table class="wide">
  <thead><tr><th>Declared</th><th>Meaning</th><th>Today's 8×8 task</th></tr></thead>
  <tbody>
    <tr><td>Grid size</td><td>Height and width, any size</td><td>8 × 8</td></tr>
    <tr><td>Entity types</td><td>Each kind of thing that can occupy a cell</td><td>Seeker, recharger, target</td></tr>
    <tr><td>Own-state alphabet</td><td>How many internal states a type has besides its cell</td><td>1 for all three</td></tr>
    <tr><td>Controlled or observed</td><td>Whether the system chooses the type's actions</td><td>Seeker and recharger controlled, target observed</td></tr>
    <tr><td>Actions</td><td>Per controlled type, plus one default action</td><td>Stay, N, S, W, E; default stay</td></tr>
    <tr><td>Observation</td><td>Per step: which cells were sensed and what was in them</td><td>Every cell sensed</td></tr>
    <tr><td>Reward and discount</td><td>Scalar reward events; the return is their discounted sum</td><td>1 when the seeker stands on the target. γ = 0.95 is proposed; the measured runs preferred the earliest reward, which ranks plans the same way</td></tr>
    <tr><td>Identity codes, optional</td><td>A code vector that ties entities together, such as a drone and its own target</td><td>Not used</td></tr>
  </tbody>
</table></div>
<ul>
  <li><code>reset(task, world)</code> starts an episode. The task fixes what is visible at the start. The world seed fixes everything hidden or random.</li>
  <li><code>step(actions)</code> returns the observation, the reward and whether the episode is over.</li>
  <li><code>peek(drone, action)</code> returns that drone's own next state for one action. It looks one step ahead and gives no reward.</li>
  <li><code>replay(actions)</code> runs a recorded list of actions from the start.</li>
</ul>
<p>A controlled drone always knows its own state. The battery is never part of any observation.</p>
<h3>The six directions as declarations</h3>
<div class="scroll"><table class="wide">
  <thead><tr><th>Direction</th><th>What changes in the declaration</th><th>Change to the learner</th></tr></thead>
  <tbody>
    <tr><td>Larger grid</td><td>The grid size</td><td>None</td></tr>
    <tr><td>Moving target</td><td>Nothing. The target is an observed entity and is reported at every step it is sensed</td><td>None</td></tr>
    <tr><td>Observer with a wider sensor</td><td>One more controlled type. The observation now marks which cells were sensed</td><td>None</td></tr>
    <tr><td>Directed sensor</td><td>The observer's own-state alphabet becomes four headings, and turn actions are added</td><td>None</td></tr>
    <tr><td>Several drones to recharge</td><td>More entities of the seeker type, with identity codes when each has its own target</td><td>None</td></tr>
    <tr><td>Fine-tuning in place of retraining</td><td>New input and output channels are appended with zero weights, so behaviour is unchanged at the start</td><td>None</td></tr>
  </tbody>
</table></div>

<h2 id="lattice"><span class="sec">§4</span>The lattice</h2>
<p>The window holds P past slices, the present slice and F future slices, each the size of the grid. Two tensors live on it. The state S has C channels and persists from step to step. The clamped input X holds the facts and is rewritten only where something becomes known or decided.</p>
<div class="scroll"><table class="wide">
  <thead><tr><th>Channel group</th><th>Count</th><th>Content</th><th>Non-zero in</th></tr></thead>
  <tbody>
    <tr><td><code>seen</code> (input)</td><td>1</td><td>A sensor covered this cell at this time</td><td>Past and present</td></tr>
    <tr><td><code>fact</code> per type (input)</td><td>Own-state alphabet size</td><td>An entity of this type and own-state is known to be here, because it was observed or because the system committed to it</td><td>Past, present, and the slice just ahead while drones commit</td></tr>
    <tr><td><code>identity</code> (input)</td><td>d</td><td>The entity's code vector</td><td>Wherever a fact is</td></tr>
    <tr><td><code>state</code></td><td>C</td><td>Free for the rule to use</td><td>Everywhere</td></tr>
    <tr><td><code>tube</code> per controlled type (output)</td><td>Own-state alphabet size</td><td>Where drones of this type will be</td><td>Read in future slices</td></tr>
    <tr><td><code>forecast</code> (output)</td><td>1 + alphabet sizes of observed types</td><td>Which cells will be sensed, and what will be found there</td><td>Read in future slices</td></tr>
  </tbody>
</table></div>
<p>The rule has the form already trained, with wider layers.</p>
@@PRE_RULE@@
<p>All weights are shared across voxels, slices and iterations. There are no coordinate channels, no pooling, no attention and no iteration counter.</p>
<div class="scroll"><table class="narrow">
  <thead><tr><th>Example sizes</th><th class="num">Today's task</th><th class="num">All six directions together</th></tr></thead>
  <tbody>
    <tr><td>Input channels</td><td class="num">4</td><td class="num">12</td></tr>
    <tr><td>Tube channels</td><td class="num">2</td><td class="num">6</td></tr>
    <tr><td>Forecast channels</td><td class="num">2</td><td class="num">2</td></tr>
  </tbody>
</table></div>
<p>The second column assumes a seeker, a recharger, an observer with four headings, a target, and identity codes of length 4.</p>

<h2 id="acting"><span class="sec">§5</span>Acting</h2>
@@PRE_ACT@@
<ul>
  <li>The order is shuffled at every step. A drone that chooses later sees the commitments of the drones before it. This is the chain rule for a joint choice, and it assumes nothing about who should lead. <span class="tag u">not yet tested</span></li>
  <li>Sampling noise enters in the argmax line only. With τ = 0 the protocol is deterministic.</li>
  <li>One step costs n · k_c + k_s iterations of the rule for n controlled drones.</li>
  <li>When the window slides, facts older than P steps leave the input. Anything still needed must by then be held in the state channels.</li>
  <li>A diagnostic readout runs K0 iterations once and then follows the tubes with no further clamping. It measures how much of the plan the dream holds by itself.</li>
</ul>

<h2 id="search"><span class="sec">§6</span>Search</h2>
<p>A <strong>strategy</strong> is one noise stream per controlled drone plus a temperature τ. The greedy strategy has τ = 0. A <strong>world</strong> is a seed for everything that is hidden or random. A <strong>clone</strong> is one strategy acting in one world with the protocol of §5.</p>
@@PRE_SEARCH@@
<figure>
  <div class="scroll diagram">@@FIG2@@</div>
  <figcaption>Selection and credit for one task with a hidden target. A filled mark means the reward was reached in that world. The strategy with the highest average is kept and stored with all four of its rollouts. Each drone's tube is later imitated with its credit as the weight (§7).</figcaption>
</figure>
<ul>
  <li>Every strategy meets the same worlds, so values are comparable.</li>
  <li>When nothing is hidden or random, W = 1 and the rule reduces to keeping the best rewarded rollout, as in the measured system.</li>
  <li>When something is hidden or random, the best single rollout is the luckiest one. Averaging over worlds before selecting removes the luck. The kept strategy is then imitated in all W worlds, including those where it earned nothing. A good way of looking for a hidden target is still right in the world where the target happened to be far away. <span class="tag u">not yet tested</span></li>
  <li>Recombination costs simulator time only, and from the same samples it finds many more rewarded plans. <span class="tag m">measured</span></li>
  <li>The population is flat. There is no tree, no value estimate and no backup.</li>
</ul>

<h2 id="credit"><span class="sec">§7</span>Credit</h2>
<p>A sparse reward says that a joint plan worked. It does not say whose behaviour made it work. The kept strategy is therefore run again with one drone lesioned: that drone acts uniformly at random for the whole episode, while all other drones act as before and react to what they now see.</p>
<div class="formula">credit[i] = clip( ( V[kept] − V[kept, drone i random] ) / V[kept], 0, 1 )&nbsp;&nbsp;&nbsp;with K lesion runs per world</div>
<ul>
  <li>A drone that did not matter gets credit 0 and is not imitated. In an open-loop form this is what stopped the recharger from learning to stand still. <span class="tag m">measured</span></li>
  <li>Because the other drones react to the lesion, a drone whose only contribution is information earns credit when the others fail without it. Replaying recorded actions cannot show that, since the recorded actions already contain what the informer found. <span class="tag u">not yet tested</span></li>
  <li>For a recombined strategy the other drones replay their action lists.</li>
</ul>

<h2 id="learning"><span class="sec">§8</span>Learning</h2>
<p>There are two buffers. One holds the kept rollouts with their credits. The other holds a sample of all clones, whatever their outcome.</p>
@@PRE_LEARN@@
<div class="formula">tube loss = − Σ over drones i and future slices j of&nbsp; credit[i] · log softmax( tube[type(i)][j] )[ u_i(j) ]</div>
<ul>
  <li>u_i(j) is the state drone i really had at that time. The softmax runs over all cells and own-states of the slice. Slices after the end of the episode carry no loss.</li>
  <li>The forecast loss is the cross-entropy between the forecast in a future slice and what was observed there later. For <code>fact</code> channels it counts only cells that were sensed.</li>
  <li>Tubes learn from kept rollouts only. Forecasts learn from every rollout, rewarded or not.</li>
  <li>Iteration counts are redrawn for every replayed episode: K0 between half and twice the window length, k_c from 1 to 2, k_s from 1 to 4. This is what allows the counts to be raised at test time.</li>
  <li>Tasks, grid sizes and numbers of entities are drawn from whatever the environment offers, in no particular order.</li>
</ul>

<h2 id="checks"><span class="sec">§9</span>Evaluation and checks</h2>
<ul>
  <li><strong>Headline number.</strong> The greedy strategy, averaged over W worlds, on tasks that were never posed in training, with task seeds fixed in advance.</li>
  <li><strong>Dream-only readout</strong> where the whole world is visible, as described in §5.</li>
  <li><strong>Commit stride.</strong> Commit every 1, 2 or 5 steps, or never. The curve shows how much replanning the rule needs.</li>
  <li><strong>Compute sweeps for rule 3.</strong> Success against iterations at ×1, ×2 and ×4, against the dream length F, and against the number of strategies M.</li>
  <li><strong>Ablations.</strong> Credit off, with all weights 1. Recombination off. Forecast loss off. All drones commit at once.</li>
  <li><strong>Domain-term search for rule 1</strong> over the learner's source.</li>
</ul>

<h2 id="dials"><span class="sec">§10</span>Compute dials</h2>
<div class="scroll"><table class="wide">
  <thead><tr><th>Dial</th><th>Starting value</th><th>More of it buys</th><th>Paid in</th></tr></thead>
  <tbody>
    <tr><td>Strategies per task, M</td><td>256</td><td>More rewarded plans to learn from</td><td>Lattice runs</td></tr>
    <tr><td>Recombined plans</td><td>All pairs, up to 65,536</td><td>More rewarded plans from the same samples</td><td>Simulator replays</td></tr>
    <tr><td>Worlds per strategy, W</td><td>1 when nothing is hidden or random, otherwise 8</td><td>Selection by skill instead of luck</td><td>Lattice runs</td></tr>
    <tr><td>Lesion runs per drone, K</td><td>8</td><td>Sharper credit</td><td>Lattice runs, for kept strategies only</td></tr>
    <tr><td>State channels C, hidden width</td><td>32, 128</td><td>Capacity of the rule</td><td>Every iteration</td></tr>
    <tr><td>Iterations K0, k_c, k_s</td><td>Window length, 1, 2</td><td>More computation per decision</td><td>Every step</td></tr>
    <tr><td>Window P, F</td><td>6, 20</td><td>Memory of the past, depth of the dream</td><td>Every iteration</td></tr>
    <tr><td>Back-propagation span n_bp</td><td>2 steps</td><td>Credit through more iterations</td><td>Training memory</td></tr>
  </tbody>
</table></div>
<p>The starting values are untuned. Other settings: batches of 64 episodes, λ = 0.1, a quarter of each batch from rollouts where search beat the greedy strategy, temperatures 0.5, 1 and 2 in rotation, identity codes of length 4.</p>

<h2 id="priors"><span class="sec">§11</span>What was removed, what remains</h2>
<h3>Hand-made parts of earlier designs and their replacements</h3>
<div class="scroll"><table class="wide">
  <thead><tr><th>Removed</th><th>Replaced by</th><th>Reason</th></tr></thead>
  <tbody>
    <tr><td>Exploration noise shared by both drones</td><td>Independent noise, more clones, recombination</td><td>It is a coordination prior. Compute reaches the same yield without it (§12)</td></tr>
    <tr><td>The more confident drone leads, the other follows</td><td>Shuffled commit order</td><td>A hand-made rule. Drones swapping cells made up 0.4% of failures</td></tr>
    <tr><td>Regrowing the lattice from nothing at every step</td><td>A lattice that keeps its state</td><td>Regrowing flips between plans: it gains 22 points on hard rendezvous and loses 5 on easy ones</td></tr>
    <tr><td>Iteration counts fixed for the 8×8 grid</td><td>Counts redrawn in training, with the check that more never hurts</td><td>The current rule loses 5 to 18 points when run twice as long</td></tr>
    <tr><td>A lattice spanning the whole episode</td><td>A sliding window</td><td>Cost must not depend on episode length</td></tr>
    <tr><td>Keeping the best rollout per task</td><td>Keeping the best strategy by its value over worlds</td><td>The best single rollout is the luckiest once anything is hidden or random</td></tr>
    <tr><td>Credit from replayed actions, a co-location window, and a copy of the battery rule inside the learner</td><td>Closed-loop lesion runs in the real simulator</td><td>Flagged by the audit; replayed actions give an informer no credit</td></tr>
    <tr><td>A death-prediction output with hand-designed lesion training</td><td>Forecasts of what will be observed, learned from all rollouts</td><td>One general mechanism in place of a special one</td></tr>
    <tr><td>Hindsight relabelling and label pruning</td><td>Nothing</td><td>Together they lowered success from 55% to 18% at the same point of training</td></tr>
    <tr><td>Survival splitting, healing buffers, nurseries</td><td>Nothing</td><td>Special-purpose machinery. Splitting cannot help before the first deadline</td></tr>
    <tr><td>Random excursions held for several steps</td><td>Plain sampling noise</td><td>One noise source is enough until shown otherwise</td></tr>
    <tr><td>An earliest-reward tie rule</td><td>The discounted return declared by the environment</td><td>The preference for sooner reward belongs to the objective</td></tr>
  </tbody>
</table></div>
<h3>Priors that remain</h3>
<div class="scroll"><table class="wide">
  <thead><tr><th>Prior</th><th>Why it is there</th><th>Test that could remove it</th></tr></thead>
  <tbody>
    <tr><td>A local rule shared by all voxels of a space-time lattice, with tubes as output</td><td>This is the object of study</td><td>None</td></tr>
    <tr><td>3×3×3 neighbourhood with zero padding</td><td>The smallest symmetric neighbourhood. Padding makes walls and the ends of the window visible</td><td>A wider neighbourhood at equal compute</td></tr>
    <tr><td>One probability distribution per drone type and slice in the tube loss</td><td>A drone is in one place at a time</td><td>A per-voxel loss</td></tr>
    <tr><td>A one-step peek that maps an action to a position in the lattice</td><td>The readout must know where an action leads</td><td>Action scores read at the drone's own voxel, at the risk of the back-and-forth seen with arrow fields</td></tr>
    <tr><td>The discount γ</td><td>Sooner reward is preferred</td><td>It is the environment's choice</td></tr>
    <tr><td>Uniformly random actions as the lesion</td><td>The simplest behaviour that carries no intent</td><td>Fresh samples from the current rule</td></tr>
    <tr><td>Recombination treats each drone as a unit</td><td>Drones are declared by the environment</td><td>Recombination off</td></tr>
    <tr><td>Three sampling temperatures</td><td>Spreads the noise level across clones</td><td>A single temperature</td></tr>
    <tr><td>The forecast loss and its weight λ</td><td>Lets every rollout teach how the world behaves</td><td>λ = 0</td></tr>
  </tbody>
</table></div>

<h2 id="evidence"><span class="sec">§12</span>Evidence so far</h2>
<p>Rule 4 removes the shared-noise prior. The measurement below shows what that costs and what compute gives back. It uses an untrained rule, so every plan is a random walk and only the simulator is involved.</p>
<figure>
  <div class="viz" id="viz1">
    <h3>Search yield before any training</h3>
    <p class="sub">Share of 400 recharge-needed tasks with at least one rewarded plan</p>
    <ul class="legend" aria-label="Series">
      <li><span class="key" style="background:var(--a)"></span>Independent noise, joint rollouts</li>
      <li><span class="key" style="background:var(--b)"></span>Shared noise, joint rollouts</li>
      <li><span class="key" style="background:var(--c)"></span>Independent noise, all pairs recombined</li>
    </ul>
    <div class="plot"><div class="scroll">@@CHART@@</div><div class="tip" hidden></div></div>
  </div>
  <figcaption>All three curves rise smoothly with the search budget. The shared-noise prior is worth a factor of roughly 4 to 10 in rollouts. Recombining 256 independently sampled plans per drone reaches 58.5%, which joint sampling with shared noise needs about 2,500 rollouts to match. Noise here includes random actions held for a few steps, as in the experiments.</figcaption>
</figure>
<div class="scroll"><table>
  <thead><tr><th class="num">Plans per drone</th><th class="num">Independent, joint</th><th class="num">Shared, joint</th><th class="num">Independent, recombined</th><th class="num">Pairs judged</th></tr></thead>
  <tbody>
    <tr><td class="num">16</td><td class="num">0.8%</td><td class="num">2.5%</td><td class="num">6.0%</td><td class="num">256</td></tr>
    <tr><td class="num">64</td><td class="num">1.8%</td><td class="num">9.0%</td><td class="num">22.5%</td><td class="num">4,096</td></tr>
    <tr><td class="num">256</td><td class="num">7.0%</td><td class="num">29.0%</td><td class="num">58.5%</td><td class="num">65,536</td></tr>
    <tr><td class="num">1,024</td><td class="num">17.3%</td><td class="num">45.3%</td><td class="num">not run</td><td class="num"></td></tr>
    <tr><td class="num">4,096</td><td class="num">37.5%</td><td class="num">65.5%</td><td class="num">not run</td><td class="num"></td></tr>
  </tbody>
</table></div>
<p>The same holds for the trained rule on the tasks it fails most, those where the recharger starts 8 to 10 cells away.</p>
<div class="scroll"><table>
  <thead><tr><th>Search finds a rewarded plan</th><th class="num">Start distance 8</th><th class="num">9</th><th class="num">10</th></tr></thead>
  <tbody>
    <tr><td>64 joint rollouts, as in training</td><td class="num">54%</td><td class="num">32%</td><td class="num">15%</td></tr>
    <tr><td>256 joint rollouts</td><td class="num">66%</td><td class="num">44%</td><td class="num">24%</td></tr>
    <tr><td>64 × 64 recombined pairs</td><td class="num">84%</td><td class="num">62%</td><td class="num">49%</td></tr>
    <tr><td>256 × 256 recombined pairs</td><td class="num">98%</td><td class="num">89%</td><td class="num">81%</td></tr>
  </tbody>
</table></div>
<h3>What earlier measurements say about each part</h3>
<div class="scroll"><table class="wide">
  <thead><tr><th>Part of the design</th><th>Measured on the 8×8 task with two drones</th></tr></thead>
  <tbody>
    <tr><td>Clamping the realised past</td><td>Adds 22 points where the recharger starts 8 or more cells away and loses 5 where it starts within 5, with the lattice regrown from nothing at each step</td></tr>
    <tr><td>Following a changing world</td><td>A randomly moving target is caught as often as a static one, 84%, when its current cell is clamped at every step, and in 47% when it is not</td></tr>
    <tr><td>Acting before the target is known</td><td>A target hidden for the first 6 steps costs nothing. With no target at all the drones still gather and keep the seeker alive in 84% of placements</td></tr>
    <tr><td>A dream that holds a whole plan</td><td>Read once, the dream is within 2 to 3 points of the step-by-step readout</td></tr>
    <tr><td>A longer horizon</td><td>30 slices in place of 20 cost nothing</td></tr>
    <tr><td>A wider grid without retraining</td><td>53% on 12×12 and 25% on 16×16. Targets farther than 14 cells are reached in 1 to 5%</td></tr>
    <tr><td>More iterations</td><td>Twice the iterations lose 5 to 18 points on the native grid</td></tr>
    <tr><td>Credit</td><td>The open-loop form was part of every run that passed. It was never switched off alone</td></tr>
  </tbody>
</table></div>
<h3>Not yet tested</h3>
<ul>
  <li>A lattice that keeps its state inside a sliding window.</li>
  <li>The shuffled commit order.</li>
  <li>Selecting strategies by their value over several worlds.</li>
  <li>Closed-loop lesion credit, including credit for a drone that only gathers information.</li>
  <li>The forecast outputs.</li>
  <li>Training with independent noise and recombination. One older run without shared noise and without recombination reached 65.5% where the run with shared noise reached 75.4%.</li>
</ul>
<h3>Where it could fail</h3>
<ul>
  <li>On large grids, long formation flights are found by independent noise only after the rule has learned to follow a partner on short tasks. If that transfer does not happen, shared noise returns as a dial with a known price.</li>
  <li>Back-propagation through two steps may be too short to teach plans that depend on events far ahead.</li>
  <li>Lesion credit costs lattice runs in proportion to the number of drones.</li>
</ul>

<h2 id="build"><span class="sec">§13</span>Build order</h2>
<ol>
  <li>Port the measured recipe to the declared interface on 8×8, with a window spanning the episode, W = 1, independent noise and recombination. Pass: at least the earlier 75% on tasks never posed in training.</li>
  <li>Add the persistent lattice and redrawn iteration counts. Pass: success does not fall when the iterations are doubled and quadrupled.</li>
  <li>Add the sliding window and train on grids from 8 to 16. Pass: 16×16 far above the 25% reached without retraining, and targets beyond 14 cells solved.</li>
  <li>Hide the target, add an observer, set W = 8. Pass: the observer's credit is above zero and success is higher than with a random observer.</li>
  <li>Give the observer a heading. Add further seekers.</li>
</ol>
<p>Each stage keeps all earlier checks.</p>

<footer>
  <p>Numbers come from the repository: <code>results/probes/search_scaling_cold_start.json</code>, <code>results/v10/diag_search.json</code>, <code>results/v10/diag_open_vs_closed.json</code> and the other files in <code>results/probes/</code>. They are single-checkpoint measurements on 300 to 2,000 tasks each.</p>
</footer>
</div>

<script>
(function(){
  var root=document.getElementById('viz1'); if(!root) return;
  var svg=root.querySelector('svg'), tip=root.querySelector('.tip'), hair=svg.querySelector('.hair'), plot=root.querySelector('.plot');
  var xs=@@XS@@, px=@@PX@@, series=@@SERIES@@;
  function show(i,band){
    hair.setAttribute('x1',px[i]); hair.setAttribute('x2',px[i]); hair.setAttribute('opacity','1');
    tip.textContent='';
    var t=document.createElement('div'); t.className='t'; t.textContent=xs[i].toLocaleString('en-US')+' plans per drone'; tip.appendChild(t);
    series.forEach(function(s){
      if(i>=s.v.length) return;
      var r=document.createElement('div'); r.className='r';
      var k=document.createElement('span'); k.className='key'; k.style.background='var(--'+s.k+')';
      var b=document.createElement('b'); b.textContent=s.v[i].toFixed(1)+'%';
      var n=document.createElement('span'); n.className='n'; n.textContent=s.n;
      r.appendChild(k); r.appendChild(b); r.appendChild(n); tip.appendChild(r);
    });
    tip.hidden=false;
    var br=band.getBoundingClientRect(), pr=plot.getBoundingClientRect();
    var x=br.left+br.width/2-pr.left, left=x+14;
    if(left+tip.offsetWidth>plot.clientWidth) left=x-tip.offsetWidth-14;
    if(left<0) left=0;
    tip.style.left=left+'px'; tip.style.top='22px';
  }
  function hide(){ tip.hidden=true; hair.setAttribute('opacity','0'); }
  Array.prototype.forEach.call(svg.querySelectorAll('.band'),function(b){
    var i=+b.getAttribute('data-i');
    b.addEventListener('pointerenter',function(){show(i,b)});
    b.addEventListener('pointermove',function(){show(i,b)});
    b.addEventListener('focus',function(){show(i,b)});
    b.addEventListener('pointerleave',hide);
    b.addEventListener('blur',hide);
  });
})();
</script>
'''


def main():
    import json
    page = PAGE.replace("@@FIG1@@", fig1()).replace("@@FIG2@@", fig2()).replace("@@CHART@@", chart())
    for tok, block in (("@@PRE_RULE@@", PRE_RULE), ("@@PRE_ACT@@", PRE_ACT), ("@@PRE_SEARCH@@", PRE_SEARCH), ("@@PRE_LEARN@@", PRE_LEARN)):
        page = page.replace(tok, block)
    page = page.replace("@@XS@@", json.dumps(XS)).replace("@@PX@@", json.dumps([round(X(v), 1) for v in XS]))
    page = page.replace("@@SERIES@@", json.dumps([dict(k=k, n=n, v=v) for k, n, v, _ in SERIES]))
    assert "@@" not in page
    if STANDALONE:
        page = HEAD + page + "</body></html>"
    open(OUT, "w").write(page)
    print(OUT, len(page), "bytes")


if __name__ == "__main__":
    main()
