"""Builds version 2 of the design page (published at its own address; version 1 keeps the earlier one). Document tooling only: it runs no project code.

usage: python3 build_design_page_v2.py [out.html] [--standalone]
"""
import re
import sys

import build_design_page as v1   # version 1 builder: style sheet, pseudo-code helper, lattice figure

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
STANDALONE = "--standalone" in sys.argv
OUT = ARGS[0] if ARGS else "wavefront_v2_design.html"
pre = v1.pre

CSS = re.search(r"<style>(.*?)</style>", v1.PAGE, re.S).group(1)
CSS += """
.tag.d{background:var(--ink);color:var(--bg)}
.tag.q{color:var(--ink);border:1px dashed var(--ink-2)}
td .tag{margin-left:0}
.phase td:first-child{white-space:nowrap}
.kv td:first-child{white-space:nowrap}
.terms{margin-top:1.9rem}
.terms h3{margin:0 0 .7rem}
.terms dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:6px 22px;margin:0;font-size:.93rem;line-height:1.45}
.terms dt{font-weight:600}
.terms dd{margin:0;max-width:68ch}
@media (max-width:560px){ .terms dl{grid-template-columns:minmax(0,1fr);gap:0} .terms dd{margin-bottom:.6rem} }
"""

M = '<span class="tag m">measured</span>'
U = '<span class="tag u">not yet tested</span>'
E = '<span class="tag u">estimate</span>'
D = '<span class="tag d">decided</span>'
Q = '<span class="tag q">needs approval</span>'


# ----------------------------------------------------------------------------- figure: who may go where
def fig_map():
    W, H = 920, 392
    x0, y0, c = 30, 34, 26
    cols, rows = 22, 12
    road = set()
    road |= {(x, 10) for x in range(0, 22)}
    road |= {(3, y) for y in range(2, 11)}
    road |= {(x, 2) for x in range(3, 15)}
    road |= {(14, y) for y in range(2, 11)}
    road |= {(18, y) for y in range(6, 11)}
    blocks = [(5, 4, 3, 3), (9, 5, 3, 3), (16, 1, 3, 3), (0, 5, 2, 3), (7, 0, 4, 1), (20, 6, 2, 2)]
    obst = {(x, y) for bx, by, bw, bh in blocks for x in range(bx, bx + bw) for y in range(by, by + bh)}
    assert not (obst & road)

    def ctr(x, y):
        return x0 + (x + .5) * c, y0 + (y + .5) * c

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Excerpt of a map with obstacle blocks and a road network with a loop and a dead end. A capturer flies around an obstacle towards the target. An observer flies straight over an obstacle. A recharger stays on the road. The target stands on a free cell and sees drones within two cells.">']
    o.append('<defs>' + "".join(
        f'<marker id="m1{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" style="fill:var(--{k})"/></marker>'
        for k in "abc") + '</defs>')
    gw, gh = cols * c, rows * c
    o.append(f'<rect x="{x0}" y="{y0}" width="{gw}" height="{gh}" rx="4" style="fill:var(--surface)" stroke="currentColor" stroke-opacity=".45" stroke-width="1.2"/>')
    for (x, y) in sorted(road):
        o.append(f'<rect x="{x0+x*c}" y="{y0+y*c}" width="{c}" height="{c}" style="fill:var(--tile)"/>')
    for k in range(1, cols):
        o.append(f'<line x1="{x0+k*c}" y1="{y0}" x2="{x0+k*c}" y2="{y0+gh}" stroke="currentColor" stroke-opacity=".09"/>')
    for k in range(1, rows):
        o.append(f'<line x1="{x0}" y1="{y0+k*c}" x2="{x0+gw}" y2="{y0+k*c}" stroke="currentColor" stroke-opacity=".09"/>')
    for bx, by, bw, bh in blocks:
        o.append(f'<rect x="{x0+bx*c+2}" y="{y0+by*c+2}" width="{bw*c-4}" height="{bh*c-4}" rx="3" fill="currentColor" opacity=".62"/>')

    def path(cells, col, dash=""):
        pts = " ".join(f"{ctr(*p)[0]:.1f},{ctr(*p)[1]:.1f}" for p in cells)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<polyline points="{pts}" fill="none" style="stroke:var(--{col})" stroke-width="2.4" stroke-linejoin="round"{extra} marker-end="url(#m1{col})"/>')

    # target and what it sees (Manhattan radius 2)
    tx, ty = ctr(12, 4)
    r = 2.5 * c
    o.append(f'<path d="M{tx:.1f} {ty-r:.1f}L{tx+r:.1f} {ty:.1f}L{tx:.1f} {ty+r:.1f}L{tx-r:.1f} {ty:.1f}Z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="4 3" opacity=".7"/>')
    # capturer A flies around the obstacle
    path([(4, 8), (4, 3), (9.45, 3)], "a")
    ax, ay = ctr(4, 8)
    o.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="8" style="fill:var(--a);stroke:var(--surface)" stroke-width="2"/>')
    # capturer B closes from the other side
    path([(13, 8), (13, 6.5)], "a")
    bx_, by_ = ctr(13, 8)
    o.append(f'<circle cx="{bx_:.1f}" cy="{by_:.1f}" r="8" style="fill:var(--a);stroke:var(--surface)" stroke-width="2"/>')
    # observer flies straight over an obstacle
    path([(10, 9), (10, 3.6)], "c", "5 4")
    ox, oy = ctr(10, 9)
    o.append(f'<path d="M{ox:.1f} {oy-9:.1f}L{ox+9:.1f} {oy+7:.1f}L{ox-9:.1f} {oy+7:.1f}Z" style="fill:var(--c);stroke:var(--surface)" stroke-width="2"/>')
    # recharger stays on the road
    path([(14, 9), (14, 6.5)], "b")
    rx, ry = ctr(14, 9)
    o.append(f'<rect x="{rx-7.5:.1f}" y="{ry-7.5:.1f}" width="15" height="15" rx="2" style="fill:var(--b);stroke:var(--surface)" stroke-width="2"/>')
    # target
    o.append(f'<path d="M{tx:.1f} {ty-8:.1f}L{tx+8:.1f} {ty:.1f}L{tx:.1f} {ty+8:.1f}L{tx-8:.1f} {ty:.1f}Z" style="fill:var(--surface)" stroke="currentColor" stroke-width="2"/>')

    # legend
    lx, ly, lh = x0 + gw + 34, y0 + 12, 31
    items = [
        ("cap", "Capturer", "every cell but obstacles; battery; captures"),
        ("obs", "Observer", "every cell, obstacles too; battery; sees far"),
        ("rec", "Recharger", "roads only; no battery limit; refills"),
        ("tgt", "Target", "every cell but obstacles; flees drones it sees"),
        ("obst", "Obstacle", ""),
        ("road", "Road", ""),
        ("sight", "What the target sees", "two cells, any drone kind"),
    ]
    for k, (kind, name, note) in enumerate(items):
        cy = ly + k * lh + (12 if k >= 4 else 0)
        cx = lx + 9
        if kind == "cap":
            o.append(f'<circle cx="{cx}" cy="{cy}" r="7.5" style="fill:var(--a)"/>')
        elif kind == "obs":
            o.append(f'<path d="M{cx} {cy-8}L{cx+8.5} {cy+6.5}L{cx-8.5} {cy+6.5}Z" style="fill:var(--c)"/>')
        elif kind == "rec":
            o.append(f'<rect x="{cx-7}" y="{cy-7}" width="14" height="14" rx="2" style="fill:var(--b)"/>')
        elif kind == "tgt":
            o.append(f'<path d="M{cx} {cy-8}L{cx+8} {cy}L{cx} {cy+8}L{cx-8} {cy}Z" style="fill:var(--surface)" stroke="currentColor" stroke-width="1.8"/>')
        elif kind == "obst":
            o.append(f'<rect x="{cx-8}" y="{cy-8}" width="16" height="16" rx="3" fill="currentColor" opacity=".62"/>')
        elif kind == "road":
            o.append(f'<rect x="{cx-8}" y="{cy-8}" width="16" height="16" style="fill:var(--tile)" stroke="currentColor" stroke-opacity=".3"/>')
        else:
            o.append(f'<path d="M{cx} {cy-8}L{cx+8} {cy}L{cx} {cy+8}L{cx-8} {cy}Z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="3 2.5" opacity=".8"/>')
        o.append(f'<text x="{lx+28}" y="{cy-1 if note else cy+4}" font-size="12.5" font-weight="600" fill="currentColor">{name}</text>')
        if note:
            o.append(f'<text x="{lx+28}" y="{cy+14}" font-size="11.5" fill="currentColor" opacity=".78">{note}</text>')
    o.append("</svg>")
    return "\n".join(o)


# ----------------------------------------------------------------------------- figure: one leg, selection and credit
def fig_leg(strategies, kept, lesions, base, title_left, title_right):
    W, H = 920, 292
    lx, ly, cp, rp = 172, 86, 40, 34
    rx0 = 672
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Left: five strategies are run from the same snapshot in the same four worlds; strategy 7 has the highest average, 0.75, beats the greedy rule at 0.50 and is kept. Right: the kept leg is replayed with recorded actions restored one drone at a time, observer, capturer A, recharger, capturer B, while the others act at random; the value rises from 0 to 0.25, 0.50 and 0.75, so the shares are 0, 0.25, 0.25 and 0.25.">']
    o.append('<defs><marker id="f2arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>')

    def mark(cx, cy, on):
        if on:
            return f'<circle cx="{cx}" cy="{cy}" r="8" fill="currentColor"/>'
        return f'<circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="currentColor" stroke-opacity=".4" stroke-width="1.4"/>'

    def header(x_first, x_val, extra=None):
        for j in range(4):
            o.append(f'<text x="{x_first+j*cp}" y="{ly-24}" font-size="11.5" text-anchor="middle" fill="currentColor" class="mono" opacity=".8">w{j+1}</text>')
        o.append(f'<text x="{x_val}" y="{ly-24}" font-size="11.5" text-anchor="middle" fill="currentColor" opacity=".8">value</text>')
        if extra:
            o.append(f'<text x="{extra}" y="{ly-24}" font-size="11.5" text-anchor="middle" fill="currentColor" opacity=".8">credit</text>')

    o.append(f'<text x="64" y="22" font-size="13" font-weight="600" fill="currentColor">{title_left}</text>')
    o.append(f'<text x="540" y="22" font-size="13" font-weight="600" fill="currentColor">{title_right}</text>')
    header(lx, lx + 4 * cp + 14)
    header(rx0, rx0 + 4 * cp + 10, rx0 + 4 * cp + 66)
    ky = ly + kept * rp
    o.append(f'<rect x="{lx-22}" y="{ky-15}" width="{4*cp+58}" height="30" rx="6" style="fill:var(--accent-soft);stroke:var(--accent)" stroke-width="1.8"/>')
    for i, (name, vals) in enumerate(strategies):
        cy = ly + i * rp
        o.append(f'<text x="{lx-32}" y="{cy+4}" font-size="12.5" text-anchor="end" fill="currentColor">{name}</text>')
        for j, v in enumerate(vals):
            o.append(mark(lx + j * cp, cy, v))
        o.append(f'<text x="{lx+4*cp+14}" y="{cy+4}" font-size="12.5" text-anchor="middle" fill="currentColor" class="mono">{sum(vals)/4:.2f}</text>')
    o.append(f'<text x="{lx+4*cp+48}" y="{ky+4}" font-size="12.5" font-weight="600" style="fill:var(--accent)">kept</text>')
    ax0, ax1 = lx + 4 * cp + 86, 520
    o.append(f'<line x1="{ax0}" y1="{ky}" x2="{ax1}" y2="{ky}" stroke="currentColor" stroke-width="1.4" marker-end="url(#f2arr)"/>')
    o.append(f'<text x="{(ax0+ax1)/2}" y="{ky-9}" font-size="12" text-anchor="middle" fill="currentColor">run again</text>')
    prev = 0.0
    for i, (name, vals) in enumerate(lesions):
        cy = ly + i * rp
        v = sum(vals) / 4
        o.append(f'<text x="{rx0-24}" y="{cy+4}" font-size="12.5" text-anchor="end" fill="currentColor">{name}</text>')
        for j, x in enumerate(vals):
            o.append(mark(rx0 + j * cp, cy, x))
        o.append(f'<text x="{rx0+4*cp+10}" y="{cy+4}" font-size="12.5" text-anchor="middle" fill="currentColor" class="mono">{v:.2f}</text>')
        o.append(f'<text x="{rx0+4*cp+66}" y="{cy+4}" font-size="12.5" font-weight="600" text-anchor="middle" fill="currentColor" class="mono">{v-prev:.2f}</text>')
        prev = v
    assert abs(prev - base) < 1e-9
    yl = ly + 5 * rp + 4
    o.append(mark(544, yl - 4, 1) + f'<text x="558" y="{yl}" font-size="12" fill="currentColor">target captured in this world</text>')
    o.append(mark(752, yl - 4, 0) + f'<text x="766" y="{yl}" font-size="12" fill="currentColor">not captured</text>')
    o.append("</svg>")
    return "\n".join(o)


def lattice_figure():
    return v1.fig1().replace(">seeker<", ">capturer<").replace("a seeker and a recharger", "a capturer and a recharger")


def build(sections, toc):
    nav = "\n".join(f'    <li><a href="#{sid}"><span>§{n}</span>{title}</a></li>' for n, (sid, title) in enumerate(toc, 1))
    body = "\n\n".join(sections)
    return f'''<title>Wavefront Design v2</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>

<div class="wrap">
@@HEADER@@
  <nav class="toc" aria-label="Sections"><ol>
{nav}
  </ol></nav>
</header>

{body}
</div>
'''


def main():
    import design_v2_part1 as cv
    import design_v2_part2 as cb
    me = sys.modules[__name__]
    page = build(cv.sections(me), cv.TOC).replace("@@HEADER@@", cv.HEADER)
    for k, v in cb.blocks(me).items():
        page = page.replace("@@" + k + "@@", v)
    page = page.replace("@@FIGLEG@@", fig_leg(
        strategies=[("greedy", [1, 0, 0, 1]), ("strategy 7", [1, 1, 0, 1]), ("strategy 12", [0, 1, 0, 0]),
                    ("strategy 19", [1, 0, 0, 0]), ("strategy 23", [0, 0, 1, 0])],
        kept=1,
        lesions=[("observer", [0, 0, 0, 0]), ("+ capturer A", [0, 1, 0, 0]), ("+ recharger", [1, 1, 0, 0]), ("+ capturer B", [1, 1, 0, 1])],
        base=0.75,
        title_left="Confirmation: every strategy meets the same fresh worlds",
        title_right="Credit: recorded actions restored one drone at a time"))
    assert "@@" not in page, re.findall(r"@@\w+@@", page)
    if STANDALONE:
        page = v1.HEAD + page + "</body></html>"
    open(OUT, "w", encoding="utf8").write(page)
    print(OUT, len(page), "bytes")


if __name__ == "__main__":
    main()
