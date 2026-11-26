"""Figures: what one voxel computes, every layer as 2D maps, growth over iterations, and training.

  python3 viz_layers.py [ckpt]      -> results/figs/*.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import torch.nn.functional as F

from env import Cfg, DroneEnv, manhattan, sample_tasks
from nca import SpaceTimeNCA, make_cond, voxel_norm

OUT = "results/figs"
os.makedirs(OUT, exist_ok=True)
CK = sys.argv[1] if len(sys.argv) > 1 else "results/v10/best_prereg.pt"


def load(path):
    ck = torch.load(path)
    a = ck["args"]
    m = SpaceTimeNCA(C=a["C"], hid=a["hid"], update=a["update"])
    m.load_state_dict(ck["model"])
    m.eval()
    return m, Cfg(T=a["T"], B=a["B"]), a, ck.get("round", 0)


model, cfg, args, rnd = load(CK)
K = args["K"]
T1 = cfg.T + 1


@torch.no_grad()
def greedy(logits, s, r, g):
    env = DroneEnv(s, r, g, cfg)
    ps, pr = [env.ps[0].tolist()], [env.pr[0].tolist()]
    for t in range(cfg.T):
        acts = []
        for d, pos in ((0, env.ps), (1, env.pr)):
            c = env.cand(pos)
            acts.append(logits[torch.arange(1)[:, None], d, t + 1, c[..., 0], c[..., 1]].argmax(1))
        env.step(*acts)
        ps.append(env.ps[0].tolist()); pr.append(env.pr[0].tolist())
    return ps, pr, bool(env.success[0]), int(env.tau[0]), int(env.n_recharge[0])


# ---------------------------------------------------------------- pick a typical solved recharge task
test = sample_tasks(200, cfg, torch.Generator().manual_seed(12345), "need")
pick = None
with torch.no_grad():
    for i in range(200):
        s, r, g = (x[i:i + 1] for x in test)
        if not (7 <= int(manhattan(s, g)) <= 9 and 4 <= int(manhattan(s, r)) <= 6):
            continue
        lg, _ = model(make_cond(s, r, g, cfg), K)
        ps, pr, ok, tau, nre = greedy(lg, s, r, g)
        if ok:
            pick = (s, r, g, ps, pr, tau, nre)
            break
s, r, g, ps, pr, tau, nre = pick
print("task", s.tolist(), r.tolist(), g.tolist(), "tau", tau, "recharges", nre)
cond = make_cond(s, r, g, cfg)
TS = list(range(0, min(tau, cfg.T) + 2))       # time slices to show (up to just after arrival)
TS = [t for t in TS if t < T1]

# ---------------------------------------------------------------- run K-1 iterations, then the last one by hand
with torch.no_grad():
    st = model.init_state(cond)
    states = [st]
    for k in range(K):
        st = model.step(st, cond)
        states.append(st)
    s_prev = states[K - 1]
    xn = voxel_norm(s_prev)
    x = torch.cat([xn, cond], 1)                    # 19 channels: 16 normalised state + 3 seeds
    pre1 = model.perc(x)
    h1 = F.relu(pre1)                               # 64
    h2 = F.relu(model.mid(h1))                      # 64
    delta = model.out(h2)                           # 16
    s_new = s_prev + delta                          # 16
    logits = model.head(s_new)                      # 2
    probs = torch.softmax(logits.reshape(1, 2, T1, -1), -1).reshape(1, 2, T1, cfg.H, cfg.W)


def mark(ax, t, which="both"):
    if t < len(ps) and t <= tau:
        if which in ("both", "s"):
            ax.plot(ps[t][1], ps[t][0], "o", ms=4, mfc="none", mec="red", mew=1.2)
        if which in ("both", "r"):
            ax.plot(pr[t][1], pr[t][0], "s", ms=4, mfc="none", mec="dodgerblue", mew=1.2)
    ax.plot(g[0, 1], g[0, 0], "*", ms=5, color="lime", mec="k", mew=0.4)


def layer_grid(tensor, names, title, fname, cmap="RdBu_r", sym=True, marks=True):
    """rows = channels, columns = time slices; every map is the 8x8 grid at that time."""
    A = tensor[0].numpy()                           # [C, T1, H, W]
    nC = A.shape[0]
    fig, axes = plt.subplots(nC, len(TS), figsize=(0.62 * len(TS) + 1.6, 0.62 * nC + 0.9), squeeze=False)
    for c in range(nC):
        v = np.abs(A[c, TS]).max() + 1e-9
        for j, t in enumerate(TS):
            ax = axes[c, j]
            if sym:
                ax.imshow(A[c, t], cmap=cmap, vmin=-v, vmax=v)
            else:
                ax.imshow(A[c, t], cmap=cmap, vmin=0, vmax=v)
            ax.set_xticks([]); ax.set_yticks([])
            if marks:
                mark(ax, t)
            if c == 0:
                ax.set_title(f"t={t}", fontsize=7)
            if j == 0:
                ax.set_ylabel(names[c], fontsize=6.5, rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.98 - 0.4 / (0.62 * nC + 0.9)))
    fig.savefig(f"{OUT}/{fname}", dpi=90)
    plt.close(fig)
    print(f"{OUT}/{fname}")


# ================================================================ FIG 1: what one voxel computes
def fig_voxel_rule():
    fig, ax = plt.subplots(figsize=(15, 7.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 7.6); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="k", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, txt=""):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.15, txt, ha="center", fontsize=7.5, color="0.3")

    ax.text(0.1, 7.35, "One NCA step for ONE voxel v = (t, x, y).  The same weights run at all 21 x 8 x 8 = 1344 voxels, "
            "32 times in a row (synchronously).", fontsize=10.5, weight="bold")

    # the 3x3x3 neighbourhood
    box(0.1, 3.4, 2.4, 2.9, "INPUT\n3x3x3 neighbourhood\n(t-1..t+1, x-1..x+1, y-1..y+1)\n= 27 voxels, each with\n\n"
        "16 state channels s\n(hidden, learned meaning)\n+ 3 seed channels\n(seeker / recharger / target\none-hot, only in slice t=0;\n"
        "in closed-loop also the\nrealised past positions)", "#fff3c4", 7.6)
    box(0.1, 1.2, 2.4, 1.6, "outside the lattice:\nzero padding\n(walls, t=0 and t=20 edges\nare 'visible' as zeros)", "#eeeeee", 7.5)

    box(3.0, 4.2, 1.9, 1.3, "voxel_norm\nper voxel:\n(s - mean_c) / std_c\nover its 16 channels", "#d9ecff", 7.8)
    arrow(2.5, 4.85, 3.0, 4.85)
    box(5.4, 3.9, 2.0, 1.9, "PERCEPTION\nConv3d 3x3x3\n19 x 27 = 513 inputs\n-> 64 features\n+ ReLU\n(32,896 weights)", "#c9f2d0", 7.8)
    arrow(4.9, 4.85, 5.4, 4.85)
    box(7.9, 4.2, 1.7, 1.3, "1x1x1 conv\n64 -> 64\n+ ReLU\n(4,160 w)", "#c9f2d0", 7.8)
    arrow(7.4, 4.85, 7.9, 4.85, "64")
    box(10.1, 4.2, 1.7, 1.3, "1x1x1 conv\n64 -> 16\n= update ds\n(1,040 w)", "#c9f2d0", 7.8)
    arrow(9.6, 4.85, 10.1, 4.85, "64")
    box(12.3, 4.2, 1.5, 1.3, "RESIDUAL\ns <- s + ds", "#ffd6d6", 8.5)
    arrow(11.8, 4.85, 12.3, 4.85, "16")
    # recurrence
    ax.annotate("", (1.3, 6.32), (13.05, 5.55), arrowprops=dict(arrowstyle="->", lw=1.0, ls="--", color="0.4",
                                                              connectionstyle="arc3,rad=0.12"))
    ax.text(7.3, 6.0, "new state of v becomes input for the neighbours in the next iteration (k = 1 ... 32)",
            fontsize=8, color="0.3", ha="center")

    box(12.0, 1.8, 2.1, 1.5, "HEAD (after 32 its)\n1x1x1 conv 16 -> 2\n(34 w)\nlogit_seeker(v)\nlogit_recharger(v)", "#e7d6ff", 7.8)
    arrow(13.05, 4.2, 13.05, 3.3)

    box(8.7, 0.2, 5.4, 1.3, "OUTPUT / STEERING: drone at cell p at time t looks at the 5 cells it can reach\n"
        "(stay, N, S, W, E) in slice t+1 and moves to the one with the highest logit.\n"
        "Nothing is minimised at test time: no energy, no search, one greedy readout.", "#f4f4f4", 7.8)
    arrow(13.05, 1.8, 13.05, 1.5)

    ax.text(3.0, 2.9, "What it is NOT:\n"
            "- not energy-based: the update is a learned residual map, not -grad E;\n"
            "  there is no scalar that the iteration descends.\n"
            "- no max / min over neighbours (no Bellman backup), no pooling,\n"
            "  no coordinates, no battery input, no iteration counter.\n\n"
            "Total: 38,130 parameters, shared by every voxel and every iteration.\n"
            "Information moves at most 1 voxel per iteration in t, x and y.",
            fontsize=8.2, va="top", family="monospace")
    fig.savefig(f"{OUT}/1_voxel_rule.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"{OUT}/1_voxel_rule.png")


# ================================================================ FIG 2: every layer, as 2D maps
def figs_layers():
    task = f"seeker {s[0].tolist()}  recharger {r[0].tolist()}  target {g[0].tolist()}  (circle=seeker, square=recharger, star=target, actual greedy flight)"
    layer_grid(cond, ["seed: seeker", "seed: recharger", "seed: target"],
               f"Layer 0 - observation seed (3 channels).  Only slice t=0 is non-zero.  {task}", "2a_seed.png",
               cmap="Greys", sym=False)
    layer_grid(xn, [f"s{c}" for c in range(16)],
               "Layer 1 - normalised state of all voxels before the last iteration (16 hidden channels, values per voxel z-scored)",
               "2b_state_in.png")
    layer_grid(h1, [f"h1_{c}" for c in range(64)],
               "Layer 2 - perception: Conv3d 3x3x3 over 19 channels, ReLU (64 features)", "2c_perception.png",
               cmap="magma", sym=False, marks=False)
    layer_grid(h2, [f"h2_{c}" for c in range(64)],
               "Layer 3 - per-voxel 1x1x1 conv 64->64, ReLU (64 features)", "2d_mlp.png", cmap="magma", sym=False, marks=False)
    layer_grid(delta, [f"ds{c}" for c in range(16)],
               "Layer 4 - update ds = conv 64->16 (small at iteration 32: the volume has nearly settled)", "2e_update.png")
    layer_grid(s_new, [f"s{c}" for c in range(16)], "Layer 5 - new state s + ds (raw, 16 channels)", "2f_state_out.png")
    layer_grid(logits, ["logit seeker", "logit recharger"], "Layer 6 - head 16->2: tube logits", "2g_logits.png")
    layer_grid(probs, ["P(seeker here)", "P(recharger here)"],
               "Tubes - softmax over the 64 cells of each time slice.  The drones fly the arg-max among reachable cells.",
               "2h_tubes.png", cmap="viridis", sym=False)


# ================================================================ FIG 3: growth over NCA iterations
def rgb(p_s, p_r):
    """seeker red, recharger blue, overlap magenta."""
    a = np.clip(p_s / 0.5, 0, 1); b = np.clip(p_r / 0.5, 0, 1)      # absolute scale: uniform (1/64) is dark
    img = np.stack([a, 0.25 * np.minimum(a, b), b], -1)
    return np.clip(img, 0, 1)


def fig_growth():
    its = [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]
    fig, axes = plt.subplots(len(its), len(TS), figsize=(0.62 * len(TS) + 1.4, 0.62 * len(its) + 1.0), squeeze=False)
    with torch.no_grad():
        for i, k in enumerate(its):
            P = torch.softmax(model.head(states[k]).reshape(1, 2, T1, -1), -1).reshape(1, 2, T1, cfg.H, cfg.W)[0].numpy()
            for j, t in enumerate(TS):
                ax = axes[i, j]
                ax.imshow(rgb(P[0, t], P[1, t]))
                ax.set_xticks([]); ax.set_yticks([])
                if i == len(its) - 1:
                    mark(ax, t)
                if i == 0:
                    ax.set_title(f"t={t}", fontsize=7)
                if j == 0:
                    ax.set_ylabel(f"iter {k}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Growth: the tubes after k NCA iterations (rows) for every time slice t (columns).\n"
                 "Red = seeker tube, blue = recharger tube, magenta = both (rendezvous / escort).  "
                 "Information enters at t=0 and spreads one slice per iteration.", fontsize=9, x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{OUT}/3_growth_over_iterations.png", dpi=95)
    plt.close(fig)
    print(f"{OUT}/3_growth_over_iterations.png")


# ================================================================ FIG 4: training
def fig_training():
    import json
    rounds = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    rows = []
    for rd in rounds:
        p = f"runs/v10/ckpt_{rd:05d}.pt"
        if not os.path.exists(p):
            continue
        m, _, _, _ = load(p)
        with torch.no_grad():
            lg, _ = m(cond, K)
        P = torch.softmax(lg.reshape(1, 2, T1, -1), -1).reshape(1, 2, T1, cfg.H, cfg.W)[0].numpy()
        gps, gpr, ok, gt, gn = greedy(lg, s, r, g)
        rows.append((rd, P, gps, gpr, ok, gt, gn))
    fig = plt.figure(figsize=(0.62 * len(TS) + 8.5, 0.62 * len(rows) + 4.2))
    gs = fig.add_gridspec(len(rows), len(TS) + 1, width_ratios=[1] * len(TS) + [13])
    for i, (rd, P, gps, gpr, ok, gt, gn) in enumerate(rows):
        for j, t in enumerate(TS):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(rgb(P[0, t], P[1, t]))
            ax.set_xticks([]); ax.set_yticks([])
            if t <= gt:
                ax.plot(gps[t][1], gps[t][0], "o", ms=3.5, mfc="none", mec="w", mew=1)
                ax.plot(gpr[t][1], gpr[t][0], "s", ms=3.5, mfc="none", mec="c", mew=1)
            ax.plot(g[0, 1], g[0, 0], "*", ms=5, color="lime", mec="k", mew=0.4)
            if i == 0:
                ax.set_title(f"t={t}", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"round {rd}\n{'SUCCESS' if ok else 'fail'}", fontsize=6.5, rotation=0, ha="right", va="center",
                              color="green" if ok else "red")
    # learning curves on the right
    log = [json.loads(l) for l in open("runs/v10/log.jsonl")]
    axc = fig.add_subplot(gs[: max(1, len(rows) // 2), -1])
    ev = [(x["round"], x["val_need"]) for x in log if "val_need" in x]
    ec = [(x["round"], x["val_need_closed"]) for x in log if "val_need_closed" in x]
    axc.plot(*zip(*ev), ".-", label="held-out, open-loop dream")
    axc.plot(*zip(*ec), ".-", label="held-out, closed-loop")
    sm = []
    acc = []
    for x in log:
        acc = (acc + [x["search_need"]])[-25:]
        sm.append(sum(acc) / len(acc))
    axc.plot([x["round"] for x in log], sm, label="search finds a reward")
    axc.axhline(0.75, ls="--", c="k", lw=0.7)
    axc.set_ylim(0, 1); axc.set_xlabel("training round"); axc.legend(fontsize=7); axc.grid(alpha=0.3)
    axc.set_title("success on recharge-needed tasks", fontsize=8)
    axl = fig.add_subplot(gs[max(1, len(rows) // 2):, -1])
    axl.axis("off")
    axl.text(0, 1, "ONE TRAINING ROUND (repeated 900x)\n\n"
             "1  sample 64 random tasks (no curriculum)\n"
             "2  GROW: 32 NCA iterations from the t=0 seeds\n"
             "3  SEARCH: 64 joint rollouts per task follow the two\n"
             "   tubes stochastically in the simulator (half with\n"
             "   noise shared by both drones)\n"
             "4  REWARD: simulator says 1 / 0; keep the earliest\n"
             "   rewarded rollout per task\n"
             "5  CREDIT: replay it with one drone random; if the\n"
             "   reward survives, that drone's part is not imitated\n"
             "6  LEARN: cross-entropy per time slice pulls the\n"
             "   tubes onto the rewarded worldlines (back-prop\n"
             "   through all 32 iterations); 60% of samples have\n"
             "   a random prefix of the past burned in\n\n"
             "Loss = sum over t, drone of  -w * log softmax(logits[t])[cell]\n"
             "Not an energy: no negatives, no contrastive term.",
             fontsize=7.6, va="top", family="monospace")
    fig.suptitle("Training: the same held-out task grown by checkpoints of the audited run (rows).  "
                 "Red/blue = seeker/recharger tube, circles/squares = the greedy flight it produces.",
                 fontsize=9, x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{OUT}/4_training.png", dpi=95)
    plt.close(fig)
    print(f"{OUT}/4_training.png")


fig_voxel_rule()
figs_layers()
fig_growth()
fig_training()
