"""Learning curves from runs/<name>/log.jsonl.

  python3 plot.py out.png name[:label] name[:label] ...
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(name):
    return [json.loads(l) for l in open(f"runs/{name}/log.jsonl")]


def smooth(xs, k=25):
    out, acc = [], []
    for x in xs:
        if x == x:
            acc.append(x)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc) if acc else float("nan"))
    return out


if __name__ == "__main__":
    out = sys.argv[1]
    specs = [a.split(":", 1) if ":" in a else [a, a] for a in sys.argv[2:]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, label in specs:
        rows = load(name)
        r = [x["round"] for x in rows]
        axes[0].plot(r, smooth([x["search_need"] for x in rows]), label=label)
        ev = [(x["round"], x["val_need"]) for x in rows if "val_need" in x]
        axes[1].plot([e[0] for e in ev], [e[1] for e in ev], marker=".", label=label)
        ec = [(x["round"], x["val_need_closed"]) for x in rows if "val_need_closed" in x]
        if ec:
            axes[2].plot([e[0] for e in ec], [e[1] for e in ec], marker=".", label=label)
    axes[0].set_title("search: share of recharge-needed tasks with a rewarded rollout")
    axes[1].set_title("held-out validation, open-loop dream (200 tasks)")
    axes[2].set_title("held-out validation, closed-loop (100 tasks)")
    for a in axes[1:]:
        a.axhline(0.75, color="k", lw=0.6, ls="--")
        a.set_ylim(0, 1)
    for a in axes:
        a.set_xlabel("training round"); a.legend(fontsize=7); a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(out)
