"""Plot learning curves from runs/<name>/log.jsonl."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return rows


def smooth(xs, k=25):
    out, acc = [], []
    for x in xs:
        if x == x:
            acc.append(x)
        acc = acc[-k:]
        out.append(sum(acc) / len(acc) if acc else float("nan"))
    return out


if __name__ == "__main__":
    names = sys.argv[1:]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in names:
        rows = load(f"runs/{name}/log.jsonl")
        r = [x["round"] for x in rows]
        axes[0].plot(r, smooth([x["search_need"] for x in rows]), label=f"{name} search")
        axes[0].plot(r, smooth([x["greedy_need"] for x in rows]), "--", label=f"{name} greedy (train tasks)")
        ev = [(x["round"], x["val_need"]) for x in rows if "val_need" in x]
        axes[1].plot([e[0] for e in ev], [e[1] for e in ev], label=f"{name} val need")
        ee = [(x["round"], x["val_easy"]) for x in rows if "val_easy" in x]
        axes[1].plot([e[0] for e in ee], [e[1] for e in ee], ":", label=f"{name} val easy")
        axes[2].plot(r, smooth([x["loss"] for x in rows]), label=name)
    axes[0].set_title("recharge-needed tasks: success during training rounds")
    axes[1].set_title("held-out greedy success")
    axes[1].axhline(0.75, color="k", lw=0.5)
    axes[2].set_title("tube loss")
    for a in axes:
        a.set_xlabel("round"); a.legend(fontsize=7); a.grid(alpha=0.3)
    plt.tight_layout()
    out = f"runs/curves_{'_'.join(names)}.png"
    plt.savefig(out, dpi=110)
    print(out)
