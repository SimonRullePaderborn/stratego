"""The paired difference between two evaluations of one checkpoint on the same tasks.  RUN ON THE CLUSTER ONLY.

    python3 -I nca_drones/cluster/paired.py <first.json> <second.json> [bf16|fp32]

Both files are written by `analyze.py --prereg`, which stores closed-loop success on the 2,000 never-posed tasks
task by task.  Per task the script takes success in the second file minus success in the first, and prints the two
rates, the tasks lost and gained, the mean difference and its 95% interval: plus or minus 1.96 standard deviations
of the differences, divided by the square root of the number of tasks (PROGRAMME.md, section 14.1).  All in points.
The last item says whether the lower bound is above -2 points, which is what the iteration gate asks.

The tasks of an evaluation are those never posed to its checkpoint in training, so two files are on the same tasks
only if they evaluate the same checkpoint; files that name different checkpoints are refused.  Standard library only.
"""
import json
import math
import sys


def read(path):
    with open(path) as f:
        return json.load(f)


def how(d):
    """How the file's evaluation ran the rule: the multiple of its iterations, and the iterations per step if the state was kept."""
    return f"x{d.get('mult', 1)}" + (f" with a kept state, {d['ks']} per step" if d.get("ks") else "")


def main():
    a, b = read(sys.argv[1]), read(sys.argv[2])
    prec = sys.argv[3] if len(sys.argv) > 3 else "bf16"
    if a["ckpt"] != b["ckpt"]:
        sys.exit(f"refused: the files evaluate different checkpoints ({a['ckpt']} and {b['ckpt']})")
    x, y = (d["prereg"][prec]["new2000_closed_per_task"] for d in (a, b))
    n = len(x)
    if n < 2 or len(y) != n:
        sys.exit(f"refused: {len(x)} and {len(y)} tasks")
    d = [int(q) - int(p) for p, q in zip(x, y)]
    m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    lo, hi = 100 * (m - half), 100 * (m + half)
    print(f"[{prec}] {n} tasks, iterations {how(a)} -> {how(b)}: "
          f"{100 * x.count('1') / n:.2f}% -> {100 * y.count('1') / n:.2f}%; lost {d.count(-1)}, gained {d.count(1)}; "
          f"paired difference {100 * m:+.2f} points, 95% interval {lo:+.2f} to {hi:+.2f}; "
          f"lower bound above -2 points: {'yes' if lo > -2 else 'no'}")


if __name__ == "__main__":
    main()
