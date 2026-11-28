"""Means of a training log's fields over a range of rounds.  RUN ON THE CLUSTER ONLY.

    python3 -I nca_drones/cluster/log_means.py <log.jsonl> <first round> <last round> [field ...]

Each line of a log is one round as JSON.  For the rounds from the first to the last named, both included, the script
prints the mean of every field over the rounds that hold a number for it (a field that is missing or not a number
in a round is left out of that field's mean), then the seconds per round over the range, then the validation
numbers of every evaluation round in the range.  Without field names it takes the ones the cards of phase 2 quote.
Standard library only.
"""
import json
import sys

FIELDS = ("greedy_easy", "greedy_need", "search_easy", "search_need", "loss", "loss_fresh", "loss_past",
          "next_fresh", "next_past", "gnorm")
VAL = ("val_easy", "val_need", "val_need_closed")


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def main():
    path, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fields = sys.argv[4:] or FIELDS
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if lo <= d["round"] <= hi:
                rows.append(d)
    if not rows:
        sys.exit(f"no round between {lo} and {hi} in {path}")
    parts = []
    for k in fields:
        v = [r[k] for r in rows if number(r.get(k))]
        parts.append(f"{k} {sum(v) / len(v):.3f}" if v else f"{k} none")
    print(f"rounds {rows[0]['round']} to {rows[-1]['round']} ({len(rows)} rounds): " + ", ".join(parts))
    if len(rows) > 1:
        print(f"  seconds per round: {(rows[-1]['time'] - rows[0]['time']) / (rows[-1]['round'] - rows[0]['round']):.2f}")
    for r in rows:
        if any(number(r.get(k)) for k in VAL):
            print(f"  round {r['round']}: " + ", ".join(f"{k} {r[k]:.3f}" for k in VAL if number(r.get(k))))


if __name__ == "__main__":
    main()
