"""Do two training logs agree round by round, apart from clock times?  RUN ON THE CLUSTER ONLY.

    python3 -I nca_drones/cluster/same_logs.py <log.jsonl> <log.jsonl> [number of rounds]

Each line of a log is one round as JSON.  Two runs that did exactly the same arithmetic write the same numbers,
digit for digit, in every field but the clock fields.  The script prints the first round and the fields that
differ, and exits with 1; it exits with 0 when all compared rounds agree.  A field that only one of the two logs
has (a later version of the trainer logs more) is not compared, and is named in the last line.  Standard library only.
"""
import json
import sys

CLOCK = ("time", "t_search")


def rounds(path):
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append({k: json.dumps(v) for k, v in d.items() if k not in CLOCK})   # as text: NaN equals NaN
    return out


def main():
    a, b = rounds(sys.argv[1]), rounds(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else min(len(a), len(b))
    if n == 0 or len(a) < n or len(b) < n:
        print(f"too few rounds: {len(a)} and {len(b)}, wanted {n}")
        sys.exit(2)
    only = set()
    for i in range(n):
        only |= set(a[i]) ^ set(b[i])
        keys = sorted(k for k in set(a[i]) & set(b[i]) if a[i][k] != b[i][k])
        if keys:
            print(f"round {i}: {len(keys)} fields differ, for example " +
                  "; ".join(f"{k}: {a[i][k]} against {b[i][k]}" for k in keys[:4]))
            sys.exit(1)
    note = f"; not compared, because only one log has them: {', '.join(sorted(only))}" if only else ""
    print(f"identical in all {len(set(a[0]) & set(b[0]))} compared fields for {n} rounds{note}")


if __name__ == "__main__":
    main()
