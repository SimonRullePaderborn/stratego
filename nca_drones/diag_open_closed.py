import json

import torch
from analyze import fresh_tasks, load
from env import manhattan
from train import evaluate, evaluate_closed
model, cfg, a, rnd = load('runs/v10/best_prereg.pt')
K = a["K"]
parts = [fresh_tasks(200, cfg, seed) for seed in range(5001, 5011)]
s, r, g = (torch.cat([p[i] for p in parts]) for i in range(3))
dsr = manhattan(s, r)
oo = evaluate(model, (s, r, g), cfg, K, True)
oc = evaluate_closed(model, (s, r, g), cfg, K, True)
o, c = oo["success"], oc["success"]
res = {}
for name, m in (("d<=5", dsr <= 5), ("d=6", dsr == 6), ("d=7", dsr == 7), ("d>=8", dsr >= 8), ("all", dsr >= 0)):
    n = int(m.sum())
    row = dict(n=n, open=float(o[m].float().mean()), closed=float(c[m].float().mean()),
               both=int((o & c & m).sum()), only_open=int((o & ~c & m).sum()), only_closed=int((~o & c & m).sum()), neither=int((~o & ~c & m).sum()),
               either=float((o | c)[m].float().mean()))
    cf = m & ~c
    met = oc["n_recharge"] > 0
    row["closed_fail_never_met"] = int((cf & ~met).sum()); row["closed_fail_met_then_died"] = int((cf & met & (oc["tau"] < cfg.T)).sum()); row["closed_fail_timeout"] = int((cf & met & (oc["tau"] >= cfg.T)).sum())
    of = m & ~o; meto = oo["n_recharge"] > 0
    row["open_fail_never_met"] = int((of & ~meto).sum()); row["open_fail_met_then_died"] = int((of & meto & (oo["tau"] < cfg.T)).sum()); row["open_fail_timeout"] = int((of & meto & (oo["tau"] >= cfg.T)).sum())
    b = o & c & m
    row["tau_open_when_both"] = float(oo["tau"][b].float().mean()); row["tau_closed_when_both"] = float(oc["tau"][b].float().mean())
    res[name] = row
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
json.dump(res, open('results/v10/diag_open_vs_closed.json', 'w'), indent=1)
