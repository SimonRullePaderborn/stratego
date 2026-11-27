#!/bin/bash
# Phase 2, old 8x8 code: one training run of the measured recipe (the settings of the stored run v10), then the
# pre-registered evaluation of the best.pt that exists when round 900 is reached (EVAL_PROTOCOL.md).
#
# RUN ON THE CLUSTER ONLY, from nca_drones/ of a run folder (PROGRAMME.md, section 9.2):
#
#     bash cluster/jobs/p2_old_recipe.sh <name> <seed> <shared> [further train.py arguments]
#
# for example:  p2s0_shared_s0 0 0.5
#
# The trainer is started with 1,500 rounds, which sets its learning-rate schedule as in the stored runs, and is
# stopped once round 900 is in its log. Everything that is kept lands in results/p2/. Arguments for the
# evaluation, such as "--device cuda", are taken from the environment variable ANALYZE_ARGS.
set -euo pipefail
name=$1; seed=$2; shared=$3; shift 3
py=${PY:-/wf/venv/bin/python3}
if [ -e runs/$name/log.jsonl ]; then echo "runs/$name exists already; use a new name"; exit 1; fi
mkdir -p results/p2
$py train.py --name $name --seed $seed --strict 1 --hind 0 --prune 0 --null_k 8 --shared $shared --ez 4 \
    --eps_max 0.3 --prefix 0.6 --hard_frac 0.25 --rounds 1500 --eval_closed 100 --eval_every 50 \
    --threads 2 --lr_final 0.2 "$@" > results/p2/$name.train.out 2>&1 &
pid=$!
until grep -q '"round": 900,' runs/$name/log.jsonl 2>/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "the trainer stopped early"; tail -n 5 results/p2/$name.train.out; exit 1; }
    sleep 30
done
cp runs/$name/best.pt runs/$name/best_r900.pt
kill $pid 2>/dev/null || true
wait $pid 2>/dev/null || true
$py analyze.py runs/$name/best_r900.pt --prereg ${ANALYZE_ARGS:-} --out results/p2/$name.json > results/p2/$name.out 2>&1
cp runs/$name/log.jsonl results/p2/$name.log.jsonl
cp runs/$name/best_r900.pt results/p2/$name.best_r900.pt
echo "done $name"
