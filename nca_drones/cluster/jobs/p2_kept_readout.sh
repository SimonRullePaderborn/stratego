#!/bin/bash
# Phase 2: the closed-loop readout of a stored checkpoint with a kept state, whatever the run was trained with.
#
# RUN ON THE CLUSTER ONLY, from nca_drones/ of a run folder that holds runs/<name>/best_r900.pt and the run's own
# evaluation results/p2/<name>.json:
#
#     bash cluster/jobs/p2_kept_readout.sh <name> <iterations per step> [more of them]
#
# For every number of iterations per step the checkpoint is evaluated by the protocol (analyze.py --prereg) with
# the kept readout forced on: the volume is grown once on the first observation with the run's own number of
# iterations, and after every step the new observation is burned in and the rule runs that many iterations on the
# state it has. Nothing is trained. cluster/paired.py then prints the paired difference of each to the run's own
# evaluation on the same 2,000 never-posed tasks. For a run that was trained with a kept state, the line with its
# own number of iterations per step repeats its evaluation, so no task may change there. Arguments for the
# evaluation, such as "--device cuda", are taken from the environment variable ANALYZE_ARGS.
set -euo pipefail
name=$1; shift
py=${PY:-/wf/venv/bin/python3}
ck=runs/$name/best_r900.pt
[ -e $ck ] || { echo "$ck is missing"; exit 1; }
[ -e results/p2/$name.json ] || { echo "results/p2/$name.json is missing"; exit 1; }
for ks in "$@"; do
    $py analyze.py $ck --prereg ${ANALYZE_ARGS:-} --kept 1 --ks $ks --out results/p2/$name.kept$ks.json > results/p2/$name.kept$ks.out 2>&1
done
{
    for prec in bf16 fp32; do
        for ks in "$@"; do $py -I cluster/paired.py results/p2/$name.json results/p2/$name.kept$ks.json $prec; done
    done
} > results/p2/$name.kept.out 2>&1
echo "done kept readout $name"
