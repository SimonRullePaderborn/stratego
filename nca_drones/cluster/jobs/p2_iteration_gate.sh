#!/bin/bash
# Phase 2, the iteration gate (PROGRAMME.md, section 14.1) for the checkpoint of one finished run.
#
# RUN ON THE CLUSTER ONLY, from nca_drones/ of a run folder that holds runs/<name>/best_r900.pt and, for the
# comparison with the run's own evaluation, results/p2/<name>.json:
#
#     bash cluster/jobs/p2_iteration_gate.sh <name>
#
# The checkpoint is evaluated by the protocol (analyze.py --prereg) with every iteration count multiplied by 1, 2,
# 4 and 8: the iterations of growth and, for a kept state, also the iterations per step. Success is stored task by
# task, and cluster/paired.py prints the paired difference of each multiple to the evaluation at 1 on the same
# 2,000 never-posed tasks. The gate asks that the lower 95% bound stays above -2 points. Per precision, the first
# line of the result compares the run's own evaluation with the evaluation at 1 made here: the same checkpoint on
# the same tasks, so no task may change. Arguments for the evaluation, such as "--device cuda", are taken from the
# environment variable ANALYZE_ARGS.
set -euo pipefail
name=$1
py=${PY:-/wf/venv/bin/python3}
ck=runs/$name/best_r900.pt
[ -e $ck ] || { echo "$ck is missing"; exit 1; }
mkdir -p results/p2
for m in 1 2 4 8; do
    $py analyze.py $ck --prereg ${ANALYZE_ARGS:-} --mult $m --out results/p2/$name.x$m.json > results/p2/$name.x$m.out 2>&1
done
{
    for prec in bf16 fp32; do
        if [ -e results/p2/$name.json ]; then $py -I cluster/paired.py results/p2/$name.json results/p2/$name.x1.json $prec; fi
        for m in 2 4 8; do $py -I cluster/paired.py results/p2/$name.x1.json results/p2/$name.x$m.json $prec; done
    done
} > results/p2/$name.gate.out 2>&1
echo "done gate $name"
