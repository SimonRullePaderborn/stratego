#!/usr/bin/env bash
# Read-only facts about a cluster node, for planning. It changes nothing.
#
#   bash nca_drones/cluster/probe.sh > probe_$(hostname).txt 2>&1
#
# Run it on the cluster (login node, and once inside a GPU allocation if GPUs are not visible on the login node).
# Never run it on the laptop: the laptop only issues ssh and git commands.
# It runs no git and contacts no GitHub address: all code and all git go through the Mac (PROGRAMME.md, 4.3).
set -u

sec() { printf '\n== %s ==\n' "$1"; }

sec "host"
hostname
uname -srm
( . /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown OS}" )
date -u +"%Y-%m-%dT%H:%M:%SZ"

sec "cpu and memory"
nproc
if command -v lscpu >/dev/null; then lscpu | grep -E 'Model name|Socket|Core|Thread|^CPU\(s\)'; fi
free -g 2>/dev/null | head -2

sec "gpus visible on this node"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
else
    echo "nvidia-smi not found on this node (normal for a login node)"
fi

sec "scheduler"
for c in sinfo squeue sbatch srun scontrol qsub bsub kubectl; do
    printf '%-9s %s\n' "$c" "$(command -v "$c" || echo -)"
done
if command -v sinfo >/dev/null; then
    echo "-- partitions: name, state, time limit, nodes, gpus, cpus per node, memory per node"
    sinfo -o '%P %a %l %D %G %c %m' 2>/dev/null | head -40
fi
if command -v sacctmgr >/dev/null; then
    echo "-- my associations: account, partition, qos, group limits, per-job limits, wall-time limit"
    sacctmgr -n -p show assoc user="$USER" format=account,partition,qos,grptres,maxtres,maxwall 2>/dev/null | head -20
fi
if command -v squeue >/dev/null; then
    echo "-- my jobs"
    squeue -u "$USER" 2>/dev/null | head -20
fi

sec "python"
for p in python3 python; do
    if command -v "$p" >/dev/null; then printf '%s: ' "$(command -v "$p")"; "$p" --version 2>&1; fi
done
if command -v conda >/dev/null; then conda --version; fi
if type module >/dev/null 2>&1; then echo "environment modules: available"; else echo "environment modules: not found in this shell"; fi
python3 - <<'PY' 2>/dev/null || echo "torch is not importable in the default python3"
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version(),
      "| cuda available", torch.cuda.is_available(), "| devices", torch.cuda.device_count())
PY

sec "disk"
df -h "$HOME" . 2>/dev/null
for v in SCRATCH WORK TMPDIR; do
    d="${!v:-}"
    if [ -n "$d" ]; then echo "$v=$d"; df -h "$d" 2>/dev/null | tail -1; fi
done

sec "network (HTTP status; 000 means unreachable)"
for u in https://pypi.org https://download.pytorch.org; do
    printf '%-34s ' "$u"
    curl -sS -m 8 -o /dev/null -w '%{http_code}\n' "$u" 2>&1 | tail -1
done

sec "code (sent from the Mac; the commit is in COMMIT next to nca_drones/)"
here="$(cd "$(dirname "$0")/../.." && pwd)"
echo "$here"
cat "$here/COMMIT" 2>/dev/null || echo "no COMMIT file in $here"
