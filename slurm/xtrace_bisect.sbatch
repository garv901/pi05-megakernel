#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-xt-bisect
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# Xtrace hang/fault bisection: one process per spliced CUDA k64 cubin in $1 (default bisect/), each killed after 90 s (a
# deadlocked cooperative kernel only ends with its process): ok / error / HANG. XT_WATCH=30: a launch still running after
# 30 s dumps where every warp stopped (XT_STUCK) before the kill.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
X=$PWD/$P/runs/xtrace-cpu
D=$X/${1:-bisect}
export PROBE_PREFIX=968 XT_WATCH=30
nvidia-smi --query-gpu=name,driver_version --format=csv
for c in $D/xtrace_gpu_*.cubin; do
  out=$(LD_PRELOAD=$X/coop_shim.so timeout -k 10 90 .venv/bin/python -m megabench.sota.pi05.probe.xtrace.xt_one $c 2>&1)
  rc=$?
  line=$(echo "$out" | grep XT_ONE | sed -E 's/"sites": \[[^]]*\]\]?, //')
  echo "$(basename $c) rc=$rc ${line:-HANG/NO-OUTPUT}" | cut -c1-300
  echo "$out" | grep XT_STUCK | cut -c1-3000
done
