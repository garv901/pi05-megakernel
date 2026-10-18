#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-gbtrace
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Per-CTA barrier trace (PTRACE) of the CUDA megakernel at P968 k64 for G batching 0 / 1 / 2 (mk/SPEC-g-batch.md).
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-gbtrace
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv
export PROBE_PREFIX=968
for v in k64 k64-gb1 k64-gb2; do
  echo "=== $v"
  export PROBE_OUT=$R/$v MK_VARIANT=$v MK_FLAGS=
  mkdir -p $PROBE_OUT
  $PY -m megabench.sota.pi05.probe.mk_trace 2>&1 | tail -6
done
ls -la $R/*
