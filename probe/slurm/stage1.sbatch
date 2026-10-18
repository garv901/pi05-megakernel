#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --job-name=p05probe-s1
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Stage 1: megakernel per-CTA barrier trace (PTRACE) + nsys node trace + GPU metrics (DRAM BW vs time) for mk / graph-pdl / graph-nopdl.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
P=megabench.sota.pi05.probe
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-stage1
mkdir -p $D; export PROBE_OUT=$D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu.csv
NSYS=/usr/local/cuda/bin/nsys
$PY -m $P.mk_trace 2>&1 | tail -8 | tee $D/mk_trace.log
for arm in graph-pdl graph-nopdl mk; do
  echo "== nsys $arm"
  F=100000
  for try in 1 2; do
    $NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=$F \
      -o $D/nsys_$arm -f true $PY -m $P.drive $arm --reps 8 --flush --range-all 2>&1 | tail -3 && break
    F=10000
  done
  echo "freq used $F" | tee -a $D/nsys_freq.txt
  $NSYS stats --report cuda_gpu_trace --format csv --output $D/nsys_${arm}_trace $D/nsys_$arm.nsys-rep 2>&1 | tail -1
  $NSYS export --type sqlite -o $D/nsys_$arm.sqlite --force-overwrite true $D/nsys_$arm.nsys-rep 2>&1 | tail -1
done
ls -la $D
