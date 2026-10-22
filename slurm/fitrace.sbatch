#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:40:00
#SBATCH --job-name=pi05-fitrace
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# FlashInfer-profiler port: CUDA (PI05_FITRACE) bitwise check, overhead A/B (unbound / bound buffer), one trace per mode;
# CuTe (PI05_FIT 0..3, one process each) bitwise check vs CUDA, CuTe / CUDA time ratio, traces for 1 / 2; CPU analysis.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-fitrace
mkdir -p $R
echo "=== fitrace"
PROBE_PREFIX=968 MK_VARIANT=k64 PROBE_OUT=$R $PY -m megabench.sota.pi05.probe.fitrace
echo "fitrace exit $?"
for m in 0 1 2 3; do
  echo "=== cute_fitrace PI05_FIT=$m"
  PI05_FIT=$m PROBE_PREFIX=968 PROBE_OUT=$R $PY -m megabench.sota.pi05.probe.cute_fitrace
  echo "cute_fitrace $m exit $?"
done
echo "=== analyze"
$PY -m megabench.sota.pi05.probe.fitrace_analyze $R
echo "analyze exit $?"
ls -la $R
