#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-gbatch
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# Stage G load batching A/B at P=968 (mk/SPEC-g-batch.md): bitwise check, then e2e bench of
# CUDA and CuTe megakernels, each at k64 (original), k64-gb1, k64-gb2.
set -uo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

echo "=== gbatch_equal"
.venv/bin/python -m megabench.sota.pi05.gbatch_equal --prefix 968 --arms megakernel,cute
echo "gbatch_equal exit $?"

echo "=== bench"
.venv/bin/python -m megabench.sota.pi05 --prefix 968 --arms megakernel,cute \
  --mk-variants k64,k64-gb1,k64-gb2 --rounds 5 --no-profile --tag p968-gbatch
echo "bench exit $?"
