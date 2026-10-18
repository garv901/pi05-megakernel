#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-mk
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# pi0.5 action chunk at B=1: ported megakernel vs SOTA graph baselines.
# Extra arguments go to `python -m megabench.sota.pi05`, e.g.
#   sbatch pi05.sbatch --seeds 7301 --rounds 1 --tag smoke
set -euo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

.venv/bin/python -m megabench.sota.pi05 "$@"
