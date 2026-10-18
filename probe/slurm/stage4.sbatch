#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=p05probe-s4
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Stage 4: megakernel ablation timings (prefetch off, WPRE off, fuse variants, barrier variant, k32).
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-stage4
mkdir -p $D; export PROBE_OUT=$D
timeout 900 .venv/bin/python -m megabench.sota.pi05.probe.ablate 2>&1 | tail -5
