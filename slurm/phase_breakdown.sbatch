#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:40:00
#SBATCH --job-name=pi05-phasebd
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# Per-phase breakdown of default vs PI05_FG_UMMA megakernel at P968 k64 (probe/mk_phase_breakdown.py). Measurement only.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-phase-breakdown
mkdir -p $R
PROBE_PREFIX=968 PROBE_OUT=$R $PY -m megabench.sota.pi05.probe.mk_phase_breakdown 2>&1 | tail -60
echo "exit ${PIPESTATUS[0]}"
