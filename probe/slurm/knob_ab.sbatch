#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:25:00
#SBATCH --job-name=p05-knob-ab
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Generic knob A/B (probe/mk_knob_ab.py): env ARMS (JSON, first = baseline), TAG, ROUNDS.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-${TAG:-knob-ab}
mkdir -p $R
PROBE_PREFIX=968 PROBE_OUT=$R .venv/bin/python -m megabench.sota.pi05.probe.mk_knob_ab
echo "driver exit $? R=$R"
