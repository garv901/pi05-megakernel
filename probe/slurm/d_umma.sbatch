#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:25:00
#SBATCH --job-name=p05-dumma
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# PI05_D_UMMA: band gate on 3 seeds + interleaved default / d16 / d8 / ctl timing (probe/mk_d_umma_ab.py). Extensions are built beforehand on the login node.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-dumma
mkdir -p $R
PROBE_PREFIX=968 PROBE_OUT=$R ROUNDS=${ROUNDS:-40} .venv/bin/python -m megabench.sota.pi05.probe.mk_d_umma_ab 2>&1 | tail -80
echo "mk_d_umma_ab exit ${PIPESTATUS[0]} R=$R"
