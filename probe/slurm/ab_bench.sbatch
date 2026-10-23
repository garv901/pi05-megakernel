#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=p05-abbench
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# e2e bench at P968, all arms in one job: sota-graph (graph-pdl / graph-nopdl / eager), megakernel k64-noabu (the pre-flip SOTA: FG_UMMA + ATT_K2, commit 96ab46c) and k64 (PI05_AB_UMMA default 1 since job 9406; job 9406 itself ran k64 / k64-abu before the flip).
# bench.py: MegaBench band check on 3 seeds + L2-flushed isolated timing. The sota-graph build (autotune) takes ~16 min, the rest ~1 min.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
.venv/bin/python -m megabench.sota.pi05 --prefix 968 --arms megakernel,sota-graph --mk-variants k64-noabu,k64 --rounds 5 --no-profile --tag p968-abu
echo "bench exit $?"
