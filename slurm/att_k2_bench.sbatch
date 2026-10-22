#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:40:00
#SBATCH --job-name=pi05-attk2b
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# e2e bench at P968: sota-graph (graph-pdl / graph-nopdl / eager) vs megakernel k64 (default), k64-fgu (PI05_FG_UMMA), k64-fgu-att (+ PI05_ATT_K2) and k64-att
# (PI05_ATT_K2 alone). Same bench.py invocation as the p968-part-bf16 run, plus sota-graph for the graph-pdl baseline; MegaBench band check + L2-flushed isolated timing.
# The megakernel extensions are built beforehand on the login node.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
$PY -m megabench.sota.pi05 --prefix 968 --arms megakernel,sota-graph \
  --mk-variants k64,k64-fgu,k64-fgu-att,k64-att --rounds 5 --no-profile --tag p968-att-k2
echo "bench exit $?"
