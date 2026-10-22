#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:25:00
#SBATCH --job-name=pi05-lhoist
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# gemm_phase next-tile weight load hoisted above the partial stores (mk/SPEC-load-hoist.md; CUDA default since job 8779),
# P968 k64 (hoisted) vs k64-nolh, CUDA only: (a) bitwise check; (b) block-0 barrier stamps of the plain binary, random variant
# order per round; (c) e2e bench (MegaBench band check + timing).
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

echo "=== variant_equal"
$PY -m megabench.sota.pi05.variant_equal --prefix 968 --arms megakernel --variants k64,k64-nolh
echo "variant_equal exit $?"

echo "=== mk_stamps_ab"
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-lhstamps
mkdir -p $R
PROBE_PREFIX=968 MK_VARIANTS=k64,k64-nolh PROBE_OUT=$R $PY -m megabench.sota.pi05.probe.mk_stamps_ab 2>&1 | tail -6
echo "mk_stamps_ab exit ${PIPESTATUS[0]}"
ls -la $R

echo "=== bench"
$PY -m megabench.sota.pi05 --prefix 968 --arms megakernel \
  --mk-variants k64,k64-nolh --rounds 5 --no-profile --tag p968-load-hoist
echo "bench exit $?"
