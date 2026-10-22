#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --job-name=pi05-pb
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# bf16 split-K partials (PI05_PART_BF16, default 1 since job 8802), P968 k64 vs k64-nopb, CUDA only. Not bit-identical, so no variant_equal:
# (a) block-0 barrier stamps, random variant order per round; (b) e2e bench (MegaBench band check + timing).
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv

echo "=== mk_stamps_ab"
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-pbstamps
mkdir -p $R
PROBE_PREFIX=968 MK_VARIANTS=k64,k64-nopb PROBE_OUT=$R $PY -m megabench.sota.pi05.probe.mk_stamps_ab 2>&1 | tail -6
echo "mk_stamps_ab exit ${PIPESTATUS[0]}"

echo "=== bench"
$PY -m megabench.sota.pi05 --prefix 968 --arms megakernel \
  --mk-variants k64,k64-nopb --rounds 5 --no-profile --tag p968-part-bf16
echo "bench exit $?"
