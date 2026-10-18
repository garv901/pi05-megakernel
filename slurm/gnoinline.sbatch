#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-gnoinl
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# G as a noinline function (mk/SPEC-g-noinline.md), P968 k64: bitwise check, plain-binary per-stage stamps, e2e bench.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
V=k64,k64-gb1,k64-gn,k64-gb1-gn,k64-gb2-gn
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
echo "=== gbatch_equal"
$PY -m megabench.sota.pi05.gbatch_equal --prefix 968 --arms megakernel; echo "gbatch_equal exit $?"
echo "=== stamps"
export PROBE_PREFIX=968 MK_VARIANTS=$V PROBE_OUT=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-gnstamps
mkdir -p $PROBE_OUT
$PY -m megabench.sota.pi05.probe.mk_stamps_ab 2>&1 | tail -4; echo "stamps exit $?"
echo "=== bench"
$PY -m megabench.sota.pi05 --prefix 968 --arms megakernel --mk-variants $V --rounds 5 --no-profile --tag p968-gnoinline
echo "bench exit $?"
