#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-p0
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Phase 0 GPU probes of the CuTe pi0.5 megakernel port: cooperative smoke, math parity, MMA tile parity, CUDA parity self-check.
#   sbatch phase0.sbatch
set -euo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-p0
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"

PKG=megabench.sota.pi05.cute
PY=.venv/bin/python
STAGES=pro,A,B,C3,I,full

$PY -m $PKG.probes.smoke_coop --out "$OUT" 2>&1 | tee "$OUT/smoke.log"
$PY -m $PKG.probes.math_parity --out "$OUT" 2>&1 | tee "$OUT/math.log"
$PY -m $PKG.probes.mma_tile --out "$OUT" 2>&1 | tee "$OUT/mma.log"

status=0
$PY -m $PKG.parity --selfcheck --prefix 128 --att-keys 16 --stages $STAGES --step 0 --layer 0 --out "$OUT" 2>&1 | tee "$OUT/parity_p128_k16.log" || status=1
$PY -m $PKG.parity --selfcheck --prefix 968 --att-keys 64 --stages $STAGES --step 0 --layer 0 --out "$OUT" 2>&1 | tee "$OUT/parity_p968_k64.log" || status=1
echo "results in $OUT"
exit $status
