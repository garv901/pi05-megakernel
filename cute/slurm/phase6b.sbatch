#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-p6b
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Phase 6 follow-up: rerun the stop parity (cnt KeyError fixed) and A/B the in-kernel reset vs host zero on the full chunk.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-p6b
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"
PY=.venv/bin/python
status=0
B128="--prefix 128 --att-keys 16"
$PY -m megabench.sota.pi05.cute.parity --compare --stages pro,A,B,C3,D,I --step 0 --layer 0 $B128 --seed 7301 --out "$OUT" 2>&1 | tee "$OUT/p128_k16_stops.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || status=1
$PY -m megabench.sota.pi05.cute.parity --compare --stages I --step 1 --layer 5 $B128 --seed 7301 --fuse 23 --out "$OUT" 2>&1 | tee "$OUT/p128_k16_fuse23_stopI.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || status=1
for rep in 1 2 3; do
  for hz in "" "--host-zero"; do
    $PY -m megabench.sota.pi05.cute.perf $B128 --stages pro,I@1,full --iters 50 $hz 2>&1 | grep LADDER | tee -a "$OUT/ab_p128.log"
  done
done
for rep in 1 2; do
  for hz in "" "--host-zero"; do
    $PY -m megabench.sota.pi05.cute.perf --prefix 968 --att-keys 64 --stages pro,I@1,full --iters 50 $hz 2>&1 | grep LADDER | tee -a "$OUT/ab_p968.log"
  done
done
echo "results in $OUT (status $status)"
exit $status
