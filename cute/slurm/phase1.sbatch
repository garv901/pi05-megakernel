#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-p1
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Phase 1: CUDA vs CuTe byte parity at stage `pro`, plus one informative timing line. Exit status nonzero if any compare fails.
#   sbatch phase1.sbatch
set -uo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-p1
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"

PY=.venv/bin/python
status=0
run() {  # name, args...
  local name=$1; shift
  $PY -m megabench.sota.pi05.cute.parity "$@" --out "$OUT" 2>&1 | tee "$OUT/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
}
for seed in 7301 7302; do
  run p128_k16_s$seed --compare --stages pro --prefix 128 --att-keys 16 --seed $seed
  run p968_k64_s$seed --compare --stages pro --prefix 968 --att-keys 64 --seed $seed
  run p128_k16_mask17_s$seed --compare --stages pro --prefix 128 --att-keys 16 --seed $seed --mask-tail 17
done
$PY -m megabench.sota.pi05.cute.parity --timing --prefix 128 --att-keys 16 --seed 7301 2>&1 | tee "$OUT/timing.log"
echo "results in $OUT (status $status)"
exit $status
