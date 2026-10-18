#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-p3
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Phase 3: CUDA vs CuTe byte parity at stages B, C3 (every workspace buffer except bar), plus informative C3-stop timing lines.
# Exit status nonzero if any compare fails.
#   sbatch phase3.sbatch
set -uo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-p3
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"

PY=.venv/bin/python
status=0
run() {  # name, args...
  local name=$1; shift
  $PY -m megabench.sota.pi05.cute.parity "$@" --stages B,C3 --out "$OUT" 2>&1 | tee "$OUT/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
}
for seed in 7301 7302; do
  run p128_k16_s$seed --compare --prefix 128 --att-keys 16 --seed $seed
done
run p128_k32_s7301 --compare --prefix 128 --att-keys 32 --seed 7301
run p128_k64_s7301 --compare --prefix 128 --att-keys 64 --seed 7301
run p968_k64_s7301 --compare --prefix 968 --att-keys 64 --seed 7301
run p128_k16_mask17_s7301 --compare --prefix 128 --att-keys 16 --seed 7301 --mask-tail 17
run p968_k64_mask200_s7301 --compare --prefix 968 --att-keys 64 --seed 7301 --mask-tail 200   # realistic padded prompt
run p128_k16_fuse16_s7301 --compare --prefix 128 --att-keys 16 --seed 7301 --fuse 16          # CC3 off: barrier path
$PY -m megabench.sota.pi05.cute.parity --timing --stages C3 --prefix 128 --att-keys 16 --seed 7301 2>&1 | tee "$OUT/timing_p128_k16.log"
$PY -m megabench.sota.pi05.cute.parity --timing --stages C3 --prefix 968 --att-keys 64 --seed 7301 2>&1 | tee "$OUT/timing_p968_k64.log"
echo "results in $OUT (status $status)"
exit $status
