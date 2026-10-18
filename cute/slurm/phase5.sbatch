#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-p5
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Phase 5: CUDA vs CuTe byte parity of the last-layer tail, step 1 and the full chunk (every workspace buffer except bar, plus
# actions), then informative full-chunk timings. Most important runs first; each run prints its elapsed seconds.
# Exit status nonzero if any compare fails.   sbatch phase5.sbatch
set -uo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-p5
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"

PY=.venv/bin/python
status=0
run() {  # name, stages, step, layer, args...
  local name=$1 stages=$2 step=$3 layer=$4; shift 4
  local t0=$SECONDS
  $PY -m megabench.sota.pi05.cute.parity --compare --stages "$stages" --step "$step" --layer "$layer" "$@" --out "$OUT" 2>&1 | tee "$OUT/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
  echo "ELAPSED $name $((SECONDS - t0)) s"
}
B128="--prefix 128 --att-keys 16"
B968="--prefix 968 --att-keys 64"
run p128_k16_s7301_full full 0 0 $B128 --seed 7301
run p128_k16_s7301_i17 I 0 17 $B128 --seed 7301
run p128_k16_s7301_step1_l0 A,I 1 0 $B128 --seed 7301
run p128_k16_s7301_step1_l17 A,I 1 17 $B128 --seed 7301
run p128_k16_s7302_full full 0 0 $B128 --seed 7302
run p128_k32_s7301_full full 0 0 --prefix 128 --att-keys 32 --seed 7301
run p128_k64_s7301_full full 0 0 --prefix 128 --att-keys 64 --seed 7301
run p968_k64_s7301_full full 0 0 $B968 --seed 7301
run p128_k16_mask17_full full 0 0 $B128 --seed 7301 --mask-tail 17
run p968_k64_mask200_full full 0 0 $B968 --seed 7301 --mask-tail 200
run p128_k16_wpre0_full full 0 0 $B128 --seed 7301 --wpre 0
run p128_k16_dwpt1_full full 0 0 $B128 --seed 7301 --d-wpt 1
run p128_k16_dwpt4_full full 0 0 $B128 --seed 7301 --d-wpt 4
run p128_k16_fuse16_full full 0 0 $B128 --seed 7301 --fuse 16
run p128_k16_fuse23_full full 0 0 $B128 --seed 7301 --fuse 23
$PY -m megabench.sota.pi05.cute.parity --timing --stages full $B128 --seed 7301 2>&1 | tee "$OUT/timing_p128_k16.log"
$PY -m megabench.sota.pi05.cute.parity --timing --stages full $B968 --seed 7301 2>&1 | tee "$OUT/timing_p968_k64.log"
echo "results in $OUT (status $status)"
exit $status
