#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-p4
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Phase 4: CUDA vs CuTe byte parity of stages D..I of step 0 (every workspace buffer except bar), then informative I-stop timings.
# Most important runs first; each run prints its elapsed seconds. Exit status nonzero if any compare fails.
#   sbatch phase4.sbatch
set -uo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-p4
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"

PY=.venv/bin/python
status=0
run() {  # name, stages, layer, args...
  local name=$1 stages=$2 layer=$3; shift 3
  local t0=$SECONDS
  $PY -m megabench.sota.pi05.cute.parity --compare --stages "$stages" --step 0 --layer "$layer" "$@" --out "$OUT" 2>&1 | tee "$OUT/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
  echo "ELAPSED $name $((SECONDS - t0)) s"
}
B128="--prefix 128 --att-keys 16"
B968="--prefix 968 --att-keys 64"
# every stage of layer 0, then the end of the layer at layers 1 and 16
run p128_k16_s7301_l0 D,E,F,G,H,I 0 $B128 --seed 7301
run p128_k16_s7301_l1 I 1 $B128 --seed 7301
run p128_k16_s7301_l16 I 16 $B128 --seed 7301
for l in 0 1 16; do
  run p128_k16_s7302_l$l I $l $B128 --seed 7302
  run p968_k64_s7301_l$l I $l $B968 --seed 7301
done
run p128_k16_mask17_l1 I 1 $B128 --seed 7301 --mask-tail 17
run p968_k64_mask200_l1 I 1 $B968 --seed 7301 --mask-tail 200
run p128_k16_wpre0_l1 I 1 $B128 --seed 7301 --wpre 0
run p128_k16_dwpt1_l1 I 1 $B128 --seed 7301 --d-wpt 1
run p128_k16_dwpt4_l1 I 1 $B128 --seed 7301 --d-wpt 4
run p128_k16_fuse16_l1 I 1 $B128 --seed 7301 --fuse 16
run p128_k16_fuse22_l1 I 1 $B128 --seed 7301 --fuse 22
run p128_k16_fuse23_l1 I 1 $B128 --seed 7301 --fuse 23
$PY -m megabench.sota.pi05.cute.parity --timing --stages I --layer 16 $B128 --seed 7301 2>&1 | tee "$OUT/timing_p128_k16.log"
$PY -m megabench.sota.pi05.cute.parity --timing --stages I --layer 16 $B968 --seed 7301 2>&1 | tee "$OUT/timing_p968_k64.log"
echo "results in $OUT (status $status)"
exit $status
