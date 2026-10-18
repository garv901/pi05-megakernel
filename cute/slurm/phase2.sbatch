#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-p2
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Phase 2: CUDA vs CuTe byte parity at stages pro, A, B (every workspace buffer except bar), plus one informative timing line.
# Exit status nonzero if any compare fails. The `--step 0 --layer 1` stops are not reachable yet (stages C..I missing).
#   sbatch phase2.sbatch
set -uo pipefail
cd /raid/garv901/vibe-megakernel

source megabench/sota/slurm/env.sh

OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-p2
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"

PY=.venv/bin/python
status=0
run() {  # name, args...
  local name=$1; shift
  $PY -m megabench.sota.pi05.cute.parity "$@" --stages pro,A,B --out "$OUT" 2>&1 | tee "$OUT/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
}
for seed in 7301 7302; do
  run p128_k16_s$seed --compare --prefix 128 --att-keys 16 --seed $seed
done
run p968_k64_s7301 --compare --prefix 968 --att-keys 64 --seed 7301
run p128_k16_mask17_s7301 --compare --prefix 128 --att-keys 16 --seed 7301 --mask-tail 17
run p128_k16_fuse19_s7301 --compare --prefix 128 --att-keys 16 --seed 7301 --fuse 19   # counter hand-off A -> B
$PY -m megabench.sota.pi05.cute.parity --timing --stages B --prefix 128 --att-keys 16 --seed 7301 2>&1 | tee "$OUT/timing.log"
echo "results in $OUT (status $status)"
exit $status
