#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-abl-par
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# FUSE_ABL (fuse 50 = 18 + 32, see NOTES.md "FUSE_ABL") parity: CUDA (fuse & 31 = 18) vs CuTe byte compare. Stop A does not exist with
# FUSE_ABL; `part` is excluded from the stop B / C3 compares (ABL never writes the A partials). One line "STEP name: ok|FAILED" per run.
#   sbatch abl_parity.sbatch
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-abl-parity
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"
PY=.venv/bin/python
status=0
note() { echo "STEP $1: $2" | tee -a "$OUT/status.txt"; }
# cmp NAME ARGS...: parity --compare with --fuse 50
cmp_run() {
  local name=$1; shift
  local t0=$SECONDS
  $PY -m megabench.sota.pi05.cute.parity --compare --fuse 50 "$@" --out "$OUT" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then status=1; note "$name" "FAILED rc=$rc"; else note "$name" ok; fi
  echo "ELAPSED $name $((SECONDS - t0)) s"
}
B128="--prefix 128 --att-keys 16"
B968="--prefix 968 --att-keys 64"

# most important first
cmp_run p128_k16_stops_s0l0 $B128 --seed 7301 --stages pro,B,C3,D,I --step 0 --layer 0
cmp_run p128_k16_stop_I_s1l5 $B128 --seed 7301 --stages I --step 1 --layer 5
cmp_run p128_k16_full $B128 --seed 7301 --stages full
cmp_run p968_k64_stops_B_I $B968 --seed 7301 --stages B,I --step 0 --layer 0
cmp_run p968_k64_full $B968 --seed 7301 --stages full
cmp_run p968_k64_mask200_full $B968 --seed 7301 --mask-tail 200 --stages full
# two back-to-back full launches, no host zeroing: counters zero after each, 2nd == 1st
t0=$SECONDS
$PY -m megabench.sota.pi05.cute.parity --check-reset --fuse 50 $B128 --seed 7301 2>&1 | tee "$OUT/reset_p128_k16.log"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then status=1; note reset_p128_k16 "FAILED rc=$rc"; else note reset_p128_k16 ok; fi
echo "ELAPSED reset_p128_k16 $((SECONDS - t0)) s"
echo "results in $OUT (status $status)"
exit $status
