#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-p6
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Phase 6: (1) parity of the in-kernel counter reset: CUDA vs CuTe byte compare of `full` (and truncated stops, which must also leave
# cnt / bar zero) + two back-to-back launches; (2) stage ladder CUDA vs CuTe; (3) bench arms. Most important first; each run prints
# its elapsed seconds; sections 2 / 3 are guarded by the time left (30 min budget). Exit status nonzero if any parity run fails.
#   sbatch phase6.sbatch
set -uo pipefail
cd /raid/garv901/vibe-megakernel
T0=$SECONDS
BUDGET=1740   # seconds (30 min minus slack)

source megabench/sota/slurm/env.sh

OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-p6
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"

PY=.venv/bin/python
status=0
left() { echo $((BUDGET - (SECONDS - T0))); }
run() {  # name, stages, step, layer, args...   (parity --compare; in-kernel reset unless --host-zero)
  local name=$1 stages=$2 step=$3 layer=$4; shift 4
  local t0=$SECONDS
  $PY -m megabench.sota.pi05.cute.parity --compare --stages "$stages" --step "$step" --layer "$layer" "$@" --out "$OUT" 2>&1 | tee "$OUT/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
  echo "ELAPSED $name $((SECONDS - t0)) s (left $(left) s)"
}
reset_check() {  # name, args...  (two back-to-back full launches: counters zero after each, 2nd == 1st)
  local name=$1; shift
  local t0=$SECONDS
  $PY -m megabench.sota.pi05.cute.parity --check-reset "$@" 2>&1 | tee "$OUT/$name.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
  echo "ELAPSED $name $((SECONDS - t0)) s (left $(left) s)"
}
guarded() {  # name, max seconds, command...  (skipped when less than 120 s are left; killed at min(max, left - 60))
  local name=$1 mx=$2; shift 2
  local l; l=$(left)
  if [ "$l" -lt 120 ]; then echo "SKIPPED $name (left $l s)"; return; fi
  local lim=$((l - 60)); [ "$lim" -gt "$mx" ] && lim=$mx
  local t0=$SECONDS
  timeout "$lim" "$@" 2>&1 | tee "$OUT/$name.log"
  echo "ELAPSED $name $((SECONDS - t0)) s (limit $lim s, left $(left) s)"
}
B128="--prefix 128 --att-keys 16"
B968="--prefix 968 --att-keys 64"

# 1. parity (in-kernel reset): full chunks
run p128_k16_s7301_full full 0 0 $B128 --seed 7301
run p128_k16_s7302_full full 0 0 $B128 --seed 7302
reset_check reset_p128_k16 $B128 --seed 7301
run p128_k16_stops pro,A,B,C3,D,I 0 0 $B128 --seed 7301        # truncated runs must leave cnt / bar zero too
run p128_k32_full full 0 0 --prefix 128 --att-keys 32 --seed 7301
run p128_k64_full full 0 0 --prefix 128 --att-keys 64 --seed 7301
run p968_k64_full full 0 0 $B968 --seed 7301
run p968_k64_mask200_full full 0 0 $B968 --seed 7301 --mask-tail 200
reset_check reset_p968_k64 $B968 --seed 7301 --mask-tail 200
run p128_k16_wpre0_full full 0 0 $B128 --seed 7301 --wpre 0
run p128_k16_dwpt1_full full 0 0 $B128 --seed 7301 --d-wpt 1
run p128_k16_fuse23_full full 0 0 $B128 --seed 7301 --fuse 23
# host_zero path (phase 5 behavior, also compares cnt)
run p128_k16_hostzero_full full 0 0 $B128 --seed 7301 --host-zero
run p128_k16_hostzero_c3 C3 0 0 $B128 --seed 7301 --host-zero

# 2. stage ladder CUDA vs CuTe (builds one CUDA extension and one CuTe kernel per stop; the P=968 ladder may be cut by the time guard)
guarded ladder_p128_k16 600 $PY -m megabench.sota.pi05.cute.perf $B128 --seed 7301 --out "$OUT"

# 3. bench arms (about a minute of compile + a few seconds of timing per variant); before the P=968 ladder so that it is not starved
guarded bench_p128 420 $PY -m megabench.sota.pi05 --prefix 128 --arms megakernel,cute --mk-variants k16,k32,k64 --seeds 7301 --rounds 3 --no-profile --tag p6-p128
guarded bench_p968 300 $PY -m megabench.sota.pi05 --prefix 968 --arms megakernel,cute --mk-variants k64 --seeds 7301 --rounds 3 --no-profile --tag p6-p968
guarded ladder_p968_k64 600 $PY -m megabench.sota.pi05.cute.perf $B968 --seed 7301 --out "$OUT"

echo "results in $OUT (status $status, elapsed $((SECONDS - T0)) s)"
exit $status
