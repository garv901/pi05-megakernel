#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-abl-perf
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# FUSE_ABL timing (see NOTES.md "FUSE_ABL"): per config, 3 reps interleaved of the stage ladder (pro,B,C3,I,I@1,full; --iters 50) once with
# --fuse 18 and once with --fuse 50 (the CUDA column is the fuse-18 CUDA kernel in both), all LADDER lines into $D/ladder.log; then
# one whole-grid IKET step with --fuse 50 and iket_summary.py. One config per job (30 min), like profile.sbatch:
#   sbatch abl_perf.sbatch                                      (P=968 k64, the default)
#   sbatch --export=ALL,CFG="128 16" abl_perf.sbatch            (P=128 k16, secondary)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
CFG=${CFG:-968 64}
OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-abl-perf-p${CFG// /-k}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv | tee "$OUT/gpu.txt"
PY=.venv/bin/python
RUNIKET=.venv/bin/run-iket
MOD=megabench.sota.pi05.cute.profile
status=0
note() { echo "STEP $1: $2" | tee -a "$OUT/status.txt"; }
run() { local log=$1; shift; "$@" 2>&1 | tee "$log"; local rc=${PIPESTATUS[0]}; if [ "$rc" -ne 0 ]; then status=1; note "$log" "FAILED rc=$rc"; else note "$log" ok; fi; }

set -- $CFG; P=$1; K=$2
D=$OUT/P${P}_k${K}
mkdir -p "$D"
echo "===== config P=$P k=$K ====="
# 1. ladders, reps interleaved (fuse 18, fuse 50, fuse 18, ...); perf prints LADDER lines and a table per run
for REP in 1 2 3; do
  for F in 18 50; do
    run "$D/perf_f${F}_r${REP}.log" $PY -m megabench.sota.pi05.cute.perf --prefix $P --att-keys $K --seed 7301 --iters 50 \
      --stages pro,B,C3,I,I@1,full --fuse $F
    grep '^LADDER ' "$D/perf_f${F}_r${REP}.log" | sed "s/^LADDER /LADDER rep=$REP /" >> "$D/ladder.log"
  done
done
# 2. IKET, whole grid, one step, FUSE_ABL (AB range + nested wpre / sync / cwait)
CUTE_DSL_NO_CACHE=1 CUTE_DSL_DISABLE_FILE_CACHING=1 run "$D/iket_s1_all_f50.log" $RUNIKET --output-dir "$D/iket_s1_all_f50" --clobber profile \
  --postprocess all -- $PY -m $MOD --prefix $P --att-keys $K --seed 7301 --fuse 50 --mode iket --n-steps 1 --warmups 1 --launches 1
run "$D/summary_s1_all_f50.log" $PY megabench/sota/pi05/cute/iket_summary.py "$D/iket_s1_all_f50" --out "$D/iket_s1_all_f50/summary.md"
echo "results in $OUT (status $status)"
exit $status
