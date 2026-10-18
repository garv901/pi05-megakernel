#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-prof
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# Profiling suite of the CuTe pi0.5 megakernel (see NOTES.md "Profiling"): per config P<p>_k<k>
#   1 plain build with PTX / SASS / cubin / MLIR dump + timing (10 steps, and 1 step for the action reference)
#   2 IKET, whole grid, 1 step      3 IKET, every 8th CTA, 10 steps      4 action bytes iket == plain
#   5 iket_summary.py               6 Nsight Compute --set full with source correlation, CSV of the details page
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
CFG=${CFG:-968 64}
OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-prof-p${CFG// /-k}
mkdir -p "$OUT"
nvidia-smi | tee "$OUT/nvidia-smi.txt"
ncu --version | tee "$OUT/ncu-version.txt"
PY=.venv/bin/python
RUNIKET=.venv/bin/run-iket
MOD=megabench.sota.pi05.cute.profile
KFILTER='regex:kernel_cutlass_pi05_megakernel.*'   # entry name from the PTX: kernel_cutlass_pi05_megakernel_tensorptr...
export CUTE_DSL_NO_CACHE=1 CUTE_DSL_DISABLE_FILE_CACHING=1  # every step compiles fresh (lineinfo / iket must not hit a cached build)
status=0
note() { echo "STEP $1: $2" | tee -a "$OUT/status.txt"; }
# run LOG CMD...: tee the log, remember a failure, go on
run() { local log=$1; shift; "$@" 2>&1 | tee "$log"; local rc=${PIPESTATUS[0]}; if [ "$rc" -ne 0 ]; then status=1; note "$log" "FAILED rc=$rc"; else note "$log" ok; fi; }
# ranges per warp ~ per step (9 stage + ~6 sync + 4 wpre + ~2 cwait) x 18 layers + prologue / exit; 8 warps per CTA; 2 timestamps each
estimate() { $PY - "$@" <<'PYEOF'
import sys
ctas, steps, launches = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
per_warp = steps * 18 * 21 + 12
print(f"ESTIMATE timestamps: {ctas} CTAs x 8 warps x {per_warp} ranges x 2 x {launches} launches = {ctas * 8 * per_warp * 2 * launches / 1e6:.2f} M (limit ~2 M)")
PYEOF
}

# one config per job (priority partition: 30 min): default 968 64; P128 only as a secondary run: sbatch --export=ALL,CFG="128 16" profile.sbatch
for CFG in "${CFG:-968 64}"; do
  set -- $CFG; P=$1; K=$2
  D=$OUT/P${P}_k${K}
  mkdir -p "$D"
  ARGS="--prefix $P --att-keys $K --seed 7301"
  echo "===== config P=$P k=$K ====="
  # 1. plain, with the dump of every compiler artifact (LINEINFO keeps the Python line info in PTX / SASS)
  CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=all CUTE_DSL_DUMP_DIR=$D/dsl CUTE_DSL_NO_CACHE=1 CUTE_DSL_DISABLE_FILE_CACHING=1 \
    run "$D/plain_s10.log" $PY -m $MOD $ARGS --mode plain --n-steps 10 --launches 20 --save-actions "$D/actions_plain_s10.pt"
  run "$D/plain_s1.log" $PY -m $MOD $ARGS --mode plain --n-steps 1 --launches 5 --save-actions "$D/actions_plain_s1.pt"
  # 2. IKET, whole grid, one step (1 warmup + 1 measured launch are both in the trace; the summary reads the last)
  estimate 148 1 2 | tee "$D/estimate_iket_s1_all.txt"
  run "$D/iket_s1_all.log" $RUNIKET --output-dir "$D/iket_s1_all" --clobber profile --postprocess all -- \
    $PY -m $MOD $ARGS --mode iket --n-steps 1 --warmups 1 --launches 1 --save-actions "$D/actions_iket_s1.pt"
  # 3. IKET, every 8th CTA (19 of 148), full chunk; no warmup launch (keeps the event count down)
  estimate 19 10 1 | tee "$D/estimate_iket_s10_cta8.txt"
  PI05_IKET_CTA_STRIDE=8 run "$D/iket_s10_cta8.log" $RUNIKET --output-dir "$D/iket_s10_cta8" --clobber profile --postprocess all -- \
    $PY -m $MOD $ARGS --mode iket --n-steps 10 --warmups 0 --launches 1 --save-actions "$D/actions_iket_s10.pt"
  # 4. the markers must not change the result
  for S in 1 10; do
    case $S in 1) IK=$D/actions_iket_s1.pt ;; 10) IK=$D/actions_iket_s10.pt ;; esac
    $PY -m $MOD --compare "$D/actions_plain_s$S.pt" "$IK" 2>&1 | tee -a "$D/compare.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { status=1; note "compare s$S" "FAIL"; }
  done
  # 5. summaries
  run "$D/summary_s1_all.log" $PY megabench/sota/pi05/cute/iket_summary.py "$D/iket_s1_all" --out "$D/iket_s1_all/summary.md"
  run "$D/summary_s10_cta8.log" $PY megabench/sota/pi05/cute/iket_summary.py "$D/iket_s10_cta8" --out "$D/iket_s10_cta8/summary.md"
  # 6. Nsight Compute: only the measured launch (--profile-from-start off); source correlation needs LINEINFO + --import-source
  mkdir -p "$D/ncu"
  CUTE_DSL_LINEINFO=1 run "$D/ncu/ncu.log" ncu --set full --import-source yes --source-folders /raid/garv901/vibe-megakernel \
    --profile-from-start off --clock-control none --cache-control all -k "$KFILTER" --launch-count 1 \
    -o "$D/ncu/pi05_cute_full" -f $PY -m $MOD $ARGS --mode ncu --n-steps 10 --launches 1
  if [ -f "$D/ncu/pi05_cute_full.ncu-rep" ]; then
    run "$D/ncu/details_csv.log" bash -c "ncu -i '$D/ncu/pi05_cute_full.ncu-rep' --page details --csv > '$D/ncu/pi05_cute_full_details.csv'"
  else
    status=1; note "ncu" "no .ncu-rep produced"
  fi
done
echo "results in $OUT (status $status)"
exit $status
