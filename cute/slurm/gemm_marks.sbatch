#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-cute-gmarks
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/cute/runs/slurm-%j.log
# gemm_phase sub-stage point markers (PI05_IKET_GEMM=1, see specs/SPEC-gemm-marks.md and NOTES.md "Profiling"), config CFG="P K" (default 968 64):
#   1 plain build with lineinfo + dump, timing (10 steps) and the 1-step action reference
#   2 IKET, stage ranges only (every 2nd CTA, 1 step)       3 IKET + PI05_IKET_GEMM=1 (same settings), lineinfo + dump
#   4 action bytes iket == plain (compare.log)   5 CUDA-event time of 2 vs 3 = the cost of the gemm markers (status.txt)
#   6 gemm_marks_check.py on the two cubins, iket_summary.py (plain summary of run 2, --gemm summary of run 3)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
CFG=${CFG:-968 64}; set -- $CFG; P=$1; K=$2
GEMM_LOOP=${GEMM_LOOP:-0}  # 1 = also the tile-loop markers (finer loop split, but ptxas spills change)
OUT=megabench/sota/pi05/cute/runs/$(date -u +%Y%m%dT%H%M%SZ)-gmarks-p${P}-k${K}-loop${GEMM_LOOP}
mkdir -p "$OUT"
nvidia-smi | tee "$OUT/nvidia-smi.txt"
PY=.venv/bin/python
RUNIKET=.venv/bin/run-iket
MOD=megabench.sota.pi05.cute.profile
C=megabench/sota/pi05/cute
export CUTE_DSL_NO_CACHE=1 CUTE_DSL_DISABLE_FILE_CACHING=1  # every step compiles fresh (lineinfo / iket must not hit a cached build)
status=0
note() { echo "STEP $1: $2" | tee -a "$OUT/status.txt"; }
# run LOG CMD...: tee the log, remember a failure, go on
run() { local log=$1; shift; "$@" 2>&1 | tee "$log"; local rc=${PIPESTATUS[0]}; if [ "$rc" -ne 0 ]; then status=1; note "$log" "FAILED rc=$rc"; else note "$log" ok; fi; }
# timestamps = CTAs x 8 warps x (steps x 18 layers x (21 ranges x 2 + G gemm markers) + 24) x launches; G = 12 with
# PI05_IKET_GEMM_LOOP=0 (3 gemm_phase calls per layer x 4 entry markers), 36 is the upper bound with the loop markers
estimate() { $PY - "$@" <<'PYEOF'
import sys
ctas, steps, launches, gemm = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
per_warp = steps * 18 * (21 * 2 + gemm) + 24
print(f"ESTIMATE timestamps: {ctas} CTAs x 8 warps x {per_warp} per warp x {launches} launches = {ctas * 8 * per_warp * launches / 1e6:.2f} M (limit ~2 M)")
PYEOF
}

D=$OUT
ARGS="--prefix $P --att-keys $K --seed 7301"
export PI05_IKET_CTA_STRIDE=2
# 1. plain
CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=all CUTE_DSL_DUMP_DIR=$D/dsl_plain \
  run "$D/plain_s10.log" env -u PI05_IKET_CTA_STRIDE $PY -m $MOD $ARGS --mode plain --n-steps 10 --launches 20
run "$D/plain_s1.log" env -u PI05_IKET_CTA_STRIDE $PY -m $MOD $ARGS --mode plain --n-steps 1 --launches 5 --save-actions "$D/actions_plain_s1.pt"
# 2. IKET, stage ranges only (1 warmup + 1 measured launch are in the trace; the summary reads the last)
estimate 74 1 2 0 | tee "$D/estimate_iket_stage.txt"
CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=all CUTE_DSL_DUMP_DIR=$D/dsl_iket \
  run "$D/iket_stage.log" $RUNIKET --output-dir "$D/iket_stage" --clobber profile --postprocess all -- \
  $PY -m $MOD $ARGS --mode iket --n-steps 1 --warmups 1 --launches 1 --save-actions "$D/actions_iket_stage.pt"
# 3. IKET + gemm entry markers (GEMM_LOOP=0, default: the tile-loop markers change the spills, spec "Spill perturbation")
estimate 74 1 2 $([ "$GEMM_LOOP" = 1 ] && echo 36 || echo 12) | tee "$D/estimate_iket_gemm.txt"
PI05_IKET_GEMM=1 PI05_IKET_GEMM_LOOP=$GEMM_LOOP CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP=all CUTE_DSL_DUMP_DIR=$D/dsl_gemm \
  run "$D/iket_gemm.log" $RUNIKET --output-dir "$D/iket_gemm" --clobber profile --postprocess all -- \
  $PY -m $MOD $ARGS --mode iket --n-steps 1 --warmups 1 --launches 1 --save-actions "$D/actions_iket_gemm.pt"
# 4. the markers must not change the result
for R in stage gemm; do
  $PY -m $MOD --compare "$D/actions_plain_s1.pt" "$D/actions_iket_$R.pt" 2>&1 | tee -a "$D/compare.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { status=1; note "compare $R" "FAIL"; }
done
# 5. cost of the gemm markers: CUDA-event time of the measured launch, stage-only vs stage + gemm markers
$PY - "$D/iket_stage.log" "$D/iket_gemm.log" <<'PYEOF' 2>&1 | tee -a "$OUT/status.txt"
import json, sys
t = {}
for p in sys.argv[1:3]:
    for ln in open(p):
        if ln.startswith("PROFILE "):
            t[p] = json.loads(ln[8:])["launch_ms"]
if len(t) == 2:
    a, b = (t[p][-1] for p in sys.argv[1:3])
    print(f"GEMM MARKER COST: iket stage-only {a:.4f} ms, iket + gemm markers {b:.4f} ms, difference {1000 * (b - a):.1f} us")
else:
    print("GEMM MARKER COST: PROFILE line missing in one of the logs")
PYEOF
# 6. checker on the lineinfo cubins (plain build vs gemm-marker build; stage-only iket build vs gemm-marker build), summaries
PC=$(ls $D/dsl_plain/*.cubin 2>/dev/null | head -1); IC=$(ls $D/dsl_iket/*.cubin 2>/dev/null | head -1); GC=$(ls $D/dsl_gemm/*.cubin 2>/dev/null | head -1)
IP=$(ls $D/dsl_iket/*.ptx 2>/dev/null | head -1); GP=$(ls $D/dsl_gemm/*.ptx 2>/dev/null | head -1)
run "$D/check_plain_vs_gemm.log" $PY $C/gemm_marks_check.py "$PC" "$GC" --ptx "$GP" --out "$D/check_plain_vs_gemm.md"
run "$D/check_iket_vs_gemm.log" $PY $C/gemm_marks_check.py "$IC" "$GC" --ptx "$GP" --plain-ptx "$IP" --out "$D/check_iket_vs_gemm.md"
run "$D/summary_stage.log" $PY $C/iket_summary.py "$D/iket_stage" --out "$D/iket_stage/summary.md"
run "$D/summary_gemm.log" $PY $C/iket_summary.py "$D/iket_gemm" --gemm --out "$D/iket_gemm/summary.md"
echo "results in $OUT (status $status)"
exit $status
