#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:15:00
#SBATCH --job-name=p05-fg
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# F+G study, step 1: validity harness (null + calib + K0, stamped and unstamped binaries, ABAB) and the megakernel default vs
# NULL_FG L0 differential. Binaries and the two torch extensions are built beforehand on the login node (build lines in fg_k0.cu).
# env: FGD=<existing run dir> (default: new probe/runs/<utc>-fg), FG_REPS (20), FG_ARGS (extra harness args, e.g. "--cold 1")
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${FGD:-$P/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-fg}
mkdir -p $D/raw
U=$P/probe/ubench/fg
REPS=${FG_REPS:-20}
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
# during-run SM clock sampler (the pre-run snapshot reads the idle 120 MHz)
nvidia-smi --query-gpu=timestamp,clocks.sm --format=csv,noheader,nounits -lms 100 > $D/clocks_sm.csv &
SAMP=$!
# gate 2 (register / spill / stack identical between stamped and unstamped builds)
for b in fg_k0 fg_k0_ns; do cuobjdump -res-usage $U/$b > $D/resusage_$b.txt 2>&1; done
echo "=== harness (stamped)"
timeout 300 $U/fg_k0 --reps $REPS --out $D ${FG_ARGS:-} > $D/fg_k0.stdout 2> $D/fg_k0.stderr; echo "fg_k0 rc=$?"
echo "=== harness (unstamped, FG_STAMPS=0)"
timeout 300 $U/fg_k0_ns --reps $REPS --out $D ${FG_ARGS:-} > $D/fg_k0_ns.stdout 2> $D/fg_k0_ns.stderr; echo "fg_k0_ns rc=$?"
if [ -n "${FG_SWEEP:-}" ]; then
  # context sensitivity of K0 - null (stamped binary only): C-phase length, context pieces (ctx bits 1 Wo pf, 2 Wd pf, 4 D partials, 8 act read), cold weights
  for cfg in "c14000:--ccyc 14000" "c25000:--ccyc 25000" "ctx7:--ctx 7" "ctx0:--ctx 0" "cold:--cold 1" "cw2:--cwarps 2" "cw4:--cwarps 4" "cw8:--cwarps 8"; do
    n=${cfg%%:*}; a=${cfg#*:}
    timeout 200 $U/fg_k0 --reps 10 --out $D --name fg_k0_$n $a > $D/fg_k0_$n.stdout 2> $D/fg_k0_$n.stderr; echo "sweep $n rc=$?"
  done
fi
if [ -z "${SKIP_MK:-}" ]; then
echo "=== megakernel default vs NULL_FG"
PROBE_PREFIX=968 PROBE_OUT=$D/mk ROUNDS=${MK_ROUNDS:-40} .venv/bin/python -m megabench.sota.pi05.probe.mk_nullfg_ab 2>&1 | tail -5
echo "mk rc=${PIPESTATUS[0]}"
fi
kill $SAMP 2>/dev/null
echo "=== analyze"
SW=""; [ -n "${FG_SWEEP:-}" ] && SW="fg_k0_c14000 fg_k0_c25000 fg_k0_ctx7 fg_k0_ctx0 fg_k0_cold fg_k0_cw2 fg_k0_cw4 fg_k0_cw8"
.venv/bin/python -I $U/fg_analyze.py $D fg_k0 $SW 2>&1 | tee $D/analyze.txt
echo "D=$D"
