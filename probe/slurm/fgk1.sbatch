#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --job-name=p05-fgk1
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# F+G study, K1 (full tcgen05 fused F+G): validity harness with null + the K1 variants, stamped and unstamped twins (gate 2), ABAB interleaved by the
# harness (round-robin over --variants), plus the cta_group::2 cluster build (fg_k1c, all launches cluster-launched, null included).
# Binaries are built beforehand on the login node (build lines at the top of probe/ubench/fg/fg_k1.cu).
# --ncal 35000: the known-delay gate (gate 4) has a constant ~260 cycle barrier-skew bias that exceeds 2% of the default 8700-cycle delay on
# these binaries (also for null); a 35000-cycle delay brings it below 1%. Same gate, longer known delay.
# Rejected experiments (non-NF staging, per-chunk A' barriers) run under their own name (fg_k1x): their stamps perturb gates 2/3, so they are
# analysed separately and a refusal there does not hide the main configs.
# env: FGD=<run dir> (default probe/runs/<utc>-fgk1), FG_REPS (30), FG_ARGS (extra harness args)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${FGD:-$P/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-fgk1}
mkdir -p $D/raw
U=$P/probe/ubench/fg
REPS=${FG_REPS:-30}
V1=null,k1_nf_s16_d3,k1_nf_e8_a51_s16_d3,k1_nf_e8_a51_fast_ts_s16_d3,k1_nf_e8_a51_fast_ts_s32_d2,k1_nf_e8_a51_fast_ts_s16_d5,k1_nf_e8_a51_fast_ts_nopre,k1_nf_e8_a51_fast_ts_warm
VX=null,k1_stage_s16_d3,k1_nf_ach_s16_d3
V2=null,cl2_s16_d3,cl2_s16_d3_fast_ts,cl2_s32_d4_fast_ts,cl2_s64_d2_fast_ts,cl2_s128_d1_fast_ts,cl2_s128_d1_fast_ts_nopre,cl2_s128_d1_fast_ts_warm,cl2_epre_s32_d4_fast_ts,cl2_s128_d1_ts
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv,noheader,nounits -lms 100 > $D/clocks_sm.csv &
SAMP=$!
for b in fg_k1 fg_k1_ns fg_k1c fg_k1c_ns; do cuobjdump -res-usage $U/$b > $D/resusage_$b.txt 2>&1; done
cp $D/resusage_fg_k1.txt $D/resusage_fg_k1x.txt; cp $D/resusage_fg_k1_ns.txt $D/resusage_fg_k1x_ns.txt
for b in fg_k1 fg_k1_ns; do timeout 400 $U/$b --reps $REPS --ncal 35000 --variants $V1 --out $D ${FG_ARGS:-} > $D/$b.stdout 2> $D/$b.stderr; echo "$b rc=$?"; done
timeout 400 $U/fg_k1 --reps $REPS --ncal 35000 --variants $VX --name fg_k1x --out $D > $D/fg_k1x.stdout 2> $D/fg_k1x.stderr; echo "fg_k1x rc=$?"
timeout 400 $U/fg_k1_ns --reps $REPS --ncal 35000 --variants $VX --name fg_k1x_ns --out $D > $D/fg_k1x_ns.stdout 2> $D/fg_k1x_ns.stderr; echo "fg_k1x_ns rc=$?"
for b in fg_k1c fg_k1c_ns; do timeout 400 $U/$b --reps $REPS --ncal 35000 --variants $V2 --out $D ${FG_ARGS:-} > $D/$b.stdout 2> $D/$b.stderr; echo "$b rc=$?"; done
kill $SAMP 2>/dev/null
echo "=== analyze"
.venv/bin/python -I $U/fg_analyze.py $D fg_k1 fg_k1c fg_k1x 2>&1 | tee $D/analyze.txt
echo "D=$D"
