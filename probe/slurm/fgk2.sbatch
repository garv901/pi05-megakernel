#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --job-name=p05-fgk2
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# F+G study, K2 (HMMA control for the redesigned F+G). Binaries built beforehand on the login node (build lines in fg_k2.cu):
#   fg_k2 (stamped), fg_k2_ns (-DFG_STAMPS=0 twin), fg_k2_tl (-DK2_TL=1 attribution build). All with -DFG_HOOKS.
# env: FGD=<run dir> (default probe/runs/<utc>-fgk2), FG_REPS (20)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${FGD:-$P/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-fgk2}
mkdir -p $D/raw
U=$P/probe/ubench/fg
REPS=${FG_REPS:-30}
VARS=null,k2_m2k4_s128_d3,k2_m4k2_s128_d3,k2_m2n2k2_s128_d3,k2_m4k2_s128_d3_nf,k2_m2k4_s128_d3_nf,k2_m4k2_s128_d3_pre1_nf,k2_m4k2_s64_d6_nf,k2_m4k2_s128_d4_nf,k2_m4k2_s128_d2_nf,k2_m4k2_s128_d3_nf_wf0,k2_m4k2_s128_d3_nf_accg,calib
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
nvidia-smi --query-gpu=timestamp,clocks.sm --format=csv,noheader,nounits -lms 100 > $D/clocks_sm.csv &
SAMP=$!
for b in fg_k2 fg_k2_ns; do cuobjdump -res-usage $U/$b > $D/resusage_$b.txt 2>&1; done
echo "=== harness (stamped)"
timeout 400 $U/fg_k2 --reps $REPS --variants $VARS --out $D --name fg_k2 > $D/fg_k2.stdout 2> $D/fg_k2.stderr; echo "fg_k2 rc=$?"
echo "=== harness (unstamped)"
timeout 400 $U/fg_k2_ns --reps $REPS --variants $VARS --out $D --name fg_k2_ns > $D/fg_k2_ns.stdout 2> $D/fg_k2_ns.stderr; echo "fg_k2_ns rc=$?"
echo "=== timeline attribution build"
mkdir -p $D/tl
timeout 300 $U/fg_k2_tl --reps 5 --variants $VARS --out $D/tl --name fg_k2_tl > $D/fg_k2_tl.stdout 2> $D/fg_k2_tl.stderr; echo "fg_k2_tl rc=$?"
kill $SAMP 2>/dev/null
echo "=== analyze"
.venv/bin/python -I $U/fgk2_report.py $D 2>&1 | tee $D/analyze.txt
.venv/bin/python -I $U/fg_analyze.py $D fg_k2 > $D/fg_analyze_strict.txt 2>&1   # strict harness analyzer (REFUSES on any per-launch gate fail)
echo "D=$D"
