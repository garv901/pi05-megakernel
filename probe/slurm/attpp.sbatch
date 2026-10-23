#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:25:00
#SBATCH --job-name=p05-attpp
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# K2 two-half pipelined attention silo (k2_tma vs k2_tma_ex2 / k2_pp / k2_pp_ex2). Binaries built into ATTD/bin before submission:
# nvcc -O3 sm_100a -DATT_HOOKS -DK2_MASK=0x3800004 (att_k2; _ns: -DATT_STAMPS=0; _mk: -DATT_MARKS=1; _fb: -DATT_STAMPS=0 -DK2_PP_FORCE_FB).
# env: ATTD=<run dir with bin/> (required), ATT_REPS (20), K2_VARS
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05; U=$P/probe/ubench/att; D=${ATTD:?set ATTD}; B=$D/bin; REPS=${ATT_REPS:-20}
VARS=${K2_VARS:-null,k2_tma,k2_tma_ex2,k2_pp,k2_pp_ex2}
CTX="--smem 197632 --cluster 2 --ncal 35000"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
echo "=== forced-fallback correctness"
mkdir -p $D/fbrun/raw
timeout 200 $B/att_k2_fb --reps 2 --variants null,k2_pp,k2_pp_ex2 --out $D/fbrun --name fb $CTX > $D/fb.stdout 2> $D/fb.stderr; echo "fb rc=$?"
for r in 1 2; do
  R=$D/r$r; mkdir -p $R/raw
  for b in att_k2 att_k2_ns; do cuobjdump -res-usage $B/$b > $R/resusage_$b.txt 2>&1; done
  ORD="att_k2 att_k2_ns"; [ $r = 2 ] && ORD="att_k2_ns att_k2"
  for b in $ORD; do timeout 400 $B/$b --reps $REPS --variants $VARS --out $R --name ${b} $CTX > $R/$b.stdout 2> $R/$b.stderr; echo "r$r $b rc=$?"; done
  .venv/bin/python -I $U/att_analyze.py $R att_k2 2>&1 | tee $R/analyze.txt
done
timeout 300 $B/att_k2_mk --reps 6 --variants $VARS --out $D/r1 --name att_k2_mk $CTX > $D/r1/mk.stdout 2> $D/r1/mk.stderr
.venv/bin/python -I $U/att_k2_report.py $D/r1 att_k2 2>&1 | tee $D/r1/report.txt
grep -H "PPFB\|timeouts" $D/*.stderr $D/r*/*.stderr
echo "D=$D"
