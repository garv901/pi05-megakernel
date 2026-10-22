#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:25:00
#SBATCH --job-name=p05-attconf
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Attention confirmation job: att_k0 (null,k0), att_k1 best (null,k0,r2_bf_sm2_c128), att_k2 (null,k0,k2_tma,k2_tma_bf,k2_tma_bf_h) in ONE job, one gate standard:
# same --ncal 35000 / --smem 197632 --cluster 2 for every binary, stamped + unstamped twins, 2 rounds (round 2 reverses the binary order).
# Binaries built fresh (login node) into ATTD/bin before submission: see build lines in the job notes (nvcc -O3 sm_100a; k1: -DATT_HOOKS -DK1_SEL_H=k1_sel_r2_bf_sm2_c128.h;
# k2: -DATT_HOOKS -DK2_MASK=0x180004). Keeping each binary small matters (null event time must match att_k0's).
# env: ATTD=<run dir with bin/> (required), ATT_REPS (20)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${ATTD:?set ATTD}
B=$D/bin
U=$P/probe/ubench/att
REPS=${ATT_REPS:-20}
CTX="--smem 197632 --cluster 2 --ncal 35000"
declare -A VAR=(
  [att_k0]=null,k0
  [att_k1]=null,k0,r2_bf_sm2_c128
  [att_k2]=null,k0,k2_tma,k2_tma_bf,k2_tma_bf_h
)
ORDER1="att_k0 att_k1 att_k2"
ORDER2="att_k2 att_k1 att_k0"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv,noheader,nounits -lms 100 > $D/clocks_sm.csv &
SAMP=$!
for r in 1 2; do
  R=$D/r$r; mkdir -p $R/raw
  for b in att_k0 att_k0_ns att_k1 att_k1_ns att_k2 att_k2_ns; do cuobjdump -res-usage $B/$b > $R/resusage_$b.txt 2>&1; done
  O=ORDER$r
  for b in ${!O}; do
    for s in "" _ns; do
      echo "round $r: $b$s $(date -u +%T)"
      timeout 400 $B/$b$s --reps $REPS --variants ${VAR[$b]} --out $R --name $b$s $CTX > $R/$b$s.stdout 2> $R/$b$s.stderr
      echo "  rc=$?"
    done
  done
  echo "=== analyze round $r"
  for b in att_k0 att_k1 att_k2; do
    echo "--- $b"
    .venv/bin/python -I $U/att_analyze.py $R $b 2>&1 | tee $R/analyze_$b.txt
    .venv/bin/python -I $U/att_k2_report.py $R $b 2>&1 | grep -v "^\s*$" | head -12 | tee $R/report_$b.txt
  done
done
kill $SAMP 2>/dev/null
echo "D=$D"
