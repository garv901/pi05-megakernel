#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:15:00
#SBATCH --job-name=p05-fgconf
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# F+G confirmation job: K0, K2 best, K1 best (no cluster) and K1 best (cta_group::2 pair) in one job under one gate standard:
# the same --ncal 35000 known delay for every binary, all 7 gates per launch, stamped + unstamped twins (gate 2), null in every binary.
# Two rounds; round 2 runs the binaries in reverse order so binary order / thermal drift shows up as a round-to-round difference.
# Binaries: built from current sources into FGD/bin before submission (build lines in each .cu header).
# env: FGD=<run dir with bin/> (required), FG_REPS (30)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${FGD:?set FGD}
B=$D/bin
U=$P/probe/ubench/fg
REPS=${FG_REPS:-30}
declare -A VAR=(
  [fg_k0]=null,k0
  [fg_k2]=null,k2_m4k2_s128_d4_nf
  [fg_k1]=null,k1_nf_e8_a51_fast_ts_s32_d2
  [fg_k1c]=null,cl2_s128_d1_fast_ts
)
ORDER1="fg_k0 fg_k2 fg_k1 fg_k1c"
ORDER2="fg_k1c fg_k1 fg_k2 fg_k0"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv,noheader,nounits -lms 100 > $D/clocks_sm.csv &
SAMP=$!
for r in 1 2; do
  R=$D/r$r; mkdir -p $R/raw
  for b in fg_k0 fg_k0_ns fg_k2 fg_k2_ns fg_k1 fg_k1_ns fg_k1c fg_k1c_ns; do cuobjdump -res-usage $B/$b > $R/resusage_$b.txt 2>&1; done
  O=ORDER$r
  for b in ${!O}; do
    for s in "" _ns; do
      echo "round $r: $b$s $(date -u +%T)"
      timeout 300 $B/$b$s --reps $REPS --ncal 35000 --variants ${VAR[$b]} --out $R --name $b$s > $R/$b$s.stdout 2> $R/$b$s.stderr
      echo "  rc=$?"
    done
  done
  echo "=== analyze round $r"
  .venv/bin/python -I $U/fg_analyze.py $R fg_k0 fg_k2 fg_k1 fg_k1c 2>&1 | tee $R/analyze.txt
done
kill $SAMP 2>/dev/null
echo "D=$D"
