#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:25:00
#SBATCH --job-name=p05-ab
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# A+B harness (probe/ubench/ab): k0 / null / null_apw (ab_k0), k1_cl2 + k1_cl2_norope + k1_ks2_c2 (ab_k1c, cta_group::2), k1_ks2_c1 (ab_k1s, cta_group::1),
# stamped + unstamped twins (gate 2), same --ncal 35000 known delay for every binary, two rounds with the binary order reversed, a null in every binary.
# Single-variant binaries (ab_k1a = cl2 only, ab_k1k = ks2_c2 only) check the cross-binary footprint artifact. Binaries are built beforehand on the login node
# (build lines at the top of ab_k0.cu / ab_k1.cu). env: ABD=<run dir with bin/> (required), AB_REPS (30)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${ABD:?set ABD}
B=$D/bin
U=$P/probe/ubench/ab
REPS=${AB_REPS:-30}
declare -A VAR=(
  [ab_k0]=null,null_apw,k0
  [ab_k1c]=null,null_apw,k1_cl2,k1_cl2_norope
  [ab_k1s]=null,null_apw,k1_ks2_c1
  [ab_k1a]=null,k1_cl2
  [ab_k1k]=null,null_apw,k1_ks2_c2
)
ORDER1="ab_k0 ab_k1c ab_k1s ab_k1a ab_k1k"
ORDER2="ab_k1k ab_k1a ab_k1s ab_k1c ab_k0"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv,noheader,nounits -lms 100 > $D/clocks_sm.csv &
SAMP=$!
for r in 1 2; do
  R=$D/r$r; mkdir -p $R/raw
  for b in $ORDER1; do cuobjdump -res-usage $B/$b > $R/resusage_$b.txt 2>&1; cuobjdump -res-usage $B/${b}_ns > $R/resusage_${b}_ns.txt 2>&1; done
  O=ORDER$r
  for b in ${!O}; do
    for s in "" _ns; do
      echo "round $r: $b$s $(date -u +%T)"
      timeout 200 $B/$b$s --reps $REPS --ncal 35000 --variants ${VAR[$b]} --out $R --name $b$s > $R/$b$s.stdout 2> $R/$b$s.stderr
      echo "  rc=$?"
    done
  done
  echo "=== analyze round $r"
  .venv/bin/python -I $U/ab_analyze.py $R ab_k0 ab_k1c ab_k1s ab_k1a ab_k1k 2>&1 | tee $R/analyze.txt
done
kill $SAMP 2>/dev/null
echo "D=$D"
