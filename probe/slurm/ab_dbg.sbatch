#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:10:00
#SBATCH --job-name=p05-abdbg
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# A+B harness debug: each binary / variant run separately with a hard timeout, short reps. env: ABD=<dir with bins>
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=${ABD:?}; O=${ABO:-/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/abdbg}; mkdir -p $O
for spec in "ab_k0:null" "ab_k0:null,null_apw,k0" "ab_k1c:null" "ab_k1c:null,null_apw" "ab_k1c:null,k1_cl2" "ab_k1c:null,k1_cl2_norope" "ab_k1c:null,k1_ks2_c2" "ab_k1s:null,k1_ks2_c1"; do
  b=${spec%%:*}; v=${spec#*:}
  echo "=== $b $v $(date -u +%T)"
  timeout -s KILL 60 $D/$b --reps 2 --ncal 35000 --variants $v --out $O 2>&1 | grep -v '"gate":"\(closure\|ordering\|clock_stable\)"' | cut -c1-400
  echo "rc=${PIPESTATUS[0]}"
done
