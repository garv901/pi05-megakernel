#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --job-name=p05-u11
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# U11 (ubench/gemm_tc2.cu): clock64 stage timeline of U10's tcgen05 GEMM task; same-job U10 rerun for the perturbation check.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-u11
mkdir -p $D/raw
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv | tee $D/gpu.csv
U=megabench/sota/pi05/probe/ubench
timeout 240 $U/gemm_tc2 100 5 $D/raw > $D/u11.txt 2>&1; echo "u11 rc=$?"
timeout 200 $U/gemm_tc 100 5 > $D/u10_same_job.txt 2>&1; echo "u10 rc=$?"
echo "D=$D"
cat $D/u11.txt; echo ---- U10 same job; grep '"cfg"' $D/u10_same_job.txt
