#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --job-name=p05-u10
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# U10 (ubench/gemm_tc.cu): tcgen05 GEMM task vs gemm_phase's mma.sync task; same-job U9 (gemm_task) rerun for comparison.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-u10
mkdir -p $D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv | tee $D/gpu.csv
U=megabench/sota/pi05/probe/ubench
timeout 200 $U/gemm_tc 100 5 > $D/u10_gemm_tc.txt 2>&1; echo "u10 rc=$?"
timeout 200 $U/gemm_task 100 5 > $D/u9_gemm_task.txt 2>&1; echo "u9 rc=$?"
timeout 200 $U/gemm_tc 100 5 > $D/u10_gemm_tc_rerun.txt 2>&1; echo "u10 rerun rc=$?"
cat $D/u10_gemm_tc.txt; echo ---- U9 same job; cat $D/u9_gemm_task.txt; echo ---- U10 rerun; grep '"cfg"' $D/u10_gemm_tc_rerun.txt
