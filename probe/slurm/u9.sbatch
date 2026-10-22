#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:06:00
#SBATCH --job-name=p05-u9
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# U9 (ubench/gemm_task.cu): gemm_phase warp-task latency vs k-depth, back-to-back tasks, stream-K red.add fixup.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-u9
mkdir -p $D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv | tee $D/gpu.csv
U=megabench/sota/pi05/probe/ubench
timeout 400 $U/gemm_task 100 5 > $D/u9_gemm_task.txt 2>&1; echo "u9 rc=$?"
cat $D/u9_gemm_task.txt
