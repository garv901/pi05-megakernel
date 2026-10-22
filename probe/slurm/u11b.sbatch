#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:08:00
#SBATCH --job-name=p05-u11b
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# U11b: U9 (gemm_task) with clock64 durations + U11 rerun with warp_period; both clock-based, same job.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-u11b
mkdir -p $D/raw
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv | tee $D/gpu.csv
U=megabench/sota/pi05/probe/ubench
timeout 240 $U/gemm_tc2 100 5 $D/raw > $D/u11.txt 2>&1; echo "u11 rc=$?"
timeout 200 $U/gemm_task 100 5 > $D/u9_clk.txt 2>&1; echo "u9 rc=$?"
echo "D=$D"
cat $D/u11.txt; echo ---- U9 clock; cat $D/u9_clk.txt
