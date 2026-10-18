#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=p05probe-u7
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# U7: fused attention + o_proj (F1/F2) vs ctl; see probe/specs/U7_attn_o_fused.md. Results: $D/results.txt
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-u7
mkdir -p $D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv | tee $D/gpu.csv
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_100a -lineinfo -Xptxas -v megabench/sota/pi05/probe/ubench/attn_o_fused.cu -o $D/attn_o_fused > $D/build.txt 2>&1 || { echo "build failed"; tail $D/build.txt; exit 1; }
grep -E "registers|spill|Compiling" $D/build.txt
B=$D/attn_o_fused
r() { timeout 120 $B "$@" 2>&1 | tee -a $D/results.txt; echo "rc=${PIPESTATUS[0]} args: $*"; }
r --check --variant F1 --kb 32
r --check --variant F1 --kb 64
r --check --variant F2
r --variant ctl --solo 0
for kb in 32 64; do for pf in 0 1; do for solo in 0 1; do r --variant F1 --kb $kb --pf $pf --solo $solo; done; done; done
for pf in 0 1; do for solo in 0 1; do r --variant F2 --pf $pf --solo $solo; done; done
r --variant F1 --kb 64 --pf 1 --l2p 0
r --variant F2 --pf 1 --l2p 0
ls -la $D
