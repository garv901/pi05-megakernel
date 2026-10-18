#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --job-name=p05probe-s3
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Stage 3: microbenchmarks U1 (HBM ramp), U3 (sync cost), U4 (graph chain/PDL) + dense nsys folds for graph arms.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
P=megabench.sota.pi05.probe
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-stage3
mkdir -p $D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv | tee $D/gpu.csv
U=megabench/sota/pi05/probe/ubench
timeout 420 $U/chain_pdl 600 > $D/u4_chain_pdl.txt 2>&1; echo "u4 rc=$?"
timeout 300 $U/sync_cost 200 11 > $D/u3_sync_cost.txt 2>&1; echo "u3 rc=$?"
timeout 480 $U/hbm_ramp 15 > $D/u1_hbm_ramp.txt 2>&1; echo "u1 rc=$?"
NSYS=/usr/local/cuda/bin/nsys
for arm in graph-pdl graph-nopdl; do
  $NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=100000 \
    -o $D/nsys_$arm -f true $PY -m $P.drive $arm --reps 8 --flush --range-all 2>&1 | tail -2
  $NSYS stats --report cuda_gpu_trace --format csv --output $D/nsys_${arm}_trace $D/nsys_$arm.nsys-rep 2>&1 | tail -1
  $NSYS export --type sqlite -o $D/nsys_$arm.sqlite --force-overwrite true $D/nsys_$arm.nsys-rep 2>&1 | tail -1
done
ls -la $D
