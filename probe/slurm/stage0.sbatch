#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=p05probe-s0
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Stage 0: baseline timings + tstamp reader + nsys / ncu feasibility.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
P=megabench.sota.pi05.probe
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-stage0
mkdir -p $D; export PROBE_OUT=$D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu.csv
$PY -m $P.stage0 2>&1 | tail -40 | tee $D/stage0.log
NSYS=/usr/local/cuda/bin/nsys; NCU=/usr/local/cuda/bin/ncu
echo "== nsys graph-pdl (graph node trace)"
$NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda -o $D/nsys_graph_pdl -f true $PY -m $P.drive graph-pdl --reps 3 --flush 2>&1 | tail -5
echo "== nsys mk"
$NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda -o $D/nsys_mk -f true $PY -m $P.drive mk --reps 3 --flush 2>&1 | tail -5
echo "== nsys gpu metrics (sample rate probe)"
$NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda --gpu-metrics-devices=0 --gpu-metrics-frequency=100000 -o $D/nsys_gm_graph -f true $PY -m $P.drive graph-pdl --reps 3 --flush 2>&1 | tail -8
$NSYS stats --report cuda_gpu_trace --format csv --output $D/nsys_graph_pdl_trace $D/nsys_graph_pdl.nsys-rep 2>&1 | tail -2
echo "== ncu graph-profiling graph, PmSampling, whole chunk"
timeout 900 $NCU --clock-control none --profile-from-start off --graph-profiling graph --launch-count 1 --section PmSampling --pm-sampling-interval 2000 -o $D/ncu_graph_pm -f $PY -m $P.drive graph-pdl --reps 1 --warm 1 --flush 2>&1 | tail -12
echo "== ncu mk PmSampling"
timeout 900 $NCU --clock-control none --profile-from-start off --kernel-name regex:pi05 --launch-count 1 --section PmSampling --section PmSampling_WarpStates --pm-sampling-interval 500 -o $D/ncu_mk_pm -f $PY -m $P.drive mk --reps 1 --warm 1 --flush 2>&1 | tail -12
ls -la $D
