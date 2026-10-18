#!/bin/bash
# Stage 5c (ncu half): (1) ncu whole-graph PM sampling (--graph-profiling graph) for graph-pdl and graph-nopdl at the minimum interval;
#           (2) nsys GPU-metrics at the maximum frequency (200 kHz) for graph-pdl, graph-nopdl (8 reps) and mk (1 rep, own tstamp, x3),
#               plus the gh100-ct (compute triage) metric set for graph-pdl and mk.
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python; P=megabench.sota.pi05.probe
D=${D:-megabench/sota/pi05/probe/runs/ncu_phase}; mkdir -p $D
NCU=/usr/local/cuda/bin/ncu; NSYS=/usr/local/cuda/bin/nsys
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee -a $D/gpu_clocks_pm_nsys_before.csv
for arm in graph-pdl graph-nopdl; do
  CMD="$NCU --clock-control none --profile-from-start off -f --graph-profiling graph --launch-count 1 --section PmSampling --pm-sampling-interval 1000 -o $D/graph_wholegraph_pm_$arm $PY -m $P.drive $arm --reps 1 --warm 1 --flush"
  echo "$CMD" > $D/graph_wholegraph_pm_$arm.cmd; echo "== ncu $arm"; eval "$CMD" 2>&1 | grep -i "error\|Report\|fail"
done
