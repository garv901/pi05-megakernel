#!/bin/bash
# Stage 5e: two more repeats of the whole-graph PM run for graph-pdl (repeatability)
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python; P=megabench.sota.pi05.probe
D=${D:-megabench/sota/pi05/probe/runs/ncu_phase}; NCU=/usr/local/cuda/bin/ncu
for r in 2 3; do
  CMD="$NCU --clock-control none --profile-from-start off -f --graph-profiling graph --launch-count 1 --section PmSampling --pm-sampling-interval 1000 -o $D/graph_wholegraph_pm_graph-pdl_r$r $PY -m $P.drive graph-pdl --reps 1 --warm 1 --flush"
  echo "$CMD" > $D/graph_wholegraph_pm_graph-pdl_r$r.cmd; echo "== r$r"; eval "$CMD" 2>&1 | grep -i "error\|Report\|fail"
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee -a $D/gpu_clocks_pm_rep_after.csv
