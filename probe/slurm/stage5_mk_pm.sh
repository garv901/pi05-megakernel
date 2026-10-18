#!/bin/bash
# Stage 5a: mk PM sampling, application replay so that every ncu pass is a real run with its own tstamp.
# One ncu invocation = one PM-Triage pass + one PM warp-sampling pass (all stall reasons) ; repeated 3 times.
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python; P=megabench.sota.pi05.probe
D=${D:-megabench/sota/pi05/probe/runs/ncu_phase}; mkdir -p $D
NCU=/usr/local/cuda/bin/ncu
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee -a $D/gpu_clocks_before.csv
for r in 1 2 3; do
  CMD="$NCU --clock-control none --profile-from-start off -f --kernel-name regex:pi05 --launch-count 1 --replay-mode application --cache-control all --section PmSampling --pm-sampling-interval 1000 -o $D/mk_pm_triage_warp_r$r $PY -m $P.drive mk --reps 1 --warm 1 --flush --tstamp --save-tstamp $D/mk_pm_triage_warp_r${r}_ts_{ts}.npy"
  echo "$CMD" > $D/mk_pm_triage_warp_r$r.cmd; echo "== r$r"; eval "$CMD" 2>&1 | grep -i "error\|Report\|fail"
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee -a $D/gpu_clocks_after.csv
