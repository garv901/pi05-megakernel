#!/bin/bash
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python; P=megabench.sota.pi05.probe
D=megabench/sota/pi05/probe/runs/explore; mkdir -p $D
NCU=/usr/local/cuda/bin/ncu
COMMON=(--clock-control none --profile-from-start off -f --kernel-name regex:pi05 --launch-count 1)
for iv in 250 100; do
echo "== C: metrics subset interval $iv"
$NCU "${COMMON[@]}" --replay-mode application --cache-control all --pm-sampling-interval $iv \
  --metrics pmsampling:dram__read_throughput.avg.pct_of_peak_sustained_elapsed,pmsampling:sm__cycles_active.avg.per_cycle_elapsed,pmsampling:sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed \
  -o $D/C$iv $PY -m $P.drive mk --reps 1 --warm 1 --flush --tstamp --save-tstamp $D/C${iv}_ts_{ts}.npy 2>&1 | grep -i "err\|warn\|Report" | head
done
