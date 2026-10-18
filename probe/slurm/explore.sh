#!/bin/bash
# exploration of ncu PM sampling pass counts (run inside an salloc allocation via srun)
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python; P=megabench.sota.pi05.probe
D=megabench/sota/pi05/probe/runs/explore; mkdir -p $D
NCU=/usr/local/cuda/bin/ncu
COMMON=(--clock-control none --profile-from-start off -f --kernel-name regex:pi05 --launch-count 1)
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv
t0=$(date +%s)
echo "== A: app replay, section PmSampling, 500ns, no warp sampling"
$NCU "${COMMON[@]}" --replay-mode application --cache-control all --section PmSampling --disable-pm-warp-sampling \
  --pm-sampling-interval 500 -o $D/A $PY -m $P.drive mk --reps 1 --warm 1 --flush --tstamp --save-tstamp $D/A_ts_{ts}.npy 2>&1 | tail -15
echo "A took $(( $(date +%s)-t0 )) s"; ls -la $D
t0=$(date +%s)
echo "== B: app replay, warp states section, 500ns"
$NCU "${COMMON[@]}" --replay-mode application --cache-control all --section PmSampling_WarpStates \
  --pm-sampling-interval 500 -o $D/B $PY -m $P.drive mk --reps 1 --warm 1 --flush --tstamp --save-tstamp $D/B_ts_{ts}.npy 2>&1 | tail -15
echo "B took $(( $(date +%s)-t0 )) s"; ls -la $D
