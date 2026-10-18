#!/bin/bash
# Stage 5d: nsys GPU-metrics at the maximum frequency (200 kHz) for graph-pdl, graph-nopdl (8 reps) and mk (1 rep, own tstamp, x3),
#               plus the gh100-ct (compute triage) metric set for graph-pdl and mk.
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python; P=megabench.sota.pi05.probe
D=${D:-megabench/sota/pi05/probe/runs/ncu_phase}; mkdir -p $D
NCU=/usr/local/cuda/bin/ncu; NSYS=/usr/local/cuda/bin/nsys
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee -a $D/gpu_clocks_nsys_before.csv
nsys_run() { # tag arm set extra-args
  CMD="$NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=200000 -o $D/nsys_$1 -f true $PY -m $P.drive $2 --flush --range-all $4"
  echo "$CMD" > $D/nsys_$1.cmd; echo "== nsys $1"; eval "$CMD" 2>&1 | tail -3
  $NSYS stats --report cuda_gpu_trace --format csv --output $D/nsys_${1}_trace $D/nsys_$1.nsys-rep 2>&1 | tail -1
  $NSYS export --type sqlite -o $D/nsys_$1.sqlite --force-overwrite true $D/nsys_$1.nsys-rep 2>&1 | tail -1
}
nsys_run graph-pdl_gb10x graph-pdl gb10x "--reps 8"
nsys_run graph-nopdl_gb10x graph-nopdl gb10x "--reps 8"
for r in 1 2 3; do
  nsys_run mk_gb10x_r$r mk gb10x "--reps 1 --tstamp --save-tstamp $D/nsys_mk_gb10x_r${r}_tstamp.npy"
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee -a $D/gpu_clocks_nsys_after.csv
ls -la $D | tail -40
