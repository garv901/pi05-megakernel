#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=p05probe-s2
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Stage 2: ncu counters. (a) graph: per-kernel metrics for the first 2 layers (18 kernels), L2 flushed by ncu;
# (b) mk kernel-level metrics (dram bytes, occupancy, stalls); (c) mk PmSampling timeline (+ tstamp for alignment);
# (d) graph whole-chunk PmSampling (--graph-profiling graph) attempt; (e) nsys mk run with tstamp for GPU-metrics alignment.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
P=megabench.sota.pi05.probe
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-stage2
mkdir -p $D; export PROBE_OUT=$D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu.csv
NCU=/usr/local/cuda/bin/ncu; NSYS=/usr/local/cuda/bin/nsys
COMMON=(--clock-control none --profile-from-start off -f)
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__cycles_active.avg,sm__cycles_active.max,sm__cycles_active.min,sm__cycles_elapsed.avg,sm__ctas_launched.sum,lts__t_sector_hit_rate.pct,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed.sum,lts__t_bytes.sum

echo "== (a) graph per-kernel layer metrics (kernels 18..35 = layers 0,1; ncu flushes L2 before each kernel)"
for arm in graph-pdl; do
timeout 420 $NCU "${COMMON[@]}" --launch-skip 18 --launch-count 18 --cache-control all \
  --section LaunchStats --section Occupancy --section WarpStateStats --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats \
  --metrics $M -o $D/ncu_graph_layer $PY -m $P.drive $arm --reps 1 --warm 1 --flush 2>&1 | tail -4
$NCU --import $D/ncu_graph_layer.ncu-rep --page raw --csv > $D/ncu_graph_layer_raw.csv
done

echo "== (b) mk kernel-level metrics"
timeout 900 $NCU "${COMMON[@]}" --kernel-name regex:pi05 --launch-count 1 --cache-control all \
  --section LaunchStats --section Occupancy --section WarpStateStats --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats \
  --metrics $M -o $D/ncu_mk_kernel $PY -m $P.drive mk --reps 1 --warm 1 --flush 2>&1 | tail -4
$NCU --import $D/ncu_mk_kernel.ncu-rep --page raw --csv > $D/ncu_mk_kernel_raw.csv

echo "== (c) mk PmSampling (+tstamp)"
timeout 900 $NCU "${COMMON[@]}" --kernel-name regex:pi05 --launch-count 1 --cache-control all --section PmSampling --section PmSampling_WarpStates \
  -o $D/ncu_mk_pm $PY -m $P.drive mk --reps 1 --warm 1 --flush --tstamp --save-tstamp $D/ncu_mk_pm_tstamp.npy 2>&1 | tail -4
$NCU --import $D/ncu_mk_pm.ncu-rep --page raw --csv --print-metric-instances values > $D/ncu_mk_pm_raw.csv

echo "== (d) graph whole-chunk PmSampling"
timeout 480 $NCU "${COMMON[@]}" --graph-profiling graph --launch-count 1 --section PmSampling --section PmSampling_WarpStates \
  -o $D/ncu_graph_pm $PY -m $P.drive graph-pdl --reps 1 --warm 1 --flush 2>&1 | tail -6
$NCU --import $D/ncu_graph_pm.ncu-rep --page details 2>&1 | head -30 > $D/ncu_graph_pm_details.txt
$NCU --import $D/ncu_graph_pm.ncu-rep --page raw --csv --print-metric-instances values > $D/ncu_graph_pm_raw.csv

echo "== (e) nsys mk with tstamp"
$NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=100000 \
  -o $D/nsys_mk_ts -f true $PY -m $P.drive mk --reps 8 --flush --range-all --tstamp --save-tstamp $D/nsys_mk_ts_tstamp.npy 2>&1 | tail -2
$NSYS export --type sqlite -o $D/nsys_mk_ts.sqlite --force-overwrite true $D/nsys_mk_ts.nsys-rep 2>&1 | tail -1
ls -la $D
