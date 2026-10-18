#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=p05-p968-prof
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# P968 profiles of the sota-graph and torch-compile baselines: nsys (graph node trace), Chrome/Perfetto traces,
# ncu one layer (graph-pdl) + first kernels of torch-compile, per-phase table.
# env: RUN_DIR (probe run dir, set by sbatch --export), TUNED_SRC (records.jsonl of the p968-baselines bench run)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
P=megabench.sota.pi05.probe
D=$RUN_DIR
export PROBE_PREFIX=968 PROBE_TUNED=$D/tuned_p968.json PROBE_OUT=$D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu.csv
TUNED_SRC=$(ls -d /raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/*-p968-baselines | tail -1)/records.jsonl
$PY -m $P.tuned $TUNED_SRC $PROBE_TUNED | tee $D/tuned_p968.log
NSYS=/usr/local/cuda/bin/nsys; NCU=/usr/local/cuda/bin/ncu
for arm in graph-pdl graph-nopdl compile; do
  echo "== nsys $arm"
  $NSYS profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda \
    -o $D/nsys_$arm -f true $PY -m $P.drive $arm --reps 3 --flush --range-all 2>&1 | tail -3
  $NSYS stats --report cuda_gpu_trace --format csv --output $D/nsys_${arm}_trace $D/nsys_$arm.nsys-rep 2>&1 | tail -1
  $NSYS export --type sqlite -o $D/nsys_$arm.sqlite --force-overwrite true $D/nsys_$arm.nsys-rep 2>&1 | tail -1
done
echo "== per-phase table"
$PY -m $P.phase_table 968 $D/nsys_graph-pdl_trace_cuda_gpu_trace.csv $D/nsys_graph-nopdl_trace_cuda_gpu_trace.csv > $D/phase_table.txt 2> $D/phase_table.err; tail -3 $D/phase_table.err
echo "== chrome traces"
for arm in graph-pdl graph-nopdl compile; do
  $PY -m $P.drive $arm --reps 1 --flush --chrome-trace $D/trace_$arm.json 2>&1 | tail -2
  gzip -kf $D/trace_$arm.json
done
COMMON=(--clock-control none --profile-from-start off -f)
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__cycles_active.avg,sm__cycles_active.max,sm__cycles_active.min,sm__cycles_elapsed.avg,sm__ctas_launched.sum,lts__t_sector_hit_rate.pct,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed.sum,lts__t_bytes.sum
SECT=(--section LaunchStats --section Occupancy --section WarpStateStats --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats)
echo "== ncu graph-pdl layer 1 (10 kernels 28..37 at P968)"
timeout 420 $NCU "${COMMON[@]}" --launch-skip 28 --launch-count 10 --cache-control all "${SECT[@]}" --metrics $M \
  -o $D/ncu_graph_layer $PY -m $P.drive graph-pdl --reps 1 --warm 1 --flush 2>&1 | tail -4
$NCU --import $D/ncu_graph_layer.ncu-rep --page raw --csv > $D/ncu_graph_layer_raw.csv
echo "== ncu torch-compile first 40 kernels"
timeout 600 $NCU "${COMMON[@]}" --launch-count 40 --cache-control all "${SECT[@]}" --metrics $M \
  -o $D/ncu_compile_first40 $PY -m $P.drive compile --reps 1 --warm 1 --flush 2>&1 | tail -4
$NCU --import $D/ncu_compile_first40.ncu-rep --page raw --csv > $D/ncu_compile_first40_raw.csv
ls -la $D
