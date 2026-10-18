#!/bin/bash
# Stage 5b: isolated per-kernel ncu of the graph arm (kernel replay, node mode), full sections, cold vs warm L2.
# kernel index in the 1638-kernel chunk: 17 prologue kernels, then layer l of step s starts at 17 + 162*s + 9*l (A qkv .. I resid2); 1637 = final boundary.
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python; P=megabench.sota.pi05.probe
D=${D:-megabench/sota/pi05/probe/runs/ncu_phase}; mkdir -p $D
NCU=/usr/local/cuda/bin/ncu
SECT="--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis"
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__cycles_active.avg,sm__cycles_elapsed.avg,sm__ctas_launched.sum,lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed.sum,sm__inst_executed.avg.per_cycle_active,sm__issue_active.avg.pct_of_peak_sustained_active,lts__t_bytes.sum,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks,launch__occupancy_limit_warps,sm__maximum_warps_per_active_cycle_pct
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee -a $D/gpu_clocks_graph_before.csv
run() { # tag skip count cache
  CMD="$NCU --clock-control none --profile-from-start off -f --launch-skip $2 --launch-count $3 --cache-control $4 $SECT --metrics $M -o $D/graph_$1 $PY -m $P.drive graph-pdl --reps 1 --warm 1 --flush"
  echo "$CMD" > $D/graph_$1.cmd; echo "== $1"; eval "$CMD" 2>&1 | grep -i "error\|Report\|fail"
  $NCU --import $D/graph_$1.ncu-rep --page raw --csv > $D/graph_$1_raw.csv
}
run layer8_step5_l2cold 899 9 all
run layer8_step5_l2warm 899 9 none
run layer0_step0_l2cold 17 9 all
run layer0_step0_l2warm 17 9 none
run layer17_step9_l2cold 1628 10 all
run layer17_step9_l2warm 1628 10 none
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee -a $D/gpu_clocks_graph_after.csv
