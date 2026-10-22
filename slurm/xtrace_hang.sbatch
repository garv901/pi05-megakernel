#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:25:00
#SBATCH --job-name=pi05-xt-hang
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# Where a hung spliced launch is stuck (cuda-gdb attached after 40 s: per-warp PC) and where a faulting one faults
# (memcheck). Args: HANG_CUBIN FAULT_CUBIN (paths under runs/xtrace-cpu).
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
X=$PWD/$P/runs/xtrace-cpu
export PROBE_PREFIX=968
G=$X/hang-$SLURM_JOB_ID
mkdir -p $G
echo "=== hang: $1"
LD_PRELOAD=$X/coop_shim.so .venv/bin/python -m megabench.sota.pi05.probe.xtrace.xt_one "$X/$1" > $G/run.out 2>&1 &
pid=$!
sleep 40
if kill -0 $pid 2>/dev/null; then
  timeout 600 cuda-gdb -p $pid -batch -ex "info cuda kernels" -ex "info cuda warps" -ex "info cuda sms" > $G/gdb.out 2>&1
  echo "gdb rc=$? lines=$(wc -l < $G/gdb.out)"
  head -30 $G/gdb.out
  kill -9 $pid
else
  echo "process ended"; cat $G/run.out | tail -5
fi
echo "=== fault: $2"
LD_PRELOAD=$X/coop_shim.so timeout 600 compute-sanitizer --tool memcheck --print-limit 8 \
  .venv/bin/python -m megabench.sota.pi05.probe.xtrace.xt_one "$X/$2" 2>&1 | grep -vE "^\s*$" | cut -c1-300 | head -80
