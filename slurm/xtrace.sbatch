#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-xtrace
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# Xtrace (gwatch) steps 1 and 2. Step 1: CUDA k64 with its kernel swapped for offline-spliced cubins through the
# LD_PRELOAD shim (bitwise, overhead A/B, traces). Step 2: CuTe k64 (and CUDA as a cross-check) through gwatch's own
# CUPTI runtime, spliced in-process. The CPU decode / analysis (xt_analyze.py) runs off the GPU node.
# Arg: 1 = single-launch checks + step 1 + memcheck, 2 = step 2, all (default) = both. OUT dir name: $2 (default: new).
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
PY=.venv/bin/python
XPY=/raid/garv901/xtrace-venv/bin/python
NV=/raid/garv901/xtrace-venv/lib/python3.12/site-packages/nvidia
XLD=$NV/cuda_runtime/lib:$NV/cuda_cupti/lib
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
STAGE=${1:-all}
R=$P/probe/runs/${2:-$(date -u +%Y%m%dT%H%M%SZ)-xtrace}
mkdir -p $R
if [ "$STAGE" != 2 ]; then
echo "=== single-launch checks"
for c in "bisect3/xtrace_gpu_b10a,b11a,b11e" "bisect3/xtrace_gpu_b10a,b12a,b12e" bisect3/xtrace_gpu_b10-12 \
         cuda/xtrace_gpu cuda/xtrace_anchor; do
  LD_PRELOAD=$PWD/$P/runs/xtrace-cpu/coop_shim.so PROBE_PREFIX=968 timeout -k 10 120 \
    $PY -m megabench.sota.pi05.probe.xtrace.xt_one "$PWD/$P/runs/xtrace-cpu/$c.cubin" 2>&1 | grep -E "XT_ONE|Error" | cut -c1-400
  echo "$c rc=${PIPESTATUS[0]}"
done
echo "=== step 1: xt_cuda (shim)"
LD_PRELOAD=$PWD/$P/runs/xtrace-cpu/coop_shim.so XT_DIR=$PWD/$P/runs/xtrace-cpu PROBE_PREFIX=968 PROBE_OUT=$R \
  timeout -k 10 900 $PY -m megabench.sota.pi05.probe.xtrace.xt_cuda
echo "xt_cuda exit $?"
fi
if [ "$STAGE" != 1 ]; then
for impl in cute cuda; do
  echo "=== step 2: xt_gwatch $impl"
  LD_LIBRARY_PATH=$XLD XT_IMPL=$impl PROBE_PREFIX=968 PROBE_OUT=$R timeout -k 10 1200 $XPY -m megabench.sota.pi05.probe.xtrace.xt_gwatch
  echo "xt_gwatch $impl exit $?"
done
fi
ls -la $R
[ "$STAGE" = 2 ] && exit 0
echo "=== memcheck: full CUDA gpu splice (one launch)"
LD_PRELOAD=$PWD/$P/runs/xtrace-cpu/coop_shim.so PROBE_PREFIX=968 timeout 600 compute-sanitizer --tool memcheck \
  --print-limit 5 $PY -m megabench.sota.pi05.probe.xtrace.xt_one $PWD/$P/runs/xtrace-cpu/cuda/xtrace_gpu.cubin 2>&1 \
  | grep -vE "^\s*$" | cut -c1-300 | head -40
