#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:15:00
#SBATCH --job-name=pi05-xt-mb
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# Xtrace step 3 (MB2 / MB3): single-launch checks of the four spliced builds in runs/mb23-cpu (a hang or fault stays
# in its own process), then xt_mb (bitwise, timing A/B, traces), then memcheck on the spliced base build. CPU analysis:
# xt_mb_analyze.py. OUT dir name: $1 (default: new).
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
PY=.venv/bin/python
D=$PWD/$P/runs/mb23-cpu
SHIM=$PWD/$P/runs/xtrace-cpu/coop_shim.so
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
R=$P/probe/runs/${1:-$(date -u +%Y%m%dT%H%M%SZ)-xt-mb}
mkdir -p $R
echo "=== single-launch checks"
for c in base/xtrace_gpu_gemm rot37/xtrace_gpu_gemm lh1/xtrace_gpu_gemm wpre0/xtrace_gpu_b3-14_gemm; do
  LD_PRELOAD=$SHIM PROBE_PREFIX=968 timeout -k 10 120 \
    $PY -m megabench.sota.pi05.probe.xtrace.xt_one "$D/$c.cubin" 2>&1 | grep -E "XT_ONE|Error" | cut -c1-300
  echo "$c rc=${PIPESTATUS[0]}"
done
echo "=== xt_mb"
LD_PRELOAD=$SHIM XT_DIR=$D PROBE_PREFIX=968 PROBE_OUT=$R timeout -k 10 600 $PY -m megabench.sota.pi05.probe.xtrace.xt_mb
echo "xt_mb exit $?"
du -sh $R
echo "=== memcheck: spliced base (one launch)"
LD_PRELOAD=$SHIM PROBE_PREFIX=968 timeout 240 compute-sanitizer --tool memcheck --print-limit 5 \
  $PY -m megabench.sota.pi05.probe.xtrace.xt_one $D/base/xtrace_gpu_gemm.cubin 2>&1 | grep -vE "^\s*$" | cut -c1-300 | head -30
