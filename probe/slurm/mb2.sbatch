#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=p05probe-mb2
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# MB2: A -> C-release window microbenchmark (split-K + barrier vs fused A+B unit); see probe/specs/MB2_abl_unit.md. Results: $D/results.txt
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-mb2
mkdir -p $D
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv | tee $D/gpu.csv
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_100a -lineinfo -Xptxas -v megabench/sota/pi05/probe/ubench/abl_unit.cu -o $D/abl_unit > $D/build.txt 2>&1 || { echo "build failed"; tail $D/build.txt; exit 1; }
grep -E "registers|spill|Compiling" $D/build.txt
B=$D/abl_unit
# r <args>: one run, stdout (JSON / check lines) appended to results.txt together with the args and the return code
r() {
  { echo "# args: $*"; timeout 120 $B "$@" 2>&1; echo "# rc=$? args: $*"; } | tee -a $D/results.txt
}
# defaults of every run: --fresh 1 --pf_kb 4224 --prefill_keys 16 --prefill_ctas 84 --place hi --jitter_ns 0 --smem_kb 132 (P128 k16 context)
P128="--pf_kb 4224 --prefill_keys 16 --prefill_ctas 84"
P968="--pf_kb 5216 --prefill_keys 64 --prefill_ctas 112"

# checks
r --check --variant V0
r --check --variant V0 --abc 1
for la in 0 1 2 4; do r --check --variant V1 --stage reg --la $la; done
r --check --variant V1 --stage cpa
for c in 1 2 4; do r --check --variant H --c $c; done
r --check --variant V0 --ldv 0
r --check --variant V1 --stage reg --la 1 --ldv 0

# A calibration (P128, then P968)
for ctx in "$P128" "$P968"; do
  r --variant ctl $ctx
  r --variant V0 $ctx
  r --variant V0 --abc 1 $ctx
  r --variant V1 --stage reg --la 1 $ctx
  r --variant V0 --ldv 0 $ctx
  r --variant V1 --stage reg --la 1 --ldv 0 $ctx
done

# B staging
for la in 0 1 2 4; do r --variant V1 --stage reg --la $la; done
r --variant V1 --stage cpa
r --variant V1 --stage reg --la 1 --ldv 0
r --variant V1 --stage reg --la 1 --norm 0
r --variant V1 --stage cpa --norm 0

# C contention
for n in 1 8 40 80; do
  r --variant V1 --stage reg --la 1 --active $n
  r --variant V1 --stage cpa --active $n
done

for n in 1 80; do r --variant V1 --stage reg --la 1 --ldv 0 --active $n; done

# D granularity (V0 is in A)
for c in 1 2 4; do r --variant H --c $c; done
r --variant V1 --stage reg --la 1
r --variant V1 --stage cpa

# E jitter
for j in 0 1000 2000; do
  r --variant V0 --jitter_ns $j
  r --variant V1 --stage reg --la 1 --jitter_ns $j
  r --variant H --c 1 --jitter_ns $j
done

# F context
for ctx in "--fresh 0" "--pf_kb 0" "--prefill_keys 0" "--fresh 0 --pf_kb 0 --prefill_keys 0"; do
  r --variant V0 $ctx
  r --variant V1 --stage reg --la 1 $ctx
done
r --variant V1 --stage reg --la 1 --place spread
r --variant V1 --stage reg --la 1 --place spread $P968

# G smem pin
for s in 117 132 160; do r --variant V0 --smem_kb $s; done

# raw stamps for offline plots
r --variant V0 --dump $D/stamps_V0.bin
r --variant V1 --stage reg --la 1 --dump $D/stamps_V1.bin
ls -la $D
