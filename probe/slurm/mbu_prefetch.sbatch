#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --job-name=p05-mbu
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# MBU / prefetch accounting: (1) traced-vs-plain overhead + non-ncu PTRACE/PFTRACE capture, (2) ncu PM sampling (application replay, 3 reps)
# of the megakernel built with -DPI05_PTRACE -DPI05_PFTRACE so the per-CTA prefetch-issue stamps share a run with each PM pass.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python; P=megabench.sota.pi05.probe
D=${D:-megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-mbu}
mkdir -p $D; export PROBE_OUT=$D
echo "run dir $D"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_clocks_before.csv
$PY -m $P.mk_pftrace 2>&1 | tail -6 | tee $D/mk_pftrace.log
TS=2212256
NCU=/usr/local/cuda/bin/ncu
for r in 1 2 3; do
  CMD="$NCU --clock-control none --profile-from-start off -f --kernel-name regex:pi05 --launch-count 1 --replay-mode application --cache-control all --section PmSampling --pm-sampling-interval 1000 -o $D/mk_pm_pf_r$r $PY -m $P.drive mk --reps 1 --warm 1 --flush --tstamp --tstamp-size $TS --mk-flags '-DPI05_PTRACE -DPI05_PFTRACE' --save-tstamp $D/mk_pm_pf_r${r}_ts_{ts}.npy"
  echo "$CMD" > $D/mk_pm_pf_r$r.cmd; echo "== r$r"; eval "$CMD" 2>&1 | grep -i "error\|Report\|fail"
done
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_clocks_after.csv
ls -la $D
