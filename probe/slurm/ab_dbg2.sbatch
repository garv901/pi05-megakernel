#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --job-name=p05-abdbg2
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# PI05_AB_UMMA crash isolation: truncated shapes, each in its own process; then compute-sanitizer on the smallest failing shape.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
export PROBE_PREFIX=968
source megabench/sota/slurm/env.sh
for sh in "0 18" "1 1" "1 2" "1 18" "2 18"; do
  set -- $sh
  echo "=== NS=$1 NL=$2"
  NS=$1 NL=$2 timeout -s KILL 120 .venv/bin/python -m megabench.sota.pi05.probe.mk_ab_dbg 2>&1 | grep -v "^\[" | tail -6 | cut -c1-300
done
echo "=== sanitizer NS=1 NL=1"
NS=1 NL=1 timeout -s KILL 400 /usr/local/cuda/bin/compute-sanitizer --tool memcheck .venv/bin/python -m megabench.sota.pi05.probe.mk_ab_dbg 2>&1 | head -60 | cut -c1-300
