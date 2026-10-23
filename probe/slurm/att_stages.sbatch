#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --job-name=p05-att-stages
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# K2 attention in-situ sub-stages on the default build: PI05_FITRACE=3 (clock64) traces from probe/mk_gemm_stages.py (bitwise check, overhead A/B,
# 3 traces), CPU analysis in probe/att_stages_analyze.py. Extensions are built beforehand on the login node.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-att-stages
mkdir -p $R
PROBE_PREFIX=968 MK_VARIANT=k64 PROBE_OUT=$R $PY -m megabench.sota.pi05.probe.mk_gemm_stages
echo "driver exit $? R=$R"
cat $R/records.jsonl
