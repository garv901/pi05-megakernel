#!/bin/bash
#SBATCH --partition=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:30:00
#SBATCH --job-name=pi05-gbstamp
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Block-0 barrier stamps of the plain binary (no PTRACE), P968 k64, G batching 0 / 1 / 2 interleaved.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-gbstamps
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem --format=csv
export PROBE_PREFIX=968
export PROBE_OUT=$R MK_VARIANTS=k64,k64-gb1,k64-gb2; mkdir -p $R
$PY -m megabench.sota.pi05.probe.mk_stamps_ab 2>&1 | tail -6
ls -la $R/*
