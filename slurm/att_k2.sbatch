#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:15:00
#SBATCH --job-name=pi05-attk2
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/runs/slurm-%j.log
# PI05_ATT_K2 in the CUDA megakernel: correctness on 3 seeds (fixture.band) + interleaved timing / in-situ attention L0 (see probe/mk_attk2_ab.py).
# Extensions are built beforehand (login node) into TORCH_EXTENSIONS_DIR; the job only loads them.
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
PY=.venv/bin/python
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
R=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-mk-attk2
mkdir -p $R
PROBE_PREFIX=968 PROBE_OUT=$R ROUNDS=${ROUNDS:-40} $PY -m megabench.sota.pi05.probe.mk_attk2_ab 2>&1 | tail -90
echo "mk_attk2_ab exit ${PIPESTATUS[0]}"
