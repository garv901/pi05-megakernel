#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=p05-u12
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# U12 (plan Step 0): TMA ingress / cluster multicast / cooperative+cluster launch (a, coop), tcgen05.ld throughput + U11 anomaly (b),
# M=64 N=56 MMA chain with TMA ring (c, msweep), A-from-TMEM (d). Whole binary run twice (rep1, rep2) for reproducibility.
# Build (login node): cd probe/ubench/fg && nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a u12.cu -o u12 -lcuda
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
D=megabench/sota/pi05/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-u12
mkdir -p $D
U=megabench/sota/pi05/probe/ubench/fg
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw --format=csv | tee $D/gpu_before.csv
# SM clock sampled DURING the run (the idle snapshot above reads ~120 MHz)
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 100 > $D/clocks_during.csv &
SMI=$!
for rep in 1 2; do
  for sec in coop a b c d; do
    t0=$(date +%s)
    timeout 900 $U/u12 $sec > $D/rep${rep}_$sec.jsonl 2> $D/rep${rep}_$sec.err; echo "rep$rep $sec rc=$? $(( $(date +%s) - t0 )) s"
  done
done
kill $SMI
echo "D=$D"
# summary of the sampled SM clock while the job ran (rows with clocks.sm > 1000 MHz = busy)
awk -F', ' 'NR>1 { v=$2+0; if (v>1000) { n++; s+=v; if (min==""||v<min) min=v; if (v>max) max=v } } END { printf "SM clock during run (samples>1000MHz): n=%d mean=%.0f min=%d max=%d MHz\n", n, s/n, min, max }' $D/clocks_during.csv | tee $D/clocks_summary.txt
grep -h '"xchk":"FLAG"' $D/rep*_*.jsonl | wc -l | xargs echo "FLAGged xchk lines:"
