#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --job-name=p05-attk2
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Attention K2 (HMMA + TMA / DSMEM / balanced decomposition) vs K0 in the FG_UMMA launch context (197632 B smem, cluster 2). Binaries att_k2, att_k2_ns,
# att_k2_mk built beforehand (probe/ubench/att/build.sh att_k2 -DATT_HOOKS).
# env: ATTD (run dir), ATT_REPS (20), K2_VARS (variant list), ATT_BIN (att_k2)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${ATTD:-$P/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-attk2}
mkdir -p $D/raw
U=$P/probe/ubench/att
B=${ATT_BIN:-att_k2}
REPS=${ATT_REPS:-20}
VARS=${K2_VARS:-null,k0,k2_cp,k2_cp_bulk,k2_tma,k2_tma_bulk,k2_tma_bulk_nopre,calib}
UMMA="--smem 197632 --cluster 2"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
for b in $B ${B}_ns ${B}_mk; do cuobjdump -res-usage $U/$b > $D/resusage_$b.txt 2>&1; done
n=${B}_umma
for b in $B ${B}_ns ${B}_mk; do cp $D/resusage_$b.txt $D/resusage_${b/$B/$n}.txt; done
echo "=== stamped"
timeout 500 $U/$B --reps $REPS --variants $VARS --out $D --name $n $UMMA > $D/$n.stdout 2> $D/$n.stderr; echo "$n rc=$?"
echo "=== unstamped"
timeout 500 $U/${B}_ns --reps $REPS --variants $VARS --out $D --name ${n}_ns $UMMA > $D/${n}_ns.stdout 2> $D/${n}_ns.stderr; echo "${n}_ns rc=$?"
echo "=== marks"
timeout 300 $U/${B}_mk --reps 6 --variants $VARS --out $D --name ${n}_mk $UMMA > $D/${n}_mk.stdout 2> $D/${n}_mk.stderr; echo "${n}_mk rc=$?"
echo "=== cross-binary k0 check (att_k0 binaries, same context)"
for b in att_k0 att_k0_ns; do timeout 300 $U/$b --reps $REPS --variants null,k0 --out $D --name ${b}_xchk --smem 197632 --cluster 2 > $D/${b}_xchk.stdout 2> $D/${b}_xchk.stderr; done
echo "=== analyze"
.venv/bin/python -I $U/att_analyze.py $D $n 2>&1 | tee $D/analyze.txt
.venv/bin/python -I $U/att_k2_report.py $D $n 2>&1 | tee $D/report.txt
for f in $D/att_k0_xchk.jsonl $D/att_k0_ns_xchk.jsonl; do grep -h '"rec":"summary"' $f | cut -c1-200; done
echo "D=$D"
