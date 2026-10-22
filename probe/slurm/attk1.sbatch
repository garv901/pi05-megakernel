#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:40:00
#SBATCH --job-name=p05-attk1
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Attention (C + C3) study, K1 (tcgen05 cta_group::2 + FA4-style): per-variant binaries (att_k1_<tag>, _ns, _mk; built by
# probe/ubench/att/build_k1.sh <tag> <variant>) each holding null + k0 + ONE K1 variant. A binary with many K1 variants has a ~12% slower null baseline
# (kernel text size), which shifts every L0 delta; one-variant binaries keep null at the att_k0 level (8.0 ms / 180 layers).
# Per tag: stamped + unstamped twins, forward and reversed variant order (ABAB round robin), then the ATT_MARKS timeline binary.
# Context: FG_UMMA launch (197632 B smem, cluster 2), --ncal 35000.
# env: ATTD (run dir), ATT_REPS (20), ATT_TAGS (space separated; tag == variant name), ATT_ARGS
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${ATTD:-$P/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-attk1}
mkdir -p $D/raw
U=$P/probe/ubench/att
REPS=${ATT_REPS:-20}
TAGS=${ATT_TAGS:-"u1_c1 r2_bf r2_bf_c128 r2_bf_sm2 r2_bf_sm2_c128 r2_bf_sm2_poly r2_bf_c2 r2_bf_c128_strm"}
UMMA="--smem 197632 --cluster 2"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv,noheader,nounits -lms 100 > $D/clocks_sm.csv &
SAMP=$!
for T in $TAGS; do
  B=att_k1_$T
  for b in $B ${B}_ns ${B}_mk; do cuobjdump -res-usage $U/$b > $D/resusage_$b.txt 2>&1; done
  for cfg in "$B:null,k0,$T" "${B}rev:$T,k0,null"; do
    n=${cfg%%:*}; vl=${cfg#*:}
    cp $D/resusage_$B.txt $D/resusage_$n.txt; cp $D/resusage_${B}_ns.txt $D/resusage_${n}_ns.txt
    echo "=== $n stamped";   timeout 300 $U/$B --reps $REPS --ncal 35000 --variants $vl --out $D --name $n $UMMA ${ATT_ARGS:-} > $D/$n.stdout 2> $D/$n.stderr; echo "rc=$?"
    echo "=== $n unstamped"; timeout 300 $U/${B}_ns --reps $REPS --ncal 35000 --variants $vl --out $D --name ${n}_ns $UMMA ${ATT_ARGS:-} > $D/${n}_ns.stdout 2> $D/${n}_ns.stderr; echo "rc=$?"
  done
  echo "=== $T timeline"; timeout 200 $U/${B}_mk --reps 4 --ncal 35000 --variants null,k0,$T --out $D --name ${B}_mk $UMMA > $D/${B}_mk.stdout 2> $D/${B}_mk.stderr; echo "rc=$?"
done
kill $SAMP 2>/dev/null
echo "=== analyze"
for T in $TAGS; do .venv/bin/python -I $U/att_k1_analyze.py $D att_k1_$T att_k1_${T}rev 2>&1 | tee -a $D/analyze.txt; done
echo "D=$D"
