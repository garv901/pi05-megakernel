#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --time=00:20:00
#SBATCH --job-name=p05-att
#SBATCH --output=/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/slurm-%j.log
# Attention (C + C3) study, step 1: validity harness (null + calib + K0, stamped and unstamped binaries, ABAB) in two contexts and the megakernel
# vs NULL_ATT L0 differential. Contexts: "att_k0" = plain build smem (117248 B, no cluster); "att_k0_umma" = the FG_UMMA build's launch (197632 B, cluster
# of 2), the context of the in-situ ground truth. Binaries (att_k0, _ns, _mk; build.sh in probe/ubench/att) and the four torch extensions (umma, umma +
# NULL_ATT, default, default + NULL_ATT) are built beforehand on the login node.
# env: ATTD=<existing run dir> (default: new probe/runs/<utc>-att), ATT_REPS (20), ATT_ARGS (extra harness args), SKIP_MK=1, ATT_BIN (binary stem, default att_k0)
set -uo pipefail
cd /raid/garv901/vibe-megakernel
source megabench/sota/slurm/env.sh
P=megabench/sota/pi05
D=${ATTD:-$P/probe/runs/$(date -u +%Y%m%dT%H%M%SZ)-att}
mkdir -p $D/raw
U=$P/probe/ubench/att
B=${ATT_BIN:-att_k0}
REPS=${ATT_REPS:-20}
UMMA="--smem 197632 --cluster 2"
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,clocks.max.sm --format=csv | tee $D/gpu_before.csv
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv,noheader,nounits -lms 100 > $D/clocks_sm.csv &
SAMP=$!
# gate 2 (register / spill / stack identical between stamped and unstamped builds)
for b in $B ${B}_ns ${B}_mk; do cuobjdump -res-usage $U/$b > $D/resusage_$b.txt 2>&1; done
for b in $B ${B}_ns ${B}_mk; do cp $D/resusage_$b.txt $D/resusage_${b/$B/${B}_umma}.txt; done
for cfg in "$B:" "${B}_umma:$UMMA"; do
  n=${cfg%%:*}; a=${cfg#*:}
  echo "=== harness $n (stamped)"
  timeout 400 $U/$B --reps $REPS --out $D --name $n $a ${ATT_ARGS:-} > $D/$n.stdout 2> $D/$n.stderr; echo "$n rc=$?"
  echo "=== harness $n (unstamped, ATT_STAMPS=0)"
  timeout 400 $U/${B}_ns --reps $REPS --out $D --name ${n}_ns $a ${ATT_ARGS:-} > $D/${n}_ns.stdout 2> $D/${n}_ns.stderr; echo "${n}_ns rc=$?"
  echo "=== sub-phase timeline binary $n (ATT_MARKS=1)"
  timeout 300 $U/${B}_mk --reps 6 --out $D --name ${n}_mk $a ${ATT_ARGS:-} > $D/${n}_mk.stdout 2> $D/${n}_mk.stderr; echo "${n}_mk rc=$?"
done
if [ -z "${SKIP_MK:-}" ]; then
echo "=== megakernel (FG_UMMA / default) vs NULL_ATT"
PROBE_PREFIX=968 PROBE_OUT=$D/mk ROUNDS=${MK_ROUNDS:-40} .venv/bin/python -m megabench.sota.pi05.probe.mk_nullatt_ab 2>&1 | tail -6
echo "mk rc=${PIPESTATUS[0]}"
fi
kill $SAMP 2>/dev/null
echo "=== analyze"
.venv/bin/python -I $U/att_analyze.py $D $B ${B}_umma 2>&1 | tee $D/analyze.txt
echo "D=$D"
