#!/bin/bash
# build.sh OUTDIR: all A+B harness binaries (stamped + _ns twins) on the login node
set -e
O=${1:?}; mkdir -p $O
N="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -lcuda -diag-suppress 550,177"
cd $(dirname $0)
for s in "" "_ns"; do
  F=""; [ -n "$s" ] && F="-DFG_STAMPS=0"
  $N $F ab_k0.cu -o $O/ab_k0$s &
  $N $F -DAB_MASK=7 ab_k1.cu -o $O/ab_k1c$s &
  $N $F -DAB_MASK=8 ab_k1.cu -o $O/ab_k1s$s &
  $N $F -DAB_MASK=1 ab_k1.cu -o $O/ab_k1a$s &
  $N $F -DAB_MASK=4 ab_k1.cu -o $O/ab_k1k$s &
  wait
done
