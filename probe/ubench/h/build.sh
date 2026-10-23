#!/bin/bash
# build.sh OUTDIR: H harness binaries (stamped + _ns twins) on the login node. h_gemm.inc = the verbatim gemm_phase block of ../ab/ab_k0.cu.
set -e
O=${1:?}; mkdir -p $O
cd $(dirname $0)
sed -n '/^\/\/ ---- VERBATIM from mk/,/^\/\/ Four consecutive/p' ../ab/ab_k0.cu | sed '$d' > h_gemm.inc
N="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -lcuda -diag-suppress 550,177"
for s in "" "_ns"; do
  F=""; [ -n "$s" ] && F="-DFG_STAMPS=0"
  $N $F h_bench.cu -o $O/h$s &
done
wait
