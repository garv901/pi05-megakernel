#!/bin/bash
# build.sh [VARIANT_FILE_STEM=att_k0] [extra nvcc flags]: builds <stem>, <stem>_ns (ATT_STAMPS=0), <stem>_mk (ATT_MARKS=1) next to the sources (login node)
set -euo pipefail
cd "$(dirname "$0")"
S=${1:-att_k0}; shift || true
N="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a $*"
$N $S.cu -o $S & $N -DATT_STAMPS=0 $S.cu -o ${S}_ns & $N -DATT_MARKS=1 $S.cu -o ${S}_mk & wait
