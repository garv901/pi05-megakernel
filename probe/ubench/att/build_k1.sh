#!/bin/bash
# build_k1.sh TAG name1,name2,...: builds att_k1_<TAG>, _ns, _mk containing only the listed K1 variants (+ null + k0).
# A small binary matters: the full variant set (13+ unit instantiations) inflates the kernel text and slows the *null* baseline by ~12% (8.0 -> 9.1 ms
# per 180 layers), which shifts every L0 delta; per-family binaries keep null at the att_k0 level.
set -euo pipefail
cd "$(dirname "$0")"
T=$1; L=$2
python3 - "$T" "$L" <<'PY'
import re,sys
t,names=sys.argv[1],sys.argv[2].split(',')
src=open('att_k1.cu').read()
ent={}
for m in re.finditer(r'X\((\d+), "([a-z0-9_]+)", (\d+), (\d+), ([^)]*)\)', src[src.index('#define K1_ALL'):src.index('#define K1_LIST(X) K1_ALL')]):
    ent[m.group(2)]=m.group(0)
open(f'k1_sel_{t}.h','w').write('#define K1_SELECT(X) '+' '.join(ent[n] for n in names)+'\n')
PY
N="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a -DATT_HOOKS -DK1_SEL_H=\"k1_sel_$T.h\""
$N att_k1.cu -o att_k1_$T & $N -DATT_STAMPS=0 att_k1.cu -o att_k1_${T}_ns & $N -DATT_MARKS=1 att_k1.cu -o att_k1_${T}_mk & wait
