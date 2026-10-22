"""Per-phase table (A..I) of graph-pdl / graph-nopdl from nsys cuda_gpu_trace CSVs at any prefix.
Handles 9 kernels/layer (single attention kernel), 10 (attention + split-KV merge kernel, both counted in C_attn)
and 11 (also down GEMM + split-K reduce, both counted in H_down; P968).
usage: python -m megabench.sota.pi05.probe.phase_table PREFIX pdl_trace.csv nopdl_trace.csv > table.md"""
import sys
import numpy as np
from . import graph_analyze as G
from .phases import PHASES, L, NSTEP, ideal_bytes

PEAK = 8.0e12


def analyze(rep):
    s0 = G.find_layer_start(rep)
    ropes = [i for i, r in enumerate(rep) if "_rope_kv_kernel" in r["Name"]]
    nk = ropes[1] - ropes[0]
    pmaps = {9: list(range(9)), 10: [0, 1, 2, 2, 3, 4, 5, 6, 7, 8], 11: [0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8]}
    assert nk in pmaps, nk
    pmap = pmaps[nk]
    ks = rep[s0:s0 + NSTEP * L * nk]
    assert len(ks) == NSTEP * L * nk
    st = np.array([int(r["Start (ns)"]) for r in ks]); du = np.array([int(r["Duration (ns)"]) for r in ks]); en = st + du
    prev_end_all = np.concatenate([[int(rep[s0 - 1]["Start (ns)"]) + int(rep[s0 - 1]["Duration (ns)"])], en[:-1]])
    c2c = {p: [] for p in PHASES}; own = {p: [] for p in PHASES}; names = {}
    for layer in range(NSTEP * L):
        base = layer * nk
        pe_prev = prev_end_all[base]
        for pi, p in enumerate(PHASES):
            idx = [base + j for j in range(nk) if pmap[j] == pi]
            end = en[idx[-1]]
            c2c[p].append((end - pe_prev) / 1e3); pe_prev = end
            own[p].append(sum(du[i] for i in idx) / 1e3)
            if layer == 0:
                names[p] = " + ".join(ks[i]["Name"][:32] for i in idx) + f" ; grid {[ (int(ks[i]['GrdX']),int(ks[i]['GrdY']),int(ks[i]['GrdZ'])) for i in idx]}"
    win = (en[-1] - int(rep[0]["Start (ns)"])) / 1e3
    return {p: float(np.mean(c2c[p])) for p in PHASES}, {p: float(np.mean(own[p])) for p in PHASES}, names, win, nk


prefix, fp, fn = int(sys.argv[1]), sys.argv[2], sys.argv[3]
ib = ideal_bytes(prefix)
res = {}
for name, f in (("pdl", fp), ("nopdl", fn)):
    outs = [analyze(r) for r in G.load_reps(f)]
    res[name] = ({p: float(np.mean([o[0][p] for o in outs])) for p in PHASES},
                 {p: float(np.mean([o[1][p] for o in outs])) for p in PHASES}, outs[0][2], [round(float(o[3]), 1) for o in outs], outs[0][4])
mbu = lambda b, t: 100 * b / (t * 1e-6) / PEAK
print(f"### Per-phase time, prefix {prefix} (completion-to-completion us per layer, mean over 180 layers x launches; {res['pdl'][4]} kernels/layer)\n")
print("| phase | graph-pdl us | graph-nopdl us | pdl own-kernel us (sum; overlaps under PDL) | ideal MB | MBU_attr pdl (8 TB/s) | MBU_attr nopdl | kernel(s) (pdl) |")
print("|---|---|---|---|---|---|---|---|")
for p in PHASES:
    b = ib[p]
    f = lambda t: f"{mbu(b, t):.1f}%" if b > 5e5 else "n/a (<0.5 MB)"
    print(f"| {p} | {res['pdl'][0][p]:.2f} | {res['nopdl'][0][p]:.2f} | {res['pdl'][1][p]:.2f} | {b/1e6:.2f} | {f(res['pdl'][0][p])} | {f(res['nopdl'][0][p])} | {res['pdl'][2][p]} |")
sp, sn = sum(res["pdl"][0].values()), sum(res["nopdl"][0].values()); tb = sum(ib.values())
print(f"| **sum/layer** | **{sp:.2f}** | **{sn:.2f}** | {sum(res['pdl'][1].values()):.2f} | {tb/1e6:.2f} | {mbu(tb, sp):.1f}% | {mbu(tb, sn):.1f}% | |")
print(f"\ntotal launch window us (first kernel start to last end): pdl {res['pdl'][3]}, nopdl {res['nopdl'][3]}")
