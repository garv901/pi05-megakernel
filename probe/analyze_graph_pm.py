"""Whole-graph ncu PM sampling (--graph-profiling graph) folded on per-node completion-to-completion windows.
Node timestamps = profiler__timestamp_workload_{start,end}_<pass> (one instance per graph node, same clock as the PM samples).
usage: python -m megabench.sota.pi05.probe.analyze_graph_pm REP.ncu-rep OUT.npz"""
import json, sys
import numpy as np
from . import ncu_pm as N
from .phases import PHASES, L, NSTEP

rep, out = sys.argv[1], sys.argv[2]
a = N.load_action(rep)
nodes = []
k = 0
while a.metric_by_name(f"profiler__timestamp_workload_start_{k}") is not None:
    st = N.series(a, f"profiler__timestamp_workload_start_{k}")[0]; en = N.series(a, f"profiler__timestamp_workload_end_{k}")[0]
    nodes.append((np.sort(st).astype(np.int64), np.sort(en).astype(np.int64))); k += 1
assert len(nodes[0][0]) == 1638, len(nodes[0][0])


def which_pass(t0):
    for k, (st, en) in enumerate(nodes):
        if st.min() - 3_000_000 <= t0 <= en.max() + 3_000_000:
            return k
    return -1


def classes(n=1638):
    cls = []
    for i in range(n):
        # 1638 graph nodes = 18 prologue nodes (17 kernels + 1 memory node) + 1620 layer kernels (identified by period-9 duration pattern: corr 0.935 at offset 18)
        if i < n - 1620: cls.append(("PRO", -1, -1, i))
        else:
            j = i - (n - 1620); m = j // 9; cls.append((PHASES[j % 9], m // L, m % L, 0))
    return cls


def windows(k):
    st, en = nodes[k]
    A = np.concatenate([[st[0]], en[:-1]]).astype(float); B = en.astype(float)
    return A, B, st.astype(float)


cls = classes()
meta = dict(phase=np.array([c[0] for c in cls]), step=np.array([c[1] for c in cls]), layer=np.array([c[2] for c in cls]), pro_idx=np.array([c[3] for c in cls]))
res = {}
val = {"n_samples": {}}
tri = {c: N.series(a, n) for c, n in N.TRIAGE.items()}
kt = which_pass(tri["dram_rd_pct"][0][0]); res["triage_pass"] = kt
A, B, ST = windows(kt)
res.update(a=A, b=B, start=ST)
for c, (t, x) in tri.items():
    F = N.integrator(t, x, 1000.0, "start")
    res[c] = N.window_mean(F, A, B); res[c + "_int"] = F(B) - F(A)
    # own (start..end) window as secondary
    res[c + "_own"] = N.window_mean(F, ST, B)
val["n_samples"]["triage"] = int(len(tri["dram_rd_pct"][0]))
ws = {r: N.series(a, N.stall_name(r)) for r in N.STALLS}
kw = which_pass(ws["wait"][0][0]); res["warp_pass"] = kw
Aw, Bw, STw = windows(kw)
for r_, (t, x) in ws.items():
    F = N.integrator(t, x, 1000.0, "start")
    res["stall_" + r_] = F(Bw) - F(Aw)
res.update(a_w=Aw, b_w=Bw)
# PDL overlap evidence
st, en = nodes[kt]
res["lead_us"] = (np.concatenate([[st[0]], en[:-1]]) - st) / 1e3   # >0: kernel started before predecessor ended
val.update(triage_pass=kt, warp_pass=kw, span_us=[float((e.max() - s.min()) / 1e3) for s, e in nodes],
           frac_overlap=float((res["lead_us"][1:] > 0).mean()), median_lead_us=float(np.median(res["lead_us"][1:])),
           dropped=float(N.series(a, "warpsampling:smsp__pcsamp_samples_data_dropped")[1].sum()))
# start-edge validation (ctas_active vs first node start)
t, x = tri["ctas_active"]
S = nodes[kt][0].min()
plateau = x[np.searchsorted(t, S + 1_200)]     # CTAs resident in the first node's steady state
for conv in ("start", "end"):
    F = N.integrator(t, x, 1000.0, conv)
    lo, hi = S - 3_000, S + 3_000
    ramp = (hi - lo) - (F(np.array([hi]))[0] - F(np.array([lo]))[0]) / plateau
    val[f"cta_weighted_start_minus_S_us_{conv}"] = float((lo + ramp - S) / 1e3)
np.savez(out, **meta, **res)
json.dump(val, open(out.replace(".npz", "_val.json"), "w"), indent=1)
print(json.dumps(val))
