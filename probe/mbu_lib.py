"""MBU accounting helpers (prefetch-aware). All bytes derive from ncu PM `dram__read_throughput` (% of ncu peak) samples.
Peak = ncu-implied 7.645 TB/s; nominal 8 TB/s and a measured copy BW are secondary."""
from __future__ import annotations
import glob, os
import numpy as np
from . import ncu_pm as N
from .phases import PHASES, mk_slot, L, NSTEP

PM_SHIFT_NS = float(os.environ.get("PM_SHIFT_NS", "0"))   # sensitivity: delay the PM series by this much (PM lead)
PEAK = 7.645e12            # B/s, ncu-implied
NOMINAL = 8.0e12
COPY_BW = 6.5e12           # measured copy BW (stage0 records.jsonl hbm_copy_bw)
# unique HBM bytes per layer stream (B), p128 (phases.ideal_bytes, kernel source)
BYTES = {"B_WoKV": 1024 * 2048 * 2 + 2 * 128 * 256 * 2,       # Wo + prefix K + V
         "C_WgWu": 2 * 4096 * 1024 * 2,
         "D_Wd": 1024 * 4096 * 2 + 6 * 1024 * 4,                # Wd + 2 mod rows (6W fp32)
         "G_QKV": (2048 + 2 * 256) * 1024 * 2}                  # next layer's Wq, Wk, Wv
STREAMS = list(BYTES)
# phase index of each stream's consumer (phase start slot k of the layer; G_QKV consumer is the next layer's A, k = 9)
CONSUMER_SLOT = {"B_WoKV": 3, "C_WgWu": 5, "D_Wd": 7, "G_QKV": 9}


def load_mk_rep(rep_path, ts_files):
    """Return dict(t, v (dram rd %), ts (first 4096 block-0 stamps), raw (whole saved buffer, absolute ns), S, E, base).
    All times in t/ts/S/E are int-shifted by base (= stamp[0]) to keep float64 precision at ns (epoch ns is 1.8e18)."""
    a = N.load_action(rep_path)
    P = N.passes(a)
    t, v = N.series(a, N.TRIAGE["dram_rd_pct"])
    vw = N.series(a, N.TRIAGE["dram_wr_pct"])[1]
    S, E = P[N.pass_of(t[0], P)]
    raw, d, f = N.match_ts(S, ts_files)
    raw = raw.astype(np.int64)
    base = int(raw[0])
    return dict(t=(t - base).astype(np.float64) + PM_SHIFT_NS, v=v, vw=vw, ts=raw[:4096] - base, raw=raw, S=float(S - base) + PM_SHIFT_NS, E=float(E - base) + PM_SHIFT_NS, tsfile=f, a=a, base=base)


def F_of(t, v):
    return N.integrator(t, v, 1000.0, "start")


def mb(F, a, b, peak=PEAK):
    """bytes in [a,b] (ns) from the % series"""
    return (F(np.asarray(b, float)) - F(np.asarray(a, float))) / 100.0 * peak * 1e-9


def layer_windows_mk(ts):
    """(step, layer, start, end) with end = next layer's A-in (last layer: its I-end stamp, i.e. excl. the step tail)."""
    out = []
    for s in range(NSTEP):
        for l in range(L):
            a = ts[mk_slot(s, l, 0)]
            b = ts[mk_slot(s, l + 1, 0)] if l < L - 1 else ts[mk_slot(s, l, 9)]
            out.append((s, l, a, b))
    return out


def phase_windows_mk(ts, s, l):
    out = []
    for i, p in enumerate(PHASES):
        a = ts[mk_slot(s, l, i)]
        b = ts[mk_slot(s, l, i + 1)] if i < 8 or l == L - 1 else ts[mk_slot(s, l + 1, 0)]
        if i == 8 and l < L - 1:
            b = ts[mk_slot(s, l + 1, 0)]
        out.append((p, a, b))
    return out


def sample_stats(t, v, a, b):
    """min/mean/max over samples whose start lies in [a,b) plus overlap-weighted mean"""
    m = (t >= a) & (t < b)
    if not m.any():
        return (np.nan,) * 3
    return float(v[m].min()), float(v[m].mean()), float(v[m].max())


def busy_fracs(t, v, a, b, thr=(10, 25, 50)):
    """time fraction (sample-count weighted, samples starting in [a,b)) above thr % of peak, and mean % of those samples"""
    m = (t >= a) & (t < b)
    x = v[m]
    out = {}
    for th in thr:
        s = x >= th
        out[th] = (float(s.mean()), float(x[s].mean()) if s.any() else 0.0)
    return out


# ----------------------------------------------------------------------------------------------- graph
GR = "megabench/sota/pi05/probe/ncu_artifacts"
GRAPH_REPS = {"graph-pdl": [("graph_pm/graph_wholegraph_pm_graph-pdl.ncu-rep", "analysis/graph_pm_graph-pdl.npz"),
                            ("graph_pm/graph_wholegraph_pm_graph-pdl_r2.ncu-rep", "analysis/graph_pm_graph-pdl_r2.npz"),
                            ("graph_pm/graph_wholegraph_pm_graph-pdl_r3.ncu-rep", "analysis/graph_pm_graph-pdl_r3.npz")],
              "graph-nopdl": [("graph_pm/graph_wholegraph_pm_graph-nopdl.ncu-rep", "analysis/graph_pm_graph-nopdl.npz")]}


def load_graph_rep(rep, npz):
    a = N.load_action(os.path.join(GR, rep))
    z = np.load(os.path.join(GR, npz), allow_pickle=True)
    t, v = N.series(a, N.TRIAGE["dram_rd_pct"])
    vw = N.series(a, N.TRIAGE["dram_wr_pct"])[1]
    kt = int(z["triage_pass"])
    st = np.sort(N.series(a, f"profiler__timestamp_workload_start_{kt}")[0]).astype(np.int64)
    en = np.sort(N.series(a, f"profiler__timestamp_workload_end_{kt}")[0]).astype(np.int64)
    base = st[0]
    return dict(t=(t - base).astype(np.float64) + PM_SHIFT_NS, v=v, vw=vw, st=st - base, en=en - base, phase=z["phase"], step=z["step"], layer=z["layer"], a=a, base=int(base))


def graph_index(G):
    """idx[(step, layer, phase)] -> node index"""
    idx = {}
    for i, (p, s, l) in enumerate(zip(G["phase"], G["step"], G["layer"])):
        if p != "PRO":
            idx[(int(s), int(l), str(p))] = i
    return idx


def layer_windows_graph(G):
    idx = graph_index(G)
    out = []
    for s in range(NSTEP):
        for l in range(L):
            a = G["st"][idx[(s, l, "A_qkv")]]
            b = G["st"][idx[(s, l + 1, "A_qkv")]] if l < L - 1 else G["en"][idx[(s, l, "I_res2")]]
            out.append((s, l, a, b))
    return out


# ----------------------------------------------------------------------------------------------- CDF / burst helpers
def t_at_cum(F, a, b, target_bytes, step=50.0):
    """first time in [a,b] where the integrated DRAM-read bytes since a reach target (linear interpolation on a 50 ns grid)."""
    g = np.arange(a, b + step, step)
    c = mb(F, a, g)
    j = np.searchsorted(c, target_bytes)
    if j <= 0:
        return float(g[0])
    if j >= len(g):
        return float("nan")
    f = (target_bytes - c[j - 1]) / max(c[j] - c[j - 1], 1e-9)
    return float(g[j - 1] + f * step)


def pf_split(arr, base=0):
    """split a saved PFTRACE+PTRACE buffer into (stamps, ptrace[1700,148,2], pf[180,4,148,8,2]) (absolute ns, 0 = unwritten)"""
    G = 148
    pt_n = 4096 + 1700 * G * 2
    pf_n = 180 * 4 * G * 8 * 2
    sh = lambda x: np.where(x > 0, x - base, 0)      # keep 0 = unwritten
    return sh(arr[:4096]), sh(arr[4096:pt_n]).reshape(1700, G, 2), sh(arr[pt_n:pt_n + pf_n]).reshape(180, 4, G, 8, 2)
