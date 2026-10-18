"""Read ncu PM-sampling series (absolute epoch-ns timestamps) through ncu_report and fold them onto megakernel phases.

Everything here is derived from the .ncu-rep files + tstamp .npy files; nothing is fitted except the single global
sample-timestamp convention offset (see calibrate_offset)."""
from __future__ import annotations
import glob, os, sys
import numpy as np
sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.4.1/extras/python")
import ncu_report  # noqa: E402
from .phases import PHASES, mk_slot, L, NSTEP  # noqa: E402


_KEEP = []   # ncu_report objects must outlive their actions (segfault otherwise)


def load_action(path, idx=0):
    r = ncu_report.load_report(path)
    rg = r.range_by_idx(0)
    _KEEP.extend([r, rg])
    return rg.action_by_idx(idx)


def passes(a):
    """[(start_ns, end_ns)] of each replay pass (profiler__timestamp_workload_{start,end}_k)."""
    out, k = [], 0
    while True:
        s = a.metric_by_name(f"profiler__timestamp_workload_start_{k}")
        e = a.metric_by_name(f"profiler__timestamp_workload_end_{k}")
        if s is None or e is None:
            break
        out.append((int(s.correlation_ids().as_uint64(0)), int(e.correlation_ids().as_uint64(0))))
        k += 1
    return out


def series(a, name):
    m = a.metric_by_name(name)
    if m is None or m.num_instances() == 0:
        return None
    n = m.num_instances()
    cid = m.correlation_ids()
    t = np.fromiter((cid.as_uint64(i) for i in range(n)), dtype=np.int64, count=n)
    v = np.fromiter((m.as_double(i) for i in range(n)), dtype=np.float64, count=n)
    return t, v


def pm_names(a):
    out = []
    for n in a.metric_names():
        m = a.metric_by_name(n)
        if m.num_instances() > 1 and m.has_correlation_ids():
            out.append(n)
    return out


def pass_of(t0, pss, slack=200_000):
    for k, (s, e) in enumerate(pss):
        if s - slack <= t0 <= e + slack:
            return k
    return -1


def windows_from_stamps(ts, t_end=None):
    """Return list of (name, step, layer, phase, t_start_ns, t_end_ns) for prologue P1..P5, 9 phases x 180 layers, tails."""
    ts = ts.astype(np.int64)
    W = []
    for i in range(5):
        W.append((f"P{i+1}", -1, -1, f"P{i+1}", ts[i], ts[i + 1]))
    for s in range(NSTEP):
        for l in range(L):
            for i, p in enumerate(PHASES):
                a = mk_slot(s, l, i)
                if i < 8:
                    b = ts[a + 1]
                elif l < L - 1:
                    b = ts[mk_slot(s, l + 1, 0)]
                else:
                    b = ts[a + 1]
                W.append((p, s, l, p, ts[a], b))
        # step tail after the last layer's residual stamp
        a = mk_slot(s, L - 1, 9)
        if s < NSTEP - 1:
            W.append(("TAIL", s, L - 1, "TAIL", ts[a], ts[mk_slot(s + 1, 0, 0)]))
        elif t_end is not None:
            W.append(("TAIL", s, L - 1, "TAIL", ts[a], t_end))
    return W


def make_integrator(t, v, dt_ns, off_ns):
    """Piecewise-constant signal: sample j holds value v_j over (e_{j-1}, e_j], e_j = t_j + off (contiguous bins; first bin is dt wide).
    Returns F(x) = integral of the signal from -inf to x (ns * value)."""
    e = t.astype(np.float64) + off_ns
    edges = np.concatenate([[e[0] - dt_ns], e])
    cum = np.concatenate([[0.0], np.cumsum(v * np.diff(edges))])
    def F(x):
        x = np.asarray(x, dtype=np.float64)
        j = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(v))   # bin index; len(v) => beyond the end
        jj = np.clip(j, 0, len(v) - 1)
        inside = (x >= edges[0]) & (x < edges[-1])
        return np.where(x >= edges[-1], cum[-1], np.where(inside, cum[jj] + v[jj] * (x - edges[jj]), 0.0))
    return F


def window_mean(F, a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    return (F(b) - F(a)) / (b - a)


# ----------------------------------------------------------------------------------------------------------------------
# megakernel per-phase folding
TRIAGE = {   # column -> ncu PM metric (pass 0 of `--section PmSampling`)
    "dram_rd_pct": "FBSP.TriageCompute.dram__read_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_wr_pct": "FBSP.TriageCompute.dram__write_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "FBSP.TriageCompute.dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_pct": "LTS.TriageCompute.lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit_pct": "LTS.TriageCompute.lts__average_t_sector_hit_rate_realtime.pct",
    "sm_active_frac": "TPC.TriageCompute.sm__cycles_active.avg.per_cycle_elapsed",
    "ctas_active": "TPC.TriageCompute.tpc__ctas_active_realtime.sum.per_cycle_elapsed",
    "ipc_active": "TPC.TriageCompute.sm__inst_executed_realtime.avg.per_cycle_active",
    "issue_pct": "TPC.TriageCompute.sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
    "tensor_pct": "TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
    "l1_pct": "SM_A.TriageCompute.l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
}
STALLS = ["selected", "not_selected", "barrier", "membar", "wait", "long_scoreboard", "short_scoreboard", "mio_throttle", "lg_throttle",
          "math_pipe_throttle", "no_instructions", "branch_resolving", "dispatch_stall", "drain", "misc", "sleeping", "tex_throttle"]


def stall_name(r):
    return f"warpsampling:smsp__pcsamp_warps_issue_stalled_{r}"


def match_ts(start_ns, files, tol=100_000):
    best = None
    for f in files:
        ts = np.load(f).astype(np.int64)
        d = start_ns - ts[0]
        if abs(d) < tol and (best is None or abs(d) < abs(best[1])):
            best = (ts, d, f)
    return best


def integrator(t, v, dt_ns=1000.0, conv="start"):
    """Sample timestamp convention: conv='start' -> sample j covers [t_j, t_{j+1}); 'end' -> (t_{j-1}, t_j]."""
    t = np.asarray(t, dtype=np.float64)
    if conv == "start":
        edges = np.concatenate([t, [t[-1] + dt_ns]])
    else:
        edges = np.concatenate([[t[0] - dt_ns], t])
    cum = np.concatenate([[0.0], np.cumsum(v * np.diff(edges))])
    n = len(v)

    def F(x):
        x = np.asarray(x, dtype=np.float64)
        j = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)
        out = cum[j] + v[j] * (x - edges[j])
        out = np.where(x >= edges[-1], cum[-1], out)
        return np.where(x <= edges[0], 0.0, out)
    return F


def val(a, name, default=None):
    m = a.metric_by_name(name)
    if m is None:
        return default
    try:
        return m.as_double()
    except Exception:
        return default


def fold_profile(t, v, starts, lag_ns=0.0, per=44, sample_ns=1000.0):
    """Mean of samples (start-convention timestamps) binned by time since the nearest preceding layer start (1 us bins)."""
    tt = np.asarray(t, float) + lag_ns
    idx = np.searchsorted(starts, tt, side="right") - 1
    ok = (idx >= 0) & (idx < len(starts))
    rel = (tt[ok] - starts[idx[ok]]) / 1e3
    ok2 = (rel < per) & (rel >= 0)
    b = rel[ok2].astype(int)
    s_ = np.bincount(b, weights=np.asarray(v)[ok][ok2], minlength=per)[:per]
    c = np.bincount(b, minlength=per)[:per]
    return s_ / np.maximum(c, 1)


def boxcar(prof, n):
    """what a sampler that averages n us (start convention) would see of a 1 us profile"""
    per = len(prof)
    return np.array([prof[np.arange(i, i + n) % per].mean() for i in range(per)])
