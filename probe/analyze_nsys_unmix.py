"""Overlap-weighted NNLS unmixing of nsys GPU-metric samples (200 kHz = 5 us) onto phases, for graph-pdl / graph-nopdl / mk.
usage: python -m megabench.sota.pi05.probe.analyze_nsys_unmix RUN_DIR OUT_JSON"""
import glob, json, sqlite3, sys
import numpy as np
from . import graph_analyze as G, unmix as U, ncu_pm as N
from .phases import PHASES, NSTEP, L

run = sys.argv[1] if len(sys.argv) > 1 else ""
out = sys.argv[2] if len(sys.argv) > 2 else ""
MET = {"gr_active": 2, "sms_active": 3, "sm_issue": 4, "tensor_active": 5, "warps_in_flight_pct": 6, "warps_per_cycle": 8,
       "dram_read_pct": 9, "dram_write_pct": 10}
LG = ["all"]   # pooled over layers: the 3-group (l0 / l1-16 / l17) variant was ill-conditioned (2-8 us windows vs 5 us samples)


def lgroup(l):
    return 0


def metrics(db):
    c = sqlite3.connect(db)
    gm = np.array(c.execute("select timestamp,metricId,value from GPU_METRICS order by timestamp").fetchall(), dtype=np.float64)
    return {n: (gm[gm[:, 1] == i, 0], gm[gm[:, 1] == i, 2]) for n, i in MET.items()}


def graph_windows(csvp):
    reps = G.load_reps(csvp)
    cls, A, B = [], [], []
    names = ["PRO"] + [f"{p}|{g}" for g in LG for p in PHASES] + ["GAP"]
    idx = {n: i for i, n in enumerate(names)}
    prev_end = None
    spans = []
    for rep in reps:
        s0 = G.find_layer_start(rep)
        st = np.array([int(r["Start (ns)"]) for r in rep], float); en = st + np.array([int(r["Duration (ns)"]) for r in rep], float)
        if prev_end is not None:
            cls.append(idx["GAP"]); A.append(prev_end); B.append(st[0])
        spans.append((st[0], en.max()))
        pe = st[0]
        for k in range(len(rep)):
            if k < s0: c = idx["PRO"]
            elif k >= s0 + NSTEP * L * 9: c = idx["PRO"]
            else:
                j = k - s0; m = j // 9; c = idx[f"{PHASES[j % 9]}|{LG[lgroup(m % L)]}"]
            cls.append(c); A.append(pe); B.append(en[k]); pe = max(pe, en[k])
        prev_end = pe
    return names, np.array(cls), np.array(A), np.array(B), spans


def mk_windows(db, ts):
    k0 = sqlite3.connect(db).execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()[-1]
    W = N.windows_from_stamps(ts, t_end=int(ts[0]) + (k0[1] - k0[0]))
    names = [f"P{i+1}" for i in range(5)] + [f"{p}|{g}" for g in LG for p in PHASES] + ["TAIL", "OUT"]
    idx = {n: i for i, n in enumerate(names)}
    cls, A, B = [], [], []
    for (nm, s, l, ph, a, b) in W:
        if nm.startswith("P") or nm == "TAIL": c = idx[nm]
        else: c = idx[f"{ph}|{LG[lgroup(l)]}"]
        cls.append(c); A.append(k0[0] + (a - ts[0])); B.append(k0[0] + (b - ts[0]))
    return names, np.array(cls), np.array(A, float), np.array(B, float), [(k0[0], k0[1])]


def run_arm(db, names, cls, A, B, spans, lags, tag, metrics_use=("dram_read_pct", "dram_write_pct", "sms_active", "sm_issue", "tensor_active", "warps_in_flight_pct", "gr_active")):
    M = metrics(db)
    dt = 5000.0
    t_lo = min(s for s, e in spans); t_hi = max(e for s, e in spans)
    best = None
    scan = {}
    for lg in lags:
        tot = 0
        for m in ("dram_read_pct",):
            t, v = M[m]; sel = (t > t_lo - 50_000) & (t < t_hi + 50_000)
            r = U.solve(t[sel], v[sel], dt, lg, cls, A, B, len(names), t_lo=None, t_hi=None)
            tot += r["rmse"] / (v[sel].std() + 1e-9)
        scan[lg] = tot
        if best is None or tot < best[1]: best = (lg, tot)
    lag = best[0]
    res = {"lag_ns": lag, "lag_scan": {str(k): v for k, v in scan.items()}, "metrics": {}}
    for m in metrics_use:
        t, v = M[m]; sel = (t > t_lo - 50_000) & (t < t_hi + 50_000)
        r = U.solve(t[sel], v[sel], dt, lag, cls, A, B, len(names))
        res["metrics"][m] = dict(x={n: float(x) for n, x in zip(names, r["x"])}, rmse=r["rmse"], r2=r["r2"], n=r["n"], cond=r["cond_cols"],
                                  sample_std=float(v[sel].std()), sample_mean=float(v[sel].mean()))
    return res


def main():
    result = {"graph-pdl": run_arm(f"{run}/nsys_graph-pdl_gb10x.sqlite", *graph_windows(f"{run}/nsys_graph-pdl_gb10x_trace_cuda_gpu_trace.csv"), np.arange(-500, 2501, 250), "graph-pdl"),
              "graph-nopdl": run_arm(f"{run}/nsys_graph-nopdl_gb10x.sqlite", *graph_windows(f"{run}/nsys_graph-nopdl_gb10x_trace_cuda_gpu_trace.csv"), np.arange(-500, 2501, 250), "graph-nopdl")}
    for r in (1, 2, 3):
        ts = np.load(f"{run}/nsys_mk_gb10x_r{r}_tstamp.npy").astype(np.int64)
        result[f"mk_r{r}"] = run_arm(f"{run}/nsys_mk_gb10x_r{r}.sqlite", *mk_windows(f"{run}/nsys_mk_gb10x_r{r}.sqlite", ts), np.arange(-3000, 3001, 250), f"mk_r{r}")
    json.dump(result, open(out, "w"), indent=1, default=float)
    for k, v in result.items():
        d = v["metrics"]["dram_read_pct"]
        print(k, "lag", v["lag_ns"], "dram_rd r2", round(d["r2"], 3), "rmse", round(d["rmse"], 2), "std", round(d["sample_std"], 2))
        print("   ", {n: round(x, 1) for n, x in d["x"].items() if n.endswith("|all") or n.startswith("P") or n in ("PRO", "TAIL", "GAP")})


if __name__ == '__main__':
    main()
