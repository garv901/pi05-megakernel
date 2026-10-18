"""Fold the ncu PM-sampling series of the megakernel (same-run tstamps) onto the phase windows.
usage: python -m megabench.sota.pi05.probe.analyze_mk_pm RUN_DIR OUT_DIR [reps]"""
import glob, json, sys
import numpy as np
from . import ncu_pm as N
from .phases import PHASES, L, NSTEP

run, out = sys.argv[1], sys.argv[2]
reps = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3]
import os; os.makedirs(out, exist_ok=True)
KERNEL_RD_BYTES = None
# kernel-level total DRAM reads of one launch (stage2 kernel replay) -> implied ncu peak
try:
    ka = N.load_action(run.rstrip("/") + "/../20261016T021936Z-stage2/ncu_mk_kernel.ncu-rep")
    KERNEL_RD_BYTES = ka.metric_by_name("dram__bytes_read.sum").as_double()
    KERNEL_WR_BYTES = ka.metric_by_name("dram__bytes_write.sum").as_double()
except Exception as e:
    print("no kernel-level bytes", e)
val = {"reps": {}}
SEL = {}
for r in reps:
    rep = f"{run}/mk_pm_triage_warp_r{r}.ncu-rep"
    files = glob.glob(f"{run}/mk_pm_triage_warp_r{r}_ts_*.npy")
    a = N.load_action(rep); P = N.passes(a)
    v = {"passes": []}
    for k, (s, e) in enumerate(P):
        m = N.match_ts(s, files)
        v["passes"].append(dict(pass_idx=k, start_ns=s, dur_us=(e - s) / 1e3, ts0_minus_start_us=(-m[1]) / 1e3 if m else None,
                                 tstamp_file=os.path.basename(m[2]) if m else None))
    data = {}
    # -- triage pass
    tr = {c: N.series(a, n) for c, n in N.TRIAGE.items()}
    t0 = tr["dram_rd_pct"][0][0]; kt = N.pass_of(t0, P)
    S, E = P[kt]; ts, d, f = N.match_ts(S, files)
    v["triage_pass"] = kt; v["triage_tstamp"] = os.path.basename(f); v["triage_samples"] = int(len(tr["dram_rd_pct"][0]))
    W = N.windows_from_stamps(ts, t_end=E)
    A = np.array([w[4] for w in W], float); B = np.array([w[5] for w in W], float)
    meta = dict(phase=np.array([w[0] for w in W]), step=np.array([w[1] for w in W]), layer=np.array([w[2] for w in W]), a=A, b=B)
    for conv in ("start", "end"):
        res = {}
        for c, (t, x) in tr.items():
            F = N.integrator(t, x, 1000.0, conv)
            res[c] = N.window_mean(F, A, B)
            res[c + "_int"] = (F(B) - F(A))      # value*ns
        data[conv] = res
    # shifted versions for the sensitivity analysis (start convention)
    sens = {}
    for sh in (-2000, -1000, 0, 1000, 2000):
        t, x = tr["dram_rd_pct"]
        F = N.integrator(t + sh, x, 1000.0, "start")
        sens[sh] = N.window_mean(F, A, B)
    # -- warp-sampling pass (own tstamp)
    ws = {r_: N.series(a, N.stall_name(r_)) for r_ in N.STALLS}
    kw = N.pass_of(ws["wait"][0][0], P); Sw, Ew = P[kw]; tsw, dw, fw = N.match_ts(Sw, files)
    Ww = N.windows_from_stamps(tsw, t_end=Ew)
    Aw = np.array([w[4] for w in Ww], float); Bw = np.array([w[5] for w in Ww], float)
    stall = {}
    for r_, (t, x) in ws.items():
        F = N.integrator(t, x, 1000.0, "start")
        stall[r_] = (F(Bw) - F(Aw))
    v["warp_pass"] = kw; v["warp_tstamp"] = os.path.basename(fw)
    v["warp_samples"] = int(len(ws["wait"][0]))
    dr = N.series(a, "warpsampling:smsp__pcsamp_samples_data_dropped")
    v["warp_dropped_total"] = float(dr[1].sum())
    # -- validation: ctas_active edges (start/end) relative to ncu workload start/end
    t, x = tr["ctas_active"]
    plateau = np.median(x[(t > S + 50_000) & (t < E - 50_000)])
    def edge(conv):
        F = N.integrator(t, x, 1000.0, conv)
        # integral of ctas (value*ns) over [S-20us, S+20us] vs plateau*(window - edge) -> edge position
        lo, hi = S - 20_000, S + 20_000
        ramp_up = (hi - lo) - (F(np.array([hi]))[0] - F(np.array([lo]))[0]) / plateau       # ns after lo where plateau begins
        e_start = lo + ramp_up - S
        lo2, hi2 = E - 20_000, E + 20_000
        ramp_dn = (F(np.array([hi2]))[0] - F(np.array([lo2]))[0]) / plateau     # ns of plateau-equivalent inside window
        # window has full for tail of CTAs: equivalent end position = lo2 + ramp_dn  (but tail is ragged: report 'CTA-time weighted end')
        return dict(cta_weighted_start_minus_S_us=e_start / 1e3, cta_weighted_end_minus_E_us=(lo2 + ramp_dn - E) / 1e3)
    v["edges"] = {c: edge(c) for c in ("start", "end")}
    v["ts0_minus_S_us"] = float(-d / 1e3)
    v["plateau_ctas"] = float(plateau)
    # -- prologue burst bytes: P3 window (stamp2->stamp3?) see phases: P_i = [ts[i-1],ts[i]]  -> P4 = ts3->ts4 is the long one
    names = meta["phase"]
    v["prologue_windows_us"] = [float((B[i] - A[i]) / 1e3) for i in range(5)]
    np.savez(f"{out}/mk_pm_windows_r{r}.npz", **meta, **{f"start_{k}": x for k, x in data["start"].items()},
             **{f"end_{k}": x for k, x in data["end"].items()}, **{f"sens_{k}": x for k, x in sens.items()},
             **{f"stall_{k}": x for k, x in stall.items()}, ts=ts, tsw=tsw, S=S, E=E, Sw=Sw, Ew=Ew,
             a_w=Aw, b_w=Bw)
    # total DRAM read pct*ns integral over the whole kernel
    F = N.integrator(*tr["dram_rd_pct"], 1000.0, "start")
    tot = float(F(np.array([E + 30_000.]))[0] - F(np.array([S - 30_000.]))[0]) / 100 * 1e-9      # (pct/100)*s
    v["rd_pct_s_total"] = tot
    if KERNEL_RD_BYTES:
        v["implied_peak_TBps"] = KERNEL_RD_BYTES / tot / 1e12
    val["reps"][r] = v
    print("rep", r, json.dumps({k: v[k] for k in ("triage_pass", "warp_pass", "edges", "ts0_minus_S_us", "implied_peak_TBps", "prologue_windows_us", "warp_dropped_total")}, default=float))
json.dump(val, open(f"{out}/mk_pm_validation_raw.json", "w"), indent=1, default=float)
