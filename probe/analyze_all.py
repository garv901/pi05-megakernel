"""Final cross-arm analysis: per-phase gap table inputs, DRAM-over-layer folds (PNG), summary JSON."""
import json, os, sqlite3, sys
import numpy as np
from . import graph_analyze as G, mk_analyze as MA, timeline as TL, pm_parse as PM, ncu_tables as NT
from .phases import PHASES, ideal_bytes, mk_slot, L, NSTEP

R = "/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/"
S0, S2, S3 = R + "20261016T013004Z-stage0/", R + "20261016T021936Z-stage2/", R + "20261016T021936Z-stage3/"
OUT = R + "analysis/"
os.makedirs(OUT, exist_ok=True)
PEAK = 8.0e12          # nominal HBM3e peak B/s
COPY = 6.567e12        # stage0 hbm_copy_bw median (records.jsonl)
res = {}

# ---- megakernel stamps (block 0): stage0 raw + stage2 nsys-run
mk0 = MA.analyze(np.load(S0 + "tstamp_raw.npy"))
mk2 = MA.analyze(np.load(S2 + "nsys_mk_ts_tstamp.npy"))
mk3 = MA.analyze(np.load(S2 + "ncu_mk_pm_tstamp.npy"))
res["mk_stamps_stage0"] = mk0; res["mk_stamps_stage2_nsys"] = mk2["phases_us_per_layer"]
mkp = {p: np.mean([m["phases_us_per_layer"][p]["mean"] for m in (mk0, mk2)]) for p in PHASES}

# ---- graph per phase (stage3, 8 reps each)
gr = {}
for a in ["graph-pdl", "graph-nopdl"]:
    reps = G.load_reps(S3 + f"nsys_{a}_trace_cuda_gpu_trace.csv")
    an = [G.analyze_rep(r) for r in reps]
    gr[a] = dict(an=an,
                 c2c={p: float(np.mean([x["phases"][p]["c2c_us_mean"] for x in an])) for p in PHASES},
                 own={p: float(np.mean([x["phases"][p]["own_us_mean"] for x in an])) for p in PHASES},
                 lead={p: float(np.mean([x["phases"][p]["start_before_prev_end_us_mean"] for x in an])) for p in PHASES},
                 info={p: an[0]["phases"][p] for p in PHASES},
                 window=[x["total_window_us"] for x in an], prologue=[x["prologue_us"] for x in an],
                 layers_window=[x["layers_window_us"] for x in an],
                 end_rel=np.mean([x["phase_end_rel_us_mean"] for x in an], axis=0).tolist())
res["graph"] = {a: {k: v for k, v in g.items() if k not in ("an",)} for a, g in gr.items()}

# ---- ncu per-kernel (graph layer 1 = kernels 9..17 ordered B..I,A) and mk kernel
tab, _, _ = NT.table(S2 + "ncu_graph_layer_raw.csv")
order = ["B_rope", "C_attn", "D_o", "E_res1", "F_gu", "G_act", "H_down", "I_res2", "A_qkv"]
ncu = {p: tab[9 + i] for i, p in enumerate(order)}
res["ncu_graph_layer1"] = ncu
mkk, _, _ = NT.table(S2 + "ncu_mk_kernel_raw.csv"); res["ncu_mk_kernel"] = mkk[0]

# ---- mk PM sampling fold (ncu, 2 us samples), aligned with the stamps of the same run
pm, meta = PM.load(S2 + "ncu_mk_pm_raw.csv")
dur_us = float(meta["gpu__time_duration.sum"]) * 1e3
def series(pm, key):
    return pm[key]
rd = series(pm, "dram__read_throughput.avg.pct_of_peak_sustained_elapsed")
wr = series(pm, "dram__write_throughput.avg.pct_of_peak_sustained_elapsed")
ten = series(pm, "sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed")
cta = series(pm, "tpc__ctas_active_realtime.sum.per_cycle_elapsed")
N = len(rd); dt = dur_us / N
tt = (np.arange(N) + 0.5) * dt
# calibrate pct -> bytes with ncu's own total read bytes
rd_bytes_total = mkk[0]["rd_MB"] * 1e6
k_rd = rd_bytes_total / float((rd * dt * 1e-6).sum())   # B/s per pct
res["mk_pm"] = dict(n=N, dt_us=dt, dur_us=dur_us, implied_peak_TBps=k_rd * 100 / 1e12, mean_rd_pct=float(rd.mean()),
                    mean_wr_pct=float(wr.mean()))
ts = np.load(S2 + "ncu_mk_pm_tstamp.npy").astype(np.int64)
us = (ts - ts[0]) / 1e3
starts = np.array([us[mk_slot(s, l, 0)] for s in range(NSTEP) for l in range(L)])
def fold_series(t, v, starts, period, bins=1.0, lag=0.0):
    nb = int(period / bins); idx = np.searchsorted(starts, t - lag, side="right") - 1
    ok = (idx >= 0) & (idx < len(starts)); rel = (t[ok] - lag - starts[idx[ok]])
    ok2 = rel < period; b = (rel[ok2] / bins).astype(int)
    sm = np.bincount(b, weights=v[ok][ok2], minlength=nb)[:nb]; ct = np.bincount(b, minlength=nb)[:nb]
    return sm / np.maximum(ct, 1)
# mk layer period: measure; align with kernel start offset (stamp 0 is at block-0 start; sample 0 at kernel start)
mk_period = float(np.mean(np.diff(starts[:L])))
res["mk_period_us"] = mk_period
PER = 48
# alignment of the PM series to layer starts is fitted against the nsys fold of the same arm (PM has no absolute clock)
def box(v, n=10):
    k = np.ones(n) / n
    return np.convolve(np.concatenate([v, v[:n - 1]]), k, mode="valid")[:len(v)]
def fit_align(t, v, starts, period, ref, scales, offs):
    ref_b = box(ref[:int(period)]); best = None
    for scale in scales:
        for off in offs:
            pr = fold_series(t, v, starts * scale + off, period)[:int(period)]
            c = float(np.corrcoef(box(pr), ref_b)[0, 1])
            if best is None or c > best[0]: best = (c, scale, off)
    return best
mk_bounds = np.concatenate([[0], np.cumsum([mkp[p] for p in PHASES])])
def phase_integral(prof, bounds, scale_BperPct, bin_us=1.0):
    out = []
    for i in range(9):
        a, b = bounds[i], bounds[i + 1]
        x = np.arange(len(prof)) * bin_us
        w = np.clip(np.minimum(x + bin_us, b) - np.maximum(x, a), 0, None)
        out.append(float((prof * w * 1e-6).sum() * scale_BperPct))
    return out
# ---- nsys GPU-metrics folds (10 us samples): graph arms from stage3, mk from stage2 (1 captured kernel)
LAG = -5000.0
folds = {}
for a in ["graph-pdl", "graph-nopdl"]:
    m = TL.load_metrics(S3 + f"nsys_{a}.sqlite")
    allstarts = np.concatenate([x["layer_starts_ns"] for x in gr[a]["an"]])
    per = 48 if a == "graph-pdl" else 52
    f = TL.fold(m, allstarts, period_us=per, bin_us=1.0, lag_ns=LAG)
    folds[a] = {k: v[0] for k, v in f.items()}
    folds[a]["period"] = float(np.mean([np.mean(np.diff(x["layer_starts_ns"])) for x in gr[a]["an"]]) / 1e3)
mm = TL.load_metrics(S2 + "nsys_mk_ts.sqlite")
c = sqlite3.connect(S2 + "nsys_mk_ts.sqlite")
k0 = c.execute("select start from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()[0][0]
mkstarts_ns = k0 + starts * 1e3
f = TL.fold(mm, mkstarts_ns, period_us=48, bin_us=1.0, lag_ns=LAG)
folds["mk"] = {k: v[0] for k, v in f.items()}
folds["mk"]["period"] = mk_period
# in-situ per-phase bytes from nsys folds (graph): dram_read % of 8 TB/s peak (calibrated below)
tot_rd_graph = {}
for a in ["graph-pdl", "graph-nopdl"]:
    prof = folds[a]["dram_read_pct"]
    per_layer_MB = float(prof[: int(folds[a]["period"]) + 1].sum() * 1e-6 * PEAK / 100 / 1e6)
    bounds = np.concatenate([[0], gr[a]["end_rel"]])
    ob = phase_integral(prof, bounds, PEAK / 100)
    # phases are overlapped; end_rel is completion time of each kernel so these are completion-window bytes
    tot_rd_graph[a] = dict(per_layer_MB=per_layer_MB, per_phase_B=dict(zip(PHASES, ob)))
res["graph_nsys_obs"] = tot_rd_graph
mkprof = folds["mk"]["dram_read_pct"]
res["mk_nsys_per_layer_MB"] = float(mkprof[: int(mk_period) + 1].sum() * 1e-6 * PEAK / 100 / 1e6)

bestm = fit_align(tt, rd, starts, mk_period, folds["mk"]["dram_read_pct"], np.arange(0.97, 1.03, 0.001), np.arange(-120, 120, 1.0))
res["mk_pm_align"] = dict(corr=bestm[0], scale=bestm[1], offset_us=bestm[2])
starts_pm = starts * bestm[1] + bestm[2]
mk_fold = {n: fold_series(tt, v, starts_pm, PER) for n, v in [("rd", rd), ("wr", wr), ("tensor", ten), ("ctas", cta)]}
# per-phase observed read bytes (in situ) for mk: integrate rd between phase boundaries (mean offsets)
mk_obs = phase_integral(folds["mk"]["dram_read_pct"], mk_bounds, PEAK / 100)   # nsys fold (PM fold alignment failed, see report)
res["mk_obs_bytes_per_phase"] = dict(zip(PHASES, mk_obs))
res["mk_bounds"] = mk_bounds.tolist()
mk_cta = [float(np.mean(mk_fold["ctas"][int(mk_bounds[i]):max(int(mk_bounds[i + 1]), int(mk_bounds[i]) + 1)])) for i in range(9)]
mk_ten = [float(np.mean(mk_fold["tensor"][int(mk_bounds[i]):max(int(mk_bounds[i + 1]), int(mk_bounds[i]) + 1)])) for i in range(9)]
res["mk_pm_per_phase"] = {p: dict(ctas_active=mk_cta[i], tensor_pct=mk_ten[i]) for i, p in enumerate(PHASES)}

# ---- graph whole-graph PM (ncu), aligned by nopdl layer starts relative to first kernel
gpm, gmeta = PM.load(S2 + "ncu_graph_pm_raw.csv")
gdur = float(gmeta["gpu__time_duration.sum"]) * 1e3
grd = gpm["dram__read_throughput.avg.pct_of_peak_sustained_elapsed"]; gN = len(grd); gdt = gdur / gN
gt = (np.arange(gN) + 0.5) * gdt
a0 = gr["graph-nopdl"]["an"][0]
reps0 = G.load_reps(S3 + "nsys_graph-nopdl_trace_cuda_gpu_trace.csv")[0]
t_first = int(reps0[0]["Start (ns)"])
gs = (np.array(a0["layer_starts_ns"]) - t_first) / 1e3
gper = folds["graph-nopdl"]["period"]
best = fit_align(gt, grd, gs, gper, folds["graph-nopdl"]["dram_read_pct"], np.arange(0.97, 1.08, 0.0025), np.arange(-200, 200, 2.0))
res["graph_pm_align"] = dict(corr=best[0], scale=best[1], offset_us=best[2], dur_us=gdur, n=gN, mean_rd_pct=float(grd.mean()))
graph_pm_fold = fold_series(gt, grd, gs * best[1] + best[2], 48.0)
graph_pm_ten = fold_series(gt, gpm["sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed"], gs * best[1] + best[2], 48.0)

# ---- figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
x = np.arange(PER) + 0.5
C = {"graph-pdl": "#1f77b4", "graph-nopdl": "#ff7f0e", "mk": "#2ca02c"}
def shade(ax, bounds, name):
    for i in range(9):
        if i % 2 == 0: ax.axvspan(bounds[i], bounds[i + 1], color="0.93", zorder=0)
        ax.text((bounds[i] + bounds[i + 1]) / 2, 1.0, PHASES[i][0], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
for ax, (nm, prof, bnds) in zip(axs, [
        ("graph-pdl (nsys, 10 us samples folded over 1440 layers)", folds["graph-pdl"]["dram_read_pct"], np.concatenate([[0], gr["graph-pdl"]["end_rel"]])),
        ("graph-nopdl (nsys)", folds["graph-nopdl"]["dram_read_pct"], np.concatenate([[0], gr["graph-nopdl"]["end_rel"]])),
        ("megakernel k16 (nsys, same instrument, stamps of last rep)", folds["mk"]["dram_read_pct"], mk_bounds)]):
    key = "graph-pdl" if "pdl (" in nm else ("graph-nopdl" if "nopdl" in nm else "mk")
    ax.plot(x, prof[:PER], color=C[key], lw=1.8); ax.fill_between(x, 0, prof[:PER], color=C[key], alpha=0.2)
    shade(ax, bnds, nm); ax.set_ylabel("DRAM read, % of 8 TB/s"); ax.set_title(nm, fontsize=9, loc="left", pad=14)
    ax.set_ylim(0, 40); ax.set_xlim(0, PER)
axs[-1].set_xlabel("time since layer start (us); shaded bands = phases A..I")
fig.suptitle("DRAM read throughput over one decoder layer (mean of all layers/steps)", y=0.995)
fig.tight_layout(); fig.savefig(OUT + "dram_over_layer.png", dpi=140)
np.savez(OUT + "folds.npz", **{f"{a}_{k}": v for a, d in folds.items() for k, v in d.items() if isinstance(v, np.ndarray)},
         mk_pm_rd=mk_fold["rd"], mk_pm_wr=mk_fold["wr"], mk_pm_ctas=mk_fold["ctas"], graph_pm_rd=graph_pm_fold)
json.dump(res, open(OUT + "summary.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# ---- print tables
print("phase  mk(us) gpdl c2c  gnopdl c2c | lead(pdl) | ncu own us | ideal MB | mk_obs MB | gpdl nsysObs MB")
ib = ideal_bytes()
for i, p in enumerate(PHASES):
    print(f"{p:7s} {mkp[p]:6.2f} {gr['graph-pdl']['c2c'][p]:6.2f} {gr['graph-nopdl']['c2c'][p]:6.2f} | {gr['graph-pdl']['lead'][p]:5.2f} | {ncu[p]['us']:6.2f} | {ib[p]/1e6:6.2f} | {mk_obs[i]/1e6:6.2f} | {tot_rd_graph['graph-pdl']['per_phase_B'][p]/1e6:6.2f}")
print("sum", sum(mkp.values()), sum(gr['graph-pdl']['c2c'].values()), sum(gr['graph-nopdl']['c2c'].values()))
print("windows", {a: (np.mean(g['window']), np.mean(g['prologue']), np.mean(g['layers_window'])) for a, g in gr.items()})
print("mk stamps prologue/tail", mk0['prologue_us'], mk0['tail_us_per_step'], "stage2 nsys:", mk2['prologue_us'], mk2['tail_us_per_step'])
print(json.dumps({k: res[k] for k in ["mk_pm", "graph_pm_align", "mk_period_us", "mk_nsys_per_layer_MB"]}, indent=1, default=float))
print({a: v["per_layer_MB"] for a, v in tot_rd_graph.items()})
print("mk pm per phase", {p: (round(v['ctas_active'], 1), round(v['tensor_pct'], 1)) for p, v in res['mk_pm_per_phase'].items()})
