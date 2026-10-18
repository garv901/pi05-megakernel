"""Build per_phase.csv, per_layer_phase.csv and phase_timeline.png from the analysis npz files.
usage: python -m megabench.sota.pi05.probe.build_tables"""
import csv, json, os
import numpy as np
from .phases import PHASES
P = os.path.dirname(__file__); R = f"{P}/runs/ncu_phase"; AN = f"{R}/analysis"
SM = 148
val = json.load(open(f"{AN}/mk_pm_validation_raw.json"))
PEAKS = [v for r in val["reps"].values() for v in [r.get("implied_peak_TBps")] if v] if False else None
NCU_PEAK = 7.645e3  # GB/s (B/ns), implied PM total vs kernel-level dram bytes
NOM = 8.0e3
STALLS = ["barrier", "membar", "wait", "long_scoreboard", "short_scoreboard", "mio_throttle", "lg_throttle", "math_pipe_throttle",
          "no_instructions", "branch_resolving", "dispatch_stall", "drain", "misc", "sleeping", "tex_throttle"]
MET = [("dram_rd_pct", ""), ("dram_wr_pct", ""), ("l2_pct", ""), ("l2_hit_pct", ""), ("sm_active_frac", ""), ("ctas_active", ""),
       ("issue_pct", ""), ("ipc_active", ""), ("tensor_pct", "")]


def load_mk(r):
    z = np.load(f"{AN}/mk_pm_windows_r{r}.npz"); d = {k: z[k] for k in z.files if z[k].ndim == 1 and len(z[k]) == len(z["phase"])}
    out = dict(phase=d["phase"], step=d["step"], layer=d["layer"], dur=(d["b"] - d["a"]) / 1e3)
    for m, _ in MET: out[m] = d["start_" + m]
    for s in STALLS: out["st_" + s] = d.get("stall_" + s, np.zeros(len(out["dur"])))
    out["st_selected"] = d["stall_selected"]; out["st_not_selected"] = d["stall_not_selected"]
    return out


def load_g(f):
    z = np.load(f); d = {k: z[k] for k in z.files if z[k].ndim == 1 and len(z[k]) == len(z["phase"])}
    out = dict(phase=d["phase"], step=d["step"], layer=d["layer"], dur=(d["b"] - d["a"]) / 1e3)
    for m, _ in MET: out[m] = d[m]
    for s in STALLS: out["st_" + s] = d.get("stall_" + s, np.zeros(len(out["dur"])))
    out["st_selected"] = d["stall_selected"]; out["st_not_selected"] = d["stall_not_selected"]
    return out


SYS = {
    "mk": [load_mk(r) for r in (1, 2, 3)],
    "graph-pdl": [load_g(f"{AN}/graph_pm_graph-pdl.npz"), load_g(f"{AN}/graph_pm_graph-pdl_r2.npz"), load_g(f"{AN}/graph_pm_graph-pdl_r3.npz")],
    "graph-nopdl": [load_g(f"{AN}/graph_pm_graph-nopdl.npz")],
}
ORDER = {"mk": ["P1", "P2", "P3", "P4", "P5"] + PHASES + ["TAIL"], "graph-pdl": ["PRO"] + PHASES, "graph-nopdl": ["PRO"] + PHASES}


def agg(d, sel):
    """duration-weighted mean over the selected windows (overlap-weighted PM average), plus p10/p90 of per-window values."""
    w = d["dur"][sel]; o = dict(n=int(sel.sum()), dur_mean=w.mean(), dur_p10=np.percentile(w, 10), dur_p90=np.percentile(w, 90), dur_sum=w.sum())
    for m, _ in MET:
        v = d[m][sel]; o[m] = float((v * w).sum() / w.sum()); o[m + "_p10"] = np.percentile(v, 10); o[m + "_p90"] = np.percentile(v, 90)
    st = {s: d["st_" + s][sel].sum() for s in STALLS}; tot = sum(st.values())
    o["stalls"] = {s: v / tot for s, v in st.items()} if tot > 0 else {}
    return o


rows = []
for sysn, reps in SYS.items():
    for ph in ORDER[sysn]:
        per = [agg(d, d["phase"] == ph) for d in reps]
        o = {"system": sysn, "phase": ph, "n_windows": per[0]["n"], "reps": len(per)}
        # prologue stamps / PRO / TAIL: PRO is the sum over 18 nodes; report sum duration
        if ph == "PRO":
            o["dur_us"] = np.mean([p["dur_sum"] / 10 if False else p["dur_sum"] for p in per]); o["dur_p10"] = o["dur_p90"] = float("nan")
        else:
            o["dur_us"] = np.mean([p["dur_mean"] for p in per]); o["dur_p10"] = np.mean([p["dur_p10"] for p in per]); o["dur_p90"] = np.mean([p["dur_p90"] for p in per])
        for m, _ in MET:
            o[m] = np.mean([p[m] for p in per]); o[m + "_p10"] = np.mean([p[m + "_p10"] for p in per]); o[m + "_p90"] = np.mean([p[m + "_p90"] for p in per])
            o[m + "_repsd"] = np.std([p[m] for p in per])
        o["dram_rd_GBps"] = o["dram_rd_pct"] / 100 * NCU_PEAK; o["dram_wr_GBps"] = o["dram_wr_pct"] / 100 * NCU_PEAK
        o["dram_rd_pct_of_8TB"] = o["dram_rd_GBps"] / NOM * 100; o["dram_wr_pct_of_8TB"] = o["dram_wr_GBps"] / NOM * 100
        o["dram_rd_MB_per_window"] = o["dram_rd_GBps"] * o["dur_us"] * 1e-3; o["dram_wr_MB_per_window"] = o["dram_wr_GBps"] * o["dur_us"] * 1e-3
        o["sm_active_pct"] = o["sm_active_frac"] * 100
        st = {s: np.mean([p["stalls"].get(s, 0) for p in per]) for s in STALLS}
        top = sorted(st.items(), key=lambda kv: -kv[1])[:3]
        o["top_stalls"] = ";".join(f"{k}={v*100:.0f}%" for k, v in top)
        rows.append(o)

# PTRACE barrier tail share (mk only; from stage3 summary)
pt = json.load(open(f"{P}/runs/analysis/ptrace_summary.txt"))
for o in rows:
    k = pt.get(o["phase"]) if o["system"] == "mk" else None
    o["ptrace_interval_us"] = k["interval_us"] if k else float("nan")
    o["ptrace_tail_share_pct"] = (1 - k["slowest_cta_done_us"] / k["interval_us"]) * 100 if k else float("nan")
    o["ptrace_barrier_exit_us"] = k["barrier_exit_latency_us"] if k else float("nan")

cols = ["system", "phase", "n_windows", "reps", "dur_us", "dur_p10", "dur_p90", "dram_rd_pct", "dram_rd_pct_p10", "dram_rd_pct_p90", "dram_wr_pct",
        "dram_rd_GBps", "dram_wr_GBps", "dram_rd_pct_of_8TB", "dram_wr_pct_of_8TB", "dram_rd_MB_per_window", "dram_wr_MB_per_window",
        "l2_pct", "l2_hit_pct", "sm_active_pct", "ctas_active", "issue_pct", "ipc_active", "tensor_pct", "dram_rd_pct_repsd", "top_stalls",
        "ptrace_interval_us", "ptrace_tail_share_pct", "ptrace_barrier_exit_us"]
with open(f"{P}/per_phase.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for o in rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k in cols})

# per layer x phase (mean over the 10 steps; first rep for mk/graph-pdl rep1)
lrows = []
for sysn in ("mk", "graph-pdl", "graph-nopdl"):
    reps = SYS[sysn]
    for l in range(18):
        for ph in PHASES:
            ps = []
            for d in reps:
                sel = (d["phase"] == ph) & (d["layer"] == l)
                ps.append(agg(d, sel))
            o = dict(system=sysn, layer=l, phase=ph, dur_us=np.mean([p["dur_mean"] for p in ps]))
            for m in ("dram_rd_pct", "dram_wr_pct", "l2_hit_pct", "ctas_active", "issue_pct"): o[m] = np.mean([p[m] for p in ps])
            o["sm_active_pct"] = np.mean([p["sm_active_frac"] for p in ps]) * 100
            lrows.append(o)
lc = ["system", "layer", "phase", "dur_us", "dram_rd_pct", "dram_wr_pct", "sm_active_pct", "ctas_active", "issue_pct", "l2_hit_pct"]
with open(f"{P}/per_layer_phase.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=lc); w.writeheader()
    for o in lrows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()})
json.dump(rows, open(f"{AN}/per_phase_rows.json", "w"), default=float)

# figure: mean per-layer phase timeline (mk vs graph+PDL), plus layer x phase heatmap of DRAM read
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 1, figsize=(11, 12))
cmap = plt.get_cmap("tab10")
for a_, (m, lab) in enumerate((("dram_rd_pct", "DRAM read % of ncu peak"), ("sm_active_pct", "SM active %"), ("ctas_active", "CTAs active (148 SMs)"))):
    for sysn, ls, lw in (("mk", "-", 2.0), ("graph-pdl", "--", 1.6)):
        t = 0; X = []; Y = []
        for ph in PHASES:
            o = next(r for r in rows if r["system"] == sysn and r["phase"] == ph)
            X += [t, t + o["dur_us"]]; Y += [o[m], o[m]]
            if a_ == 0: ax[0].text(t + o["dur_us"] / 2, 1.0, ph[0], ha="center", va="bottom", fontsize=8, color="C0" if sysn == "mk" else "C1", transform=ax[0].get_xaxis_transform() if False else ax[0].transData)
            t += o["dur_us"]
        ax[a_].step(X, Y, ls, lw=lw, label=f"{sysn} (layer = {t:.1f} us)", color="C0" if sysn == "mk" else "C1")
    ax[a_].set_ylabel(lab); ax[a_].legend(fontsize=8); ax[a_].set_xlabel("time within one mean layer (us); phases A..I concatenated, each system on its own clock")
ax[0].set_title("Per-phase ncu PM means (duration-weighted over 180 layer windows); labels = phase letters (blue mk, orange graph)")
for k, sysn in enumerate(("mk", "graph-pdl")):
    Mx = np.array([[next(o for o in lrows if o["system"] == sysn and o["layer"] == l and o["phase"] == ph)["dram_rd_pct"] for ph in PHASES] for l in range(18)])
    if k == 0: im = ax[3].imshow(Mx, aspect="auto")
ax[3].set_title("mk: DRAM read % by layer index (rows) x phase (cols)"); ax[3].set_xticks(range(9)); ax[3].set_xticklabels([p[0] for p in PHASES]); ax[3].set_ylabel("layer"); plt.colorbar(im, ax=ax[3])
plt.tight_layout(); plt.savefig(f"{R}/phase_timeline.png", dpi=110)
print("ok", len(rows), len(lrows))
