"""Prefetch-aware MBU accounting for the megakernel (mk) vs graph+PDL, from ncu PM sampling (1 us, dram__read_throughput).
usage: python -m megabench.sota.pi05.probe.mbu_prefetch_analyze OUT_DIR PF_DIR [PLAIN_DIR]
  PF_DIR   : dir with mk_pm_pf_r{1,2,3}.ncu-rep + *_ts_*.npy (PI05_PTRACE+PI05_PFTRACE build; saved buffer holds stamps+ptrace+pf)
  PLAIN_DIR: dir with the original untraced mk_pm_triage_warp_r{1,2,3} (default: ncu_artifacts/mk_pm)
Writes OUT_DIR/{results.json, mbu_prefetch.csv, per_instance_streams.csv, fig_*.png}."""
import csv, glob, json, os, sys
import numpy as np
from . import ncu_pm as N
from .mbu_lib import *

OUT, PFD = sys.argv[1], sys.argv[2]
PLAIN = sys.argv[3] if len(sys.argv) > 3 else "megabench/sota/pi05/probe/ncu_artifacts/mk_pm"
os.makedirs(OUT, exist_ok=True)
rows = []          # long CSV


def add(table, system, item, metric, vals, unit=""):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        rows.append([table, system, item, metric, unit, 0, "", "", "", "", ""])
        return
    rows.append([table, system, item, metric, unit, len(v), f"{v.mean():.4g}", f"{np.percentile(v,10):.4g}",
                 f"{np.median(v):.4g}", f"{np.percentile(v,90):.4g}", f"{v.std():.4g}"])


def pct(bytes_, ns):          # % of PEAK for bytes over ns
    return bytes_ / (ns * 1e-9) / PEAK * 100.0


res = {}
# ======================================================================================= timeline-level (a)(b)(c)
def timeline_mk(tag, R):
    F = F_of(R["t"], R["v"]); ts = R["ts"]
    W = layer_windows_mk(ts)
    by = np.array([mb(F, a, b) for s, l, a, b in W]); du = np.array([b - a for s, l, a, b in W], float)
    add("a_layer", tag, "layer", "MB", by / 1e6, "MB"); add("a_layer", tag, "layer", "us", du / 1e3, "us")
    add("a_layer", tag, "layer", "TBps", by / (du * 1e-9) / 1e12, "TB/s")
    add("a_layer", tag, "layer", "pct_peak7.645", by / (du * 1e-9) / PEAK * 100, "%")
    add("a_layer", tag, "layer", "pct_8TBps", by / (du * 1e-9) / NOMINAL * 100, "%")
    add("a_layer", tag, "layer", "pct_6.5TBps_copy", by / (du * 1e-9) / COPY_BW * 100, "%")
    Fw = F_of(R["t"], R["vw"]); wr = np.array([mb(Fw, a, b) for s, l, a, b in W])
    add("a_layer", tag, "layer", "dram_write_MB", wr / 1e6, "MB")
    add("a_layer", tag, "layer", "rd+wr_pct_peak7.645", (by + wr) / (du * 1e-9) / PEAK * 100, "%")
    add("a_kernel", tag, "whole kernel", "MB", [mb(F, R["S"], R["E"]) / 1e6], "MB")
    add("a_kernel", tag, "whole kernel", "TBps", [mb(F, R["S"], R["E"]) / ((R["E"] - R["S"]) * 1e-9) / 1e12], "TB/s")
    bf = {th: [] for th in (10, 25, 50)}; bw = {th: [] for th in (10, 25, 50)}
    for s, l, a, b in W:
        d = busy_fracs(R["t"], R["v"], a, b)
        for th in bf:
            bf[th].append(d[th][0]); bw[th].append(d[th][1])
    for th in bf:
        add("b_busy", tag, f">={th}% of peak", "time_fraction", bf[th], "frac")
        add("b_busy", tag, f">={th}% of peak", "mean_rd_pct_while_busy", [x for x in bw[th] if x > 0], "% of peak")
    # (c) per-op windows (mk: stamp windows)
    for i, p in enumerate(PHASES):
        mins, means, maxs, ovl = [], [], [], []
        for s in range(NSTEP):
            for l in range(L):
                pw = phase_windows_mk(ts, s, l)[i]
                lo, mean, hi = sample_stats(R["t"], R["v"], pw[1], pw[2])
                mins.append(lo); means.append(mean); maxs.append(hi); ovl.append(mb(F, pw[1], pw[2]) / ((pw[2] - pw[1]) * 1e-9) / PEAK * 100)
        add("c_op", tag, p, "sample_min", mins, "%"); add("c_op", tag, p, "sample_mean", means, "%")
        add("c_op", tag, p, "sample_max", maxs, "%"); add("c_op", tag, p, "overlap_weighted_mean", ovl, "%")
    return W


def timeline_graph(tag, G):
    F = F_of(G["t"], G["v"]); W = layer_windows_graph(G); idx = graph_index(G)
    by = np.array([mb(F, a, b) for s, l, a, b in W]); du = np.array([b - a for s, l, a, b in W], float)
    add("a_layer", tag, "layer", "MB", by / 1e6, "MB"); add("a_layer", tag, "layer", "us", du / 1e3, "us")
    add("a_layer", tag, "layer", "TBps", by / (du * 1e-9) / 1e12, "TB/s")
    add("a_layer", tag, "layer", "pct_peak7.645", by / (du * 1e-9) / PEAK * 100, "%")
    add("a_layer", tag, "layer", "pct_8TBps", by / (du * 1e-9) / NOMINAL * 100, "%")
    add("a_layer", tag, "layer", "pct_6.5TBps_copy", by / (du * 1e-9) / COPY_BW * 100, "%")
    Fw = F_of(G["t"], G["vw"]); wr = np.array([mb(Fw, a, b) for s, l, a, b in W])
    add("a_layer", tag, "layer", "dram_write_MB", wr / 1e6, "MB")
    add("a_layer", tag, "layer", "rd+wr_pct_peak7.645", (by + wr) / (du * 1e-9) / PEAK * 100, "%")
    bf = {th: [] for th in (10, 25, 50)}; bw = {th: [] for th in (10, 25, 50)}
    for s, l, a, b in W:
        d = busy_fracs(G["t"], G["v"], a, b)
        for th in bf:
            bf[th].append(d[th][0]); bw[th].append(d[th][1])
    for th in bf:
        add("b_busy", tag, f">={th}% of peak", "time_fraction", bf[th], "frac")
        add("b_busy", tag, f">={th}% of peak", "mean_rd_pct_while_busy", [x for x in bw[th] if x > 0], "% of peak")
    for p in PHASES:
        mins, means, maxs, ovl = [], [], [], []
        for s in range(NSTEP):
            for l in range(L):
                k = idx[(s, l, p)]
                a = G["en"][k - 1]; b = G["en"][k]       # completion-to-completion
                lo, mean, hi = sample_stats(G["t"], G["v"], a, b)
                mins.append(lo); means.append(mean); maxs.append(hi); ovl.append(mb(F, a, b) / ((b - a) * 1e-9) / PEAK * 100)
        add("c_op", tag, p, "sample_min", mins, "%"); add("c_op", tag, p, "sample_mean", means, "%")
        add("c_op", tag, p, "sample_max", maxs, "%"); add("c_op", tag, p, "overlap_weighted_mean", ovl, "%")
    return W


# ======================================================================================= graph GEMM bursts (d)
GRAPH_GEMM = {"A_qkv": 5242880, "D_o": 4194304, "F_gu": 16777216 + 2 * 3 * 1024 * 4, "H_down": 8388608}


def graph_bursts(tag, G, THR=5.0):
    F = F_of(G["t"], G["v"]); idx = graph_index(G); t, v = G["t"], G["v"]
    out = {p: [] for p in GRAPH_GEMM}
    for s in range(NSTEP):
        for l in range(L):
            for p, ideal in GRAPH_GEMM.items():
                k = idx[(s, l, p)]; st, en = G["st"][k], G["en"][k]
                lo, hi = st - 4000, en + 1000
                m = np.where((t >= lo) & (t <= hi))[0]
                if len(m) == 0 or v[m].max() < THR:
                    continue
                j0 = m[np.argmax(v[m])]
                a = j0
                while a > 0 and v[a - 1] >= THR: a -= 1
                b = j0
                while b < len(v) - 1 and v[b + 1] >= THR: b += 1
                ra, rb = t[a], t[b] + 1000.0
                tot = mb(F, ra, rb)
                t5 = t_at_cum(F, ra, rb, 0.05 * tot); t95 = t_at_cum(F, ra, rb, 0.95 * tot)
                d = t95 - t5
                out[p].append(dict(step=s, layer=l, run_bytes=tot, run_us=(rb - ra) / 1e3, t5_us=(t5 - st) / 1e3, dur_us=d / 1e3,
                                   bw_tbps=0.9 * tot / (d * 1e-9) / 1e12, run_bw_tbps=tot / ((rb - ra) * 1e-9) / 1e12,
                                   peak_pct=float(v[a:b + 1].max()), ideal_over_dur_tbps=ideal / (d * 1e-9) / 1e12 * 0.9 / 0.9,
                                   kernel_us=(en - st) / 1e3, t95_minus_kstart_us=(t95 - st) / 1e3, t95_minus_kend_us=(t95 - en) / 1e3))
    for p, lst in out.items():
        for key in ("run_bytes", "run_us", "dur_us", "bw_tbps", "run_bw_tbps", "peak_pct", "kernel_us", "t5_us", "t95_minus_kstart_us", "t95_minus_kend_us"):
            add("d_graph_gemm_burst", tag, p, key, [x[key] for x in lst], {"run_bytes": "B", "peak_pct": "%"}.get(key, "us" if "us" in key else "TB/s"))
        add("d_graph_gemm_burst", tag, p, "pct_peak_eff", [x["bw_tbps"] * 1e12 / PEAK * 100 for x in lst], "%")
        add("d_graph_gemm_burst", tag, p, "ideal_bytes_over_burst_dur_TBps", [GRAPH_GEMM[p] / (x["dur_us"] * 1e-6) / 1e12 / 0.9 * 0.9 for x in lst], "TB/s")
    return out


# ======================================================================================= mk streams (PF data)
def mk_streams(tag, R):
    F = F_of(R["t"], R["v"])
    ts, pt, pf = pf_split(R["raw"], R["base"])
    ts = R["ts"]
    GROUPS = [("B", ["B_WoKV"]), ("CD", ["C_WgWu", "D_Wd"]), ("G", ["G_QKV"])]    # DRAM landing is resolved per group (C and D interleave)
    tot_model = float(sum(BYTES.values()))
    inst = []
    for s in range(NSTEP):
        for l in range(L):
            a0 = ts[mk_slot(s, l, 0)]; b0 = ts[mk_slot(s, l + 1, 0)] if l < L - 1 else ts[mk_slot(s, l, 9)]
            tot = mb(F, a0, b0)
            i_ = s * L + l
            iss = {}
            for si, st in enumerate(STREAMS):
                x = pf[i_, si]                                  # [148,8,2]
                b_ = x[..., 0][x[..., 0] > 0]; e_ = x[..., 1][x[..., 1] > 0]
                cs = np.where(x[..., 0] > 0, x[..., 0], np.iinfo(np.int64).max).min(axis=1)      # per-CTA first warp start
                cs = cs[cs < np.iinfo(np.int64).max]
                iss[st] = (b_.min(), e_.max(), np.percentile(cs, 10), np.percentile(cs, 50), np.percentile(cs, 90), cs.max()) if len(b_) and len(e_) else (np.nan,) * 6
            # DRAM landing per group, from the window-normalised cumulative bytes between the quiet gaps of the PM series (robust to
            # small unmodelled bytes): B: [A-in, gap before Wg issue]; CD: [gap, 1.5 us before G issue]; G: [that, next layer A-in / I end]
            tv, vv = R["t"], R["v"]
            def gap(lo, hi):
                m = np.where((tv >= lo) & (tv < hi))[0]
                return float(tv[m[np.argmin(vv[m])]]) + 500.0 if len(m) else float(lo)
            cutBC = gap(iss["B_WoKV"][0] + 1500, iss["C_WgWu"][0] + 500)
            cutCG = iss["G_QKV"][0] - 1500.0
            edges_ = {"B": (a0, cutBC), "CD": (cutBC, cutCG), "G": (cutCG, b0)}
            land = {}
            for gname, members in GROUPS:
                lo_, hi_ = edges_[gname]; gt = mb(F, lo_, hi_)
                land[gname] = tuple(t_at_cum(F, lo_, hi_, gt * q) for q in (0.05, 0.95, 0.999)) + (sum(BYTES[m] for m in members), gt)
            gof = {"B_WoKV": "B", "C_WgWu": "CD", "D_Wd": "CD", "G_QKV": "G"}
            for si, st in enumerate(STREAMS):
                by = BYTES[st]
                t5, t95, t999, gby, gmeas = land[gof[st]]
                ci = CONSUMER_SLOT[st]
                if ci == 9:
                    cslot = mk_slot(s, l + 1, 0) if l < L - 1 else (mk_slot(s + 1, 0, 0) if s < NSTEP - 1 else None)
                else:
                    cslot = mk_slot(s, l, ci)
                rec = dict(source=tag, step=s, layer=l, stream=st, bytes=by, group_bytes=gby, group_meas_bytes=gmeas, a0=a0, tot_layer_bytes=tot, issue_first=iss[st][0], issue_last=iss[st][1],
                           cta_start_p10=iss[st][2], cta_start_p50=iss[st][3], cta_start_p90=iss[st][4], cta_start_max=iss[st][5], t5=t5, t95=t95, t999=t999)
                if cslot is not None:
                    arr_ = pt[cslot, :, 0]; ex_ = pt[cslot, :, 1]
                    arr_ = arr_[arr_ > 0]; ex_ = ex_[ex_ > 0]
                    rec.update(demand_first=float(arr_.min()), demand_last=float(arr_.max()), main_start=float(np.median(ex_)),
                               main_start_blk0=float(ts[cslot]))
                    # consumer window for the re-read check: [median consumer barrier exit, min(next block-0 stamp, 1.5 us before the next other-stream issue)];
                    # DRAM bytes there are L2 misses of the consumer (or the tail of other traffic)
                    st0 = rec["main_start"]
                    if st == "B_WoKV":          # Wo is consumed by D, while the Wg/Wu/Wd burst may still be landing: start after it
                        st0 = max(st0, land["CD"][2])
                    nxt = ts[cslot + 1] if cslot + 1 < 1635 else ts[cslot] + 3000
                    other = [iss[o][0] - 1500 for o in STREAMS if o != st and np.isfinite(iss[o][0]) and iss[o][0] - 1500 > st0]
                    cw_end = min([nxt] + other)
                    rec.update(cw_start=float(st0), cw_end=float(cw_end), cons_dram_bytes=float(mb(F, st0, cw_end)) if cw_end > st0 else 0.0)
                m = (R["t"] >= t5 - 1000) & (R["t"] <= t95)
                rec["peak_pct"] = float(R["v"][m].max()) if m.any() else np.nan
                inst.append(rec)
    return inst


def summarize_streams(tag, inst):
    for st in STREAMS:
        r = [x for x in inst if x["stream"] == st]
        g = lambda k: np.array([x.get(k, np.nan) for x in r], float)
        a0 = g("a0")
        by = BYTES[st]; gby = g("group_bytes")
        dur = g("t95") - g("t5")
        add("d_mk_stream", tag, st, "bytes_model", [by / 1e6], "MB")
        add("d_mk_stream", tag, st, "issue_first_us_from_layer_start", (g("issue_first") - a0) / 1e3, "us")
        add("d_mk_stream", tag, st, "issue_last_us_from_layer_start", (g("issue_last") - a0) / 1e3, "us")
        add("d_mk_stream", tag, st, "issue_span_us", (g("issue_last") - g("issue_first")) / 1e3, "us")
        add("d_mk_stream", tag, st, "dram_t5_us_from_layer_start", (g("t5") - a0) / 1e3, "us")
        add("d_mk_stream", tag, st, "dram_t95_us_from_layer_start", (g("t95") - a0) / 1e3, "us")
        add("d_mk_stream", tag, st, "burst_dur_t5_t95_us", dur / 1e3, "us")
        add("d_mk_stream", tag, st, "eff_TBps_burst(0.9*group_bytes/dur)", 0.9 * gby / (dur * 1e-9) / 1e12, "TB/s")
        add("d_mk_stream", tag, st, "eff_pct_peak_burst", 0.9 * gby / (dur * 1e-9) / PEAK * 100, "%")
        add("d_mk_stream", tag, st, "cta_start_p10_us", (g("cta_start_p10") - a0) / 1e3, "us")
        add("d_mk_stream", tag, st, "cta_start_p50_us", (g("cta_start_p50") - a0) / 1e3, "us")
        add("d_mk_stream", tag, st, "cta_start_p90_us", (g("cta_start_p90") - a0) / 1e3, "us")
        add("d_mk_stream", tag, st, "cta_start_last_us", (g("cta_start_max") - a0) / 1e3, "us")
        add("d_mk_stream", tag, st, "peak_1us_pct", g("peak_pct"), "%")
        add("d_mk_stream", tag, st, "issue_first_to_t95_us", (g("t95") - g("issue_first")) / 1e3, "us")
        add("d_mk_stream", tag, st, "issue_first_to_t95_TBps", gby / ((g("t95") - g("issue_first")) * 1e-9) / 1e12, "TB/s")
        add("d_mk_stream", tag, st, "issue_to_consumer_demand_us", (g("demand_first") - g("issue_first")) / 1e3, "us")
        add("d_mk_stream", tag, st, "eff_TBps_issue_to_consumer_demand", by / ((g("demand_first") - g("issue_first")) * 1e-9) / 1e12, "TB/s")
        add("d_mk_stream", tag, st, "pct_peak_issue_to_consumer_demand", by / ((g("demand_first") - g("issue_first")) * 1e-9) / PEAK * 100, "%")
        add("d_mk_stream", tag, st, "slack_demand_minus_t95_us", (g("demand_first") - g("t95")) / 1e3, "us")
        add("d_mk_stream", tag, st, "slack_main_minus_t95_us", (g("main_start") - g("t95")) / 1e3, "us")
        add("d_mk_stream", tag, st, "slack_demand_minus_issue_last_us", (g("demand_first") - g("issue_last")) / 1e3, "us")
        ok = np.isfinite(g("demand_first"))
        add("d_mk_stream", tag, st, "frac_landed_before_consumer_demand", ((g("t95") < g("demand_first"))[ok]).astype(float), "frac")
        add("d_mk_stream", tag, st, "consumer_window_dram_MB", g("cons_dram_bytes") / 1e6, "MB")
        add("d_mk_stream", tag, st, "consumer_window_dram_frac_of_stream", g("cons_dram_bytes") / by, "frac")
        add("d_mk_stream", tag, st, "consumer_window_us", (g("cw_end") - g("cw_start")) / 1e3, "us")
    rc = {(x["step"], x["layer"]): x for x in inst if x["stream"] == "C_WgWu"}
    rd = {(x["step"], x["layer"]): x for x in inst if x["stream"] == "D_Wd"}
    ks = sorted(rc)
    nm = "CD_Wg+Wu+Wd(combined)"; byt = BYTES["C_WgWu"] + BYTES["D_Wd"]
    a0 = np.array([rc[k]["a0"] for k in ks]); i0 = np.array([min(rc[k]["issue_first"], rd[k]["issue_first"]) for k in ks])
    i1 = np.array([max(rc[k]["issue_last"], rd[k]["issue_last"]) for k in ks]); t5 = np.array([rc[k]["t5"] for k in ks]); t95 = np.array([rc[k]["t95"] for k in ks])
    pk = np.array([rc[k]["peak_pct"] for k in ks])
    add("d_mk_stream", tag, nm, "bytes_model", [byt / 1e6], "MB")
    add("d_mk_stream", tag, nm, "issue_first_us_from_layer_start", (i0 - a0) / 1e3, "us")
    add("d_mk_stream", tag, nm, "issue_last_us_from_layer_start", (i1 - a0) / 1e3, "us")
    add("d_mk_stream", tag, nm, "dram_t5_us_from_layer_start", (t5 - a0) / 1e3, "us")
    add("d_mk_stream", tag, nm, "dram_t95_us_from_layer_start", (t95 - a0) / 1e3, "us")
    add("d_mk_stream", tag, nm, "burst_dur_t5_t95_us", (t95 - t5) / 1e3, "us")
    add("d_mk_stream", tag, nm, "eff_TBps_burst(0.9*bytes/dur)", 0.9 * byt / ((t95 - t5) * 1e-9) / 1e12, "TB/s")
    add("d_mk_stream", tag, nm, "eff_pct_peak_burst", 0.9 * byt / ((t95 - t5) * 1e-9) / PEAK * 100, "%")
    add("d_mk_stream", tag, nm, "peak_1us_pct", pk, "%")
    add("d_mk_stream", tag, nm, "issue_first_to_t95_us", (t95 - i0) / 1e3, "us")
    add("d_mk_stream", tag, nm, "issue_first_to_t95_TBps", byt / ((t95 - i0) * 1e-9) / 1e12, "TB/s")
    for gn, nm_ in (("B_WoKV", "B"), ("C_WgWu", "CD"), ("G_QKV", "G")):
        add("d_mk_stream", tag, gn, "group_window_dram_MB_measured(vs model group_bytes)", [x["group_meas_bytes"] / 1e6 for x in inst if x["stream"] == gn], "MB")
    add("d_mk_stream", tag, "all streams", "layer_total_MB(PM)", [x["tot_layer_bytes"] / 1e6 for x in inst if x["stream"] == "B_WoKV"], "MB")


def wpre_stats(tag, R):
    """WPRE first-tile loads (A, D, F, H): per barrier [arrive_c, exit_c] of each CTA is the window in which its first-tile weight loads are in flight."""
    ts, pt, pf = pf_split(R["raw"], R["base"]); F = F_of(R["t"], R["v"])
    names = {0: "A_in(qkv, WPRE)", 3: "D_in(o-proj, WPRE)", 5: "F_in(gate/up, WPRE)", 7: "H_in(down, WPRE)"}
    for k, nm in names.items():
        w_med, w_p90, w_max, dram, span = [], [], [], [], []
        for s in range(NSTEP):
            for l in range(L):
                sl = mk_slot(s, l, k)
                ar = pt[sl, :, 0]; ex = pt[sl, :, 1]; m = (ar > 0) & (ex > 0)
                w = (ex[m] - ar[m]) / 1e3
                w_med.append(np.median(w)); w_p90.append(np.percentile(w, 90)); w_max.append(w.max())
                span.append((np.median(ex[m]) - ar[m].min()) / 1e3)
                dram.append(mb(F, ar[m].min(), np.median(ex[m])) / 1e6)
        add("e_wpre", tag, nm, "per_CTA_arrive_to_exit_median_us", w_med, "us"); add("e_wpre", tag, nm, "per_CTA_arrive_to_exit_p90_us", w_p90, "us")
        add("e_wpre", tag, nm, "per_CTA_arrive_to_exit_max_us", w_max, "us")
        add("e_wpre", tag, nm, "first_arrive_to_median_exit_us", span, "us")
        add("e_wpre", tag, nm, "dram_MB_in_[first_arrive, median_exit]", dram, "MB")


# ======================================================================================= run
res["notes"] = "see mbu_prefetch.csv"
mk_plain = {}
for r in (1, 2, 3):
    R = load_mk_rep(f"{PLAIN}/mk_pm_triage_warp_r{r}.ncu-rep", glob.glob(f"{PLAIN}/mk_pm_triage_warp_r{r}_ts_*.npy"))
    mk_plain[r] = R
    timeline_mk(f"mk_untraced_r{r}", R)
mk_pf, streams = {}, {}
for r in (1, 2, 3):
    fs = glob.glob(f"{PFD}/mk_pm_pf_r{r}.ncu-rep")
    if not fs:
        continue
    R = load_mk_rep(fs[0], glob.glob(f"{PFD}/mk_pm_pf_r{r}_ts_*.npy"))
    mk_pf[r] = R
    timeline_mk(f"mk_pftrace_r{r}", R)
    wpre_stats(f"mk_pftrace_r{r}", R)
    streams[r] = mk_streams(f"mk_pftrace_r{r}", R)
    summarize_streams(f"mk_pftrace_r{r}", streams[r])
allinst = [x for r in streams for x in streams[r]]
if allinst:
    summarize_streams("mk_pftrace_all", allinst)
graphs = {}; gb = {}
for name, lst in GRAPH_REPS.items():
    for i, (rep, npz) in enumerate(lst):
        G = load_graph_rep(rep, npz)
        graphs[(name, i)] = G
        timeline_graph(f"{name}_r{i+1}", G)
        gb[(name, i)] = graph_bursts(f"{name}_r{i+1}", G)

with open(f"{OUT}/mbu_prefetch.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["table", "system", "item", "metric", "unit", "n", "mean", "p10", "median", "p90", "std"]); w.writerows(rows)
if allinst:
    keys = list(allinst[0].keys())
    with open(f"{OUT}/per_instance_streams.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for x in allinst for k in x})); w.writeheader(); w.writerows(allinst)

import pickle
pickle.dump(dict(allinst=allinst), open(f"{OUT}/_inst.pkl", "wb"))
print("wrote", OUT, len(rows), "rows")

# ======================================================================================= figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888", "axes.linewidth": 0.6})
INK, MUTED = "#222222", "#777777"
C_MK, C_GR = "#1f5fa8", "#b5541c"
SCOL = {"B_WoKV": "#7a3e9d", "C_WgWu": "#1b8a6b", "D_Wd": "#c79a00", "G_QKV": "#c0392b"}
SLAB = {"B_WoKV": "Wo+KV 4.3MB", "C_WgWu": "Wg+Wu 16.8MB", "D_Wd": "Wd 8.4MB", "G_QKV": "next QKV 5.2MB"}
SHIFT_NOTE = f"PM series delayed by +{PM_SHIFT_NS/1e3:.1f} us (causal alignment)" if PM_SHIFT_NS else "raw PM timestamps (no lead correction)"


def stepline(ax, t, v, a0, color, lo, hi, lw=1.1):
    m = (t >= a0 + lo) & (t < a0 + hi)
    tt = (t[m] - a0) / 1e3
    ax.step(np.append(tt, tt[-1] + 1.0), np.append(v[m], v[m][-1]), where="post", color=color, lw=lw)


def one_layer_fig(s=5, l=8, r=1):
    R = mk_pf.get(r) or mk_plain[r]
    G = graphs[("graph-pdl", 0)]; idx = graph_index(G)
    ts = R["ts"]
    a0 = ts[mk_slot(s, l, 0)]; ga0 = G["st"][idx[(s, l, "A_qkv")]]
    lo, hi = -3000.0, 48000.0
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    # mk
    A = ax[0]
    stepline(A, R["t"], R["v"], a0, C_MK, lo, hi)
    for i, p in enumerate(PHASES):
        x = (ts[mk_slot(s, l, i)] - a0) / 1e3
        A.axvline(x, color=MUTED, lw=0.5, ls=":")
        x2 = (ts[mk_slot(s, l, i + 1)] - a0) / 1e3
        A.text((x + x2) / 2, 101, p[0], ha="center", va="bottom", color=INK, fontsize=8)
    A.set_ylim(-34, 100); A.set_ylabel("DRAM read, % of 7.645 TB/s")
    if mk_pf and r in mk_pf:
        ins = {x["stream"]: x for x in streams[r] if x["step"] == s and x["layer"] == l}
        for k, st in enumerate(STREAMS):
            x = ins[st]; y = -6 - 6.5 * k
            A.plot([(x["issue_first"] - a0) / 1e3, (x["issue_last"] - a0) / 1e3], [y, y], color=SCOL[st], lw=4, solid_capstyle="butt")
            A.plot([(x["t5"] - a0) / 1e3, (x["t95"] - a0) / 1e3], [y - 2.6, y - 2.6], color=SCOL[st], lw=1.2)
            A.plot([(x["demand_first"] - a0) / 1e3], [y], marker="v", color=SCOL[st], ms=6, mec="white", mew=0.5, ls="none")
            A.plot([(x["main_start"] - a0) / 1e3], [y], marker="|", color=SCOL[st], ms=9, ls="none")
            A.text(-2.8, y, SLAB[st], ha="left", va="center", fontsize=7, color=INK)
    A.set_title(f"mk (PM-sampled run r{r}), step {s} layer {l}; op letters A..I = block-0 barrier stamps; {SHIFT_NOTE}\nthick bar = prefetch issue span (first CTA start .. last warp end, PFTRACE); thin line = DRAM landing 5..95%; v = consumer first demand (barrier arrive); | = consumer main start (median barrier exit)", loc="left", fontsize=8, color=INK, pad=14)
    # graph
    B = ax[1]
    stepline(B, G["t"], G["v"], ga0, C_GR, lo, hi)
    cols = {"A_qkv": "#6b6b6b", "D_o": "#6b6b6b", "F_gu": "#6b6b6b", "H_down": "#6b6b6b"}
    for p in PHASES:
        k = idx[(s, l, p)]; x0 = (G["st"][k] - ga0) / 1e3; x1 = (G["en"][k] - ga0) / 1e3
        gem = p in GRAPH_GEMM
        yy = -4 - 4.2 * PHASES.index(p) / 1.0 * 0.0 - (PHASES.index(p) % 3) * 4.5
        B.plot([x0, x1], [yy - 2, yy - 2], color=INK if gem else MUTED, lw=3 if gem else 2, solid_capstyle="butt")
        B.text((x0 + x1) / 2, yy - 5.2, p[0], ha="center", va="top", fontsize=6.5, color=INK)
    B.set_ylim(-26, 100); B.set_ylabel("DRAM read, % of 7.645 TB/s")
    B.set_xlabel("time since layer start (us); graph-pdl layer start = qkv GEMM kernel start")
    B.set_title(f"graph + PDL (ncu graph-profiling, rep 1), step {s} layer {l}: bars = kernel start..end (GEMMs A, D, F, H bold)", loc="left", fontsize=8.5, color=INK, pad=14)
    for a in ax:
        a.grid(axis="y", color="#ddd", lw=0.4); a.set_xlim(-3, 48)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_one_layer.png", dpi=170)
    plt.close(fig)


def fold(t, v, starts, lo=-3.0, hi=48.0):
    nb = int(hi - lo); vals = [[] for _ in range(nb)]
    for a0 in starts:
        m = (t >= a0 + lo * 1e3) & (t < a0 + hi * 1e3)
        b = np.floor((t[m] - a0) / 1e3 - lo).astype(int)
        for bb, x in zip(b, v[m]):
            if 0 <= bb < nb: vals[bb].append(x)
    mean = np.array([np.mean(x) if x else 0 for x in vals]); p10 = np.array([np.percentile(x, 10) if x else 0 for x in vals])
    p90 = np.array([np.percentile(x, 90) if x else 0 for x in vals])
    return np.arange(nb) + lo, mean, p10, p90


def fold_fig(r=1):
    R = mk_pf.get(r) or mk_plain[r]
    G = graphs[("graph-pdl", 0)]; idx = graph_index(G)
    ts = R["ts"]
    starts = np.array([ts[mk_slot(s, l, 0)] for s in range(NSTEP) for l in range(L)], float)
    gst = np.array([G["st"][idx[(s, l, "A_qkv")]] for s in range(NSTEP) for l in range(L)], float)
    x, m, p10, p90 = fold(R["t"], R["v"], starts)
    gx, gm, g10, g90 = fold(G["t"], G["v"], gst)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    for a, (xx, mm, a10, a90, c, ttl) in zip(ax, [(x, m, p10, p90, C_MK, f"mk, fold over all 180 layers (1 us bins from layer start; line = mean, band = p10..p90 of samples); {SHIFT_NOTE}\nmarkers = medians over layers: thick = issue span, thin = DRAM landing 5..95%, v = first consumer demand, | = consumer main start"),
                                                   (gx, gm, g10, g90, C_GR, "graph + PDL, same fold (layer start = qkv kernel start)")]):
        a.fill_between(np.append(xx, xx[-1] + 1), np.append(a10, a10[-1]), np.append(a90, a90[-1]), step="post", color=c, alpha=0.18, lw=0)
        a.step(np.append(xx, xx[-1] + 1), np.append(mm, mm[-1]), where="post", color=c, lw=1.3)
        a.set_title(ttl, loc="left", fontsize=8.5, color=INK, pad=14); a.set_ylabel("DRAM read, % of 7.645 TB/s"); a.grid(axis="y", color="#ddd", lw=0.4)
        a.set_ylim(-30, 100); a.set_xlim(-3, 48)
    A = ax[0]
    mean_off = np.mean([[ts[mk_slot(s, l, i)] - ts[mk_slot(s, l, 0)] for i in range(10)] for s in range(NSTEP) for l in range(L - 1)], axis=0) / 1e3
    for i, p in enumerate(PHASES):
        A.axvline(mean_off[i], color=MUTED, lw=0.5, ls=":")
        A.text((mean_off[i] + mean_off[i + 1]) / 2, 101, p[0], ha="center", va="bottom", fontsize=8)
    if allinst and r in streams:
        for k, st in enumerate(STREAMS):
            z = [q for q in streams[r] if q["stream"] == st]
            g_ = lambda key: np.array([q.get(key, np.nan) for q in z], float)
            a0_ = g_("a0")
            y = -6 - 6.5 * k
            i0, i1 = np.nanmedian(g_("issue_first") - a0_) / 1e3, np.nanmedian(g_("issue_last") - a0_) / 1e3
            A.plot([i0, i1], [y, y], color=SCOL[st], lw=4, solid_capstyle="butt")
            A.plot([np.nanmedian(g_("t5") - a0_) / 1e3, np.nanmedian(g_("t95") - a0_) / 1e3], [y - 2.6] * 2, color=SCOL[st], lw=1.2)
            A.plot([np.nanmedian(g_("demand_first") - a0_) / 1e3], [y], marker="v", color=SCOL[st], ms=6, mec="white", mew=0.5, ls="none")
            A.plot([np.nanmedian(g_("main_start") - a0_) / 1e3], [y], marker="|", color=SCOL[st], ms=9, ls="none")
            A.text(-2.8, y, SLAB[st], ha="left", va="center", fontsize=7, color=INK)
    B = ax[1]
    for p in GRAPH_GEMM:
        d0 = np.mean([G["st"][idx[(s, l, p)]] - G["st"][idx[(s, l, "A_qkv")]] for s in range(NSTEP) for l in range(L)]) / 1e3
        d1 = np.mean([G["en"][idx[(s, l, p)]] - G["st"][idx[(s, l, "A_qkv")]] for s in range(NSTEP) for l in range(L)]) / 1e3
        k = list(GRAPH_GEMM).index(p)
        B.plot([d0, d1], [-6 - 6.5 * k] * 2, color=INK, lw=4, solid_capstyle="butt")
        B.text(d1 + 0.3, -6 - 6.5 * k, {"A_qkv": "A qkv", "D_o": "D o-proj", "F_gu": "F gate/up", "H_down": "H down"}[p] + " kernel (mean start..end)", va="center", fontsize=7, color=INK)
    B.set_xlabel("time since layer start (us)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_fold_all_layers.png", dpi=170); plt.close(fig)


one_layer_fig(); fold_fig()
print("figures done")
