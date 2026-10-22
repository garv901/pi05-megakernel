"""Analysis of the FlashInfer-style traces written by probe/fitrace.py (CPU only).

Per mode (fit1 = no fence, fit2 = FlashInfer's __threadfence_block):
  - completeness: entries per warp, barrier events per warp vs the block-0 tstamp count
  - timer: smallest nonzero step of globaltimer_lo between consecutive entries of one warp
  - gemm_phase sub-stages per call (A / D / F / H), medians over warps and layers, tile 1 and tile 2 separately
  - barrier: per gemm phase, warp idle from its last gemm event to the barrier exit; per barrier, exit spread inside a CTA
    (the 8 warps) and across the grid
  - CuTe traces (fit_cute1 / fit_cute2, probe/cute_fitrace.py; gemm_phase markers only, no barrier entries): calls assigned
    A / F / H by order (3 per layer per warp, checked), sub-stage medians between the iket_marks sites
  - fit1 only: Chrome-trace JSON (loads in ui.perfetto.dev) of step 0, layers 0-1, every warp: trace_s0_l01.json
usage: python -m megabench.sota.pi05.probe.fitrace_analyze RUN_DIR"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from .phases import L, NSTEP, PHASES, mk_slot

NWARPS = 8
EV = ["BAR", "G0", "WISS", "ADONE", "SYNC", "M0", "MMA", "ST", "WNEXT", "GEND"]
GEMM_PHASE = {0: "A_qkv", 3: "D_o", 5: "F_gu", 7: "H_down"}
SLOT = {mk_slot(s, l, i): (s, l, i) for s in range(NSTEP) for l in range(L) for i in range(9)}


def load(path):
    z = np.load(path)
    w, k, tag, t = z["warp"], z["k"], z["tag"], z["t"].astype(np.int64)
    ref = int(t[np.argmin(k)])
    rel = ((t - ref + (1 << 31)) % (1 << 32)) - (1 << 31)  # signed ns from one early entry (a chunk is << 2 s)
    ev = (tag >> 2) & 0x3FF
    sm = tag >> 24
    order = np.lexsort((k, w))
    per = {}
    ws, ks, evs, ts, sms = w[order], k[order], ev[order], rel[order], sm[order]
    cuts = np.flatnonzero(np.diff(ws)) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, ws.size]):
        per[int(ws[a])] = (evs[a:b].astype(int), ts[a:b].astype(np.float64), int(sms[a]))
    n_tstamp = int(np.count_nonzero(z["tstamp"][1:]))
    return per, n_tstamp, int(z["last_k"])


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def analyze(per, n_tstamp):
    res = {}
    nbar = {w: int((e == 0).sum()) for w, (e, _, _) in per.items()}
    res["entries_per_warp"] = [int(min(len(e) for e, _, _ in per.values())), int(max(len(e) for e, _, _ in per.values()))]
    res["bar_per_warp"] = sorted(set(nbar.values()))
    res["tstamp_slots"] = n_tstamp
    steps = np.concatenate([np.diff(t) for _, t, _ in per.values()])
    pos = steps[steps > 0]
    res["timer_min_step_ns"] = float(pos.min())
    res["timer_steps_multiple_of_32ns_pct"] = float(100.0 * np.mean(np.isclose(pos % 32, 0)))
    res["zero_step_pct"] = float(100.0 * np.mean(steps == 0))

    # barrier exit times: bar[j][warp] (j = slot index, 1-based like tcount)
    nb = min(nbar.values())
    warps = sorted(per)
    bar = np.full((nb + 1, len(warps)), np.nan)
    for c, w in enumerate(warps):
        e, t, _ = per[w]
        bar[1:, c] = t[e == 0][:nb]
    blocks = np.array(warps) // NWARPS
    cta_spread = [np.nanmax(bar[j].reshape(-1, NWARPS), 1) - np.nanmin(bar[j].reshape(-1, NWARPS), 1) for j in range(1, nb + 1)]
    cta_spread = np.concatenate(cta_spread)
    grid_spread = np.nanmax(bar[1:], 1) - np.nanmin(bar[1:], 1)
    res["bar_exit_spread_in_cta_ns"] = {"p50": med(cta_spread), "p90": float(np.percentile(cta_spread, 90)),
                                        "max": float(cta_spread.max())}
    res["bar_exit_spread_grid_ns"] = {"p50": med(grid_spread), "p90": float(np.percentile(grid_spread, 90))}
    # phase durations from the earliest exit (grid) between consecutive layer slots
    gmin = np.nanmin(bar, 1)
    ph = {p: [] for p in PHASES}
    for s in range(NSTEP):
        for l in range(L):
            for i in range(9):
                a = mk_slot(s, l, i)
                b = mk_slot(s, l, i + 1) if i < 8 else (mk_slot(s, l + 1, 0) if l < L - 1 else None)
                if b is not None and b <= nb:
                    ph[PHASES[i]].append(gmin[b] - gmin[a])
    res["phase_us_median"] = {p: med(v) / 1e3 for p, v in ph.items()}

    # gemm_phase calls
    calls = {p: [] for p in GEMM_PHASE.values()}
    idle = {p: [] for p in GEMM_PHASE.values()}
    for w in warps:
        e, t, _ = per[w]
        bar_idx = np.cumsum(e == 0)  # slot number of the latest barrier at or before each entry
        starts = np.flatnonzero(e == 1)
        for g0 in starts:
            end = g0 + np.argmax(e[g0:] == 9)
            seg_e, seg_t = e[g0:end + 1], t[g0:end + 1]
            slot = int(bar_idx[end])  # an inner (wpre) barrier counts; else the barrier before the call
            if slot not in SLOT or SLOT[slot][2] not in GEMM_PHASE:
                continue
            p = GEMM_PHASE[SLOT[slot][2]]
            d = {"issue": None, "wpre_wait": None}
            pos = {name: [] for name in EV}
            for x, tt in zip(seg_e, seg_t):
                pos[EV[x]].append(tt)
            if pos["WISS"]:
                d["issue"] = pos["WISS"][0] - pos["G0"][0]
            if pos["BAR"]:
                d["wpre_wait"] = pos["BAR"][0] - pos["WISS"][0]
            d["stage_a"] = pos["ADONE"][0] - (pos["BAR"] or pos["WISS"])[0]
            d["cta_sync"] = pos["SYNC"][0] - pos["ADONE"][0]
            m0, mma, st, wn = pos["M0"], pos["MMA"], pos["ST"], pos["WNEXT"]
            d["ntiles"] = len(m0)
            if m0:
                d["t1_first_mma"] = m0[0] - pos["SYNC"][0]  # sync -> first 8 MMAs issued (tile-1 weights arrived)
                d["t1_mma"] = mma[0] - m0[0]
                d["t1_store"] = st[0] - mma[0]
            if len(m0) > 1:
                d["t1_load_next"] = wn[0] - st[0]
                d["t2_first_mma"] = m0[1] - wn[0]  # next-tile load issued -> its first 8 MMAs (exposed weight wait)
                d["t2_mma"] = mma[1] - m0[1]
                d["t2_store"] = st[1] - mma[1]
            calls[p].append(d)
            nxt = np.flatnonzero((e == 0) & (np.arange(len(e)) > end))
            if nxt.size:
                idle[p].append(t[nxt[0]] - t[end])
    out = {}
    for p, cs in calls.items():
        if not cs:  # D runs gemm_phase_wsplit (no markers) in the default build
            continue
        keys = ["issue", "wpre_wait", "stage_a", "cta_sync", "t1_first_mma", "t1_mma", "t1_store", "t1_load_next",
                "t2_first_mma", "t2_mma", "t2_store"]
        out[p] = {"calls": len(cs), "two_tile_pct": 100.0 * np.mean([c["ntiles"] > 1 for c in cs]) if cs else 0.0}
        for kk in keys:
            v = [c[kk] for c in cs if c.get(kk) is not None]
            out[p][kk + "_us"] = med(v) / 1e3 if v else None
        out[p]["idle_to_barrier_us"] = {"p50": med(idle[p]) / 1e3, "p90": float(np.percentile(idle[p], 90)) / 1e3}
    res["gemm"] = out
    return res, bar


def chrome_trace(per, path, slots):
    """Slices between consecutive entries of a warp, named after the entry that opens them; pid = SM, tid = block.warp."""
    evs = []
    lo = hi = None
    for w, (e, t, sm) in per.items():
        bi = np.cumsum(e == 0)
        keep = np.isin(bi, slots)
        idx = np.flatnonzero(keep)
        for a in idx:
            if a + 1 >= len(e):
                continue
            name = EV[e[a]] + (f"@{PHASES[SLOT[int(bi[a])][2]]}" if e[a] == 0 and int(bi[a]) in SLOT else "")
            evs.append({"name": name, "ph": "X", "pid": int(sm), "tid": int(w), "ts": t[a] / 1e3, "dur": (t[a + 1] - t[a]) / 1e3})
    for w, (e, t, sm) in per.items():
        evs.append({"name": "thread_name", "ph": "M", "pid": int(sm), "tid": int(w), "args": {"name": f"blk{w // NWARPS}.w{w % NWARPS}"}})
    with open(path, "w") as f:
        json.dump({"traceEvents": evs, "displayTimeUnit": "ns"}, f)
    return len(evs)


CUTE_EV = ["-", "g_0", "g_wiss", "g_adone", "g_sync", "g_wrdy", "g_mma", "g_st", "g_wnext"]


def analyze_cute(per):
    """Sub-stage medians (us) per gemm (A / F / H by call order) between consecutive CuTe marker sites."""
    res = {"calls_per_warp": sorted({int((e == 1).sum()) for e, _, _ in per.values()})}
    seg = {}
    for e, t, _ in per.values():
        starts = np.flatnonzero(e == 1)
        for c, a in enumerate(starts):
            b = starts[c + 1] if c + 1 < len(starts) else len(e)
            ph = ("A_qkv", "F_gu", "H_down")[c % 3]
            tile = 0
            for i in range(a, b - 1):
                name = f"{CUTE_EV[e[i]]}->{CUTE_EV[e[i + 1]]}"
                if e[i] == 8:  # g_wnext opens the next tile
                    tile += 1
                key = (ph, name if e[i] < 5 else f"t{tile + 1}:{name}")
                seg.setdefault(key, []).append(t[i + 1] - t[i])
    out = {}
    for (ph, name), v in sorted(seg.items()):
        out.setdefault(ph, {})[name] = {"p50_us": med(v) / 1e3, "n": len(v)}
    res["gemm"] = out
    return res


def main():
    run = sys.argv[1]
    summary = {}
    for m in ("cute1", "cute2"):
        p = os.path.join(run, f"fit_{m}.npz")
        if os.path.exists(p):
            per, _, last_k = load(p)
            r = analyze_cute(per)
            r["last_k"] = last_k
            summary["fit_" + m] = r
            print(f"## fit_{m} calls/warp {r['calls_per_warp']}")
            for ph, d in r["gemm"].items():
                print(ph, "  ".join(f"{k} {x['p50_us']:.2f}" for k, x in d.items()))
    for m in ("fit1", "fit2"):
        p = os.path.join(run, f"fit_{m}.npz")
        if not os.path.exists(p):
            continue
        per, n_tstamp, last_k = load(p)
        res, _ = analyze(per, n_tstamp)
        res["last_k"] = last_k
        summary[m] = res
        if m == "fit1":
            slots = [mk_slot(0, l, i) for l in (0, 1) for i in range(9)]
            res["chrome_trace_events"] = chrome_trace(per, os.path.join(run, "trace_s0_l01.json"), slots)
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for m, r in summary.items():
        if m.startswith("fit_cute"):
            continue
        print(f"## {m}")
        print(json.dumps({k: v for k, v in r.items() if k != "gemm"}, indent=1))
        print("| gemm | calls | 2-tile % | issue | wpre wait | stage A | CTA sync | t1 first MMA | t1 MMA | t1 store | load next |"
              " t2 first MMA | t2 MMA | t2 store | idle->bar p50 / p90 (us) |")
        print("|---" * 15 + "|")
        for p, g in r["gemm"].items():
            f = lambda x: "-" if x is None else f"{x:.2f}"
            print(f"| {p} | {g['calls']} | {g['two_tile_pct']:.0f} | " + " | ".join(
                f(g[k + "_us"]) for k in ["issue", "wpre_wait", "stage_a", "cta_sync", "t1_first_mma", "t1_mma", "t1_store",
                                          "t1_load_next", "t2_first_mma", "t2_mma", "t2_store"])
                  + f" | {g['idle_to_barrier_us']['p50']:.2f} / {g['idle_to_barrier_us']['p90']:.2f} |")


if __name__ == "__main__":
    main()
