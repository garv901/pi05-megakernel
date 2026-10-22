"""CPU analysis of xt_mb.py's traces (MB2: imbalance or jitter; MB3: exposed weight wait in the GEMMs).

Decode (vectorized form of gwatch's _collect_warp, gpu clock): a record is a 32-bit word, low site_id_bits = site id + 1,
the rest the low half of the GPU timer; the warp header's first u64 is the timer at the warp's entry (high half, wrap).
Per launch: grid-barrier arrive / exit pairs as in xt_analyze (warp 0's exit is its CTA's release, DEFER_BLOCKING), and the
gemm_phase records grouped per (warp, GEMM call, tile). Barriers are labeled by the phase they close, from where the
GEMM calls fall (a call runs between its pre-barrier and post-barrier; the pattern of a layer is fixed).

Outputs xt_mb_summary.json in the run dir; usage: python -m megabench.sota.pi05.probe.xtrace.xt_mb_analyze RUN_DIR"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nan reductions over warps / calls without a tile

WARPS_PER_CTA = 8
GEMMS = ("A", "F", "H")  # gemm_phase calls in a layer: QKV, gate/up, down
# the phase each barrier closes, by its position relative to the GEMM calls (see label_barriers)
TILE_BYTES = 32 * 128 * 2  # one warp tile of weights: 32 columns x a 128-k chunk, bf16
HBM_FLOOR_US = {"A": 2.7, "F": 4.9, "H": 3.3}  # ubench U1 (job 7382): best in-kernel time to stream 5 / 16.8 / 8 MB
# gemm_phase shapes per GEMM: K, N (virtual, all weight tensors), first column of each later tensor; post-GEMM barrier label
SHAPES = {"A": (1024, 2560, (2048, 2304), "A>B"), "F": (1024, 8192, (4096,), "F>G"), "H": (4096, 1024, (), "H>res")}
GRID, KC, CALLS_PER_LAUNCH = 148, 128, 180
ROT = {"rot37_x": 37}  # PI05_TILE_ROT of a slot (0 otherwise)


def tile_model(x, rot=0, grid=GRID):
    """gemm_phase's work assignment (mk/pi05_mk.cu): block b takes K chunk s = v % S, slot v / S (v = (b + rot) % grid),
    warp w the 32-column tasks ng = slot * 8 + w, + nblk * 8, ... < NG. Per warp: s, first ng, tile count, tensor."""
    K, N, cuts, _ = SHAPES[x]
    S, NG = K // KC, N // 32
    warp = np.arange(grid * WARPS_PER_CTA)
    v = (warp // WARPS_PER_CTA + rot) % grid
    s, slot = v % S, v // S
    nblk = (grid - s + S - 1) // S
    ng = slot * WARPS_PER_CTA + warp % WARPS_PER_CTA
    count = np.where(ng < NG, (NG - 1 - ng) // (nblk * WARPS_PER_CTA) + 1, 0)
    return {"s": s, "ng": ng, "count": count, "tensor": np.digitize(ng * 32, cuts), "S": S, "NG": NG, "nblk": nblk}


def sk_critical_iters(x, dp, grid=GRID):
    """Critical path in k16 iterations (8 per 128-k tile) per GEMM: now (whole tiles) and under hybrid stream-K (dp = one
    data-parallel tile per warp first, else all stream-K; the rest of chunk s spread evenly over its warps)."""
    K, N, _, _ = SHAPES[x]
    S, NG = K // KC, N // 32
    now = sk = 0
    for s in range(S):
        nwarp = (grid - s + S - 1) // S * WARPS_PER_CTA
        now = max(now, -(-NG // nwarp) * 8)
        full = min(NG, nwarp) if dp else 0
        sk = max(sk, (8 if full else 0) + -(-(NG - full) * 8 // nwarp))
    return now, sk


def imbalance_launch(r, rot):
    """Per GEMM, per warp (mean over the launch's calls): finish vs the median max-count warp, start skew (g_async),
    first-tile wait (g_async -> g_m0), body (g_m0 -> g_end); per call: finish spread among max-count warps; the model
    check; per-tile body by count (for the stream-K bound)."""
    G = r["G"]
    nw, ncall = G["g_async"].shape
    out = {}
    for gi, x in enumerate(GEMMS):
        calls = np.arange(gi, ncall, 3)
        m = tile_model(x, rot)
        ga, ge, m0 = G["g_async"][:, calls], G["g_end"][:, calls], G["g_m0"][:, calls, 0]
        mx = m["count"] == m["count"].max()
        ref = np.nanmedian(ge[mx], 0)
        fin = ge - ref
        st = ga - np.nanmedian(ga[mx], 0)
        out[x] = {"model_ok": bool((G["tiles"][:, calls] == m["count"][:, None]).all()),
                  "finish": np.nanmean(fin, 1), "start": np.nanmean(st, 1), "wait1": np.nanmean(m0 - ga, 1),
                  "body": np.nanmean(ge - m0, 1), "dur_var": float(np.nanvar(fin[mx] - st[mx])),
                  "spread": np.nanmax(fin[mx], 0) - np.nanmin(fin[mx], 0)}
    return out


def imbalance_summary(per, rot):
    """Aggregate imbalance_launch over launches: is the post-GEMM lateness the tile count (structural) or per-warp cost?"""
    out = {}
    for x in GEMMS:
        m = tile_model(x, rot)
        cnt, mx = m["count"], m["count"] == m["count"].max()
        P = {k: np.stack([p[x][k] for p in per]) for k in ("finish", "start", "wait1", "body")}  # [launch, warp]
        dur = P["finish"] - P["start"]
        mean_dur = dur[:, mx].mean(0)
        within = float(np.mean([p[x]["dur_var"] for p in per]))
        o = {"model_matches_measured_tiles": all(p[x]["model_ok"] for p in per),
             "warps_by_tile_count": {int(k): int(v) for k, v in zip(*np.unique(cnt, return_counts=True))},
             "busy_blocks": int((cnt.reshape(-1, WARPS_PER_CTA).max(1) > 0).sum()),
             "finish_vs_max_count_median_ns_by_count": {int(k): round(float(np.nanmean(P["finish"][:, cnt == k])))
                                                        for k in np.unique(cnt) if k > 0},
             "max_count_finish_spread_per_call_ns": pct(np.concatenate([p[x]["spread"] for p in per]), (50, 90)),
             "max_count_start_skew_std_ns": round(float(np.nanstd(P["start"][:, mx]))),
             "max_count_first_tile_wait_ns": pct(P["wait1"][:, mx].ravel(), (50, 90)),
             "body_ns_by_count": {int(k): round(float(np.nanmedian(P["body"][:, cnt == k]))) for k in np.unique(cnt)
                                  if k > 0},
             # share of the max-count warps' duration variance (per call) that is the same warp in every call / launch
             "max_count_duration_repeatable_share": round(float(np.var(mean_dur) / within), 3) if within else None,
             "max_count_mean_duration_ns_by": {
                 f: [round(float(mean_dur[m[f][mx] == v].mean())) for v in np.unique(m[f][mx])]
                 for f in ("tensor", "s")}}
        # hybrid stream-K bound: k16 iterations on the critical path x the measured per-iteration time (linear, no fixed
        # latency, no fixup), over a launch's calls of this GEMM
        b = o["body_ns_by_count"]
        it_ns = (b[max(b)] - b[min(b)]) / 8 / (max(b) - min(b)) if len(b) > 1 else b[max(b)] / 8
        now, dp = sk_critical_iters(x, True)
        _, nodp = sk_critical_iters(x, False)
        o["stream_k"] = {"iter_ns": round(it_ns, 1), "critical_iters_now": now, "critical_iters_sk_dp": dp,
                         "critical_iters_sk_all": nodp,
                         "bound_saving_us_per_launch": round((now - min(dp, nodp)) * it_ns * CALLS_PER_LAUNCH / 1e3, 1)}
        out[x] = o
    return out


def pct(a, qs=(50, 90, 99)):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return {f"p{q}": round(float(np.percentile(a, q)), 1) for q in qs} | {"n": int(a.size)} if a.size else {}


def decode(npz, meta):
    """(warp, site, t ns, smid) of every record, each warp's in the order it wrote them."""
    z = np.load(npz)
    hdr, bits = meta["header_bytes"], meta["site_id_bits"]
    if "slab" in z:
        nw, slab = int(z["nw"]), z["slab"]
    else:  # a full buffer (xt_cuda.py)
        nw = meta["num_warps"]
        slab = z["raw"][:nw * meta["warp_slab_bytes"]].reshape(nw, -1)
    assert meta["record_bytes"] == 4 and meta["clock"] == "gpu" and hdr == 16, meta["record_bytes"]
    head = slab[:, :hdr].copy().view("<u8")
    entry = head[:, 0].astype(np.int64)                     # the timer at the warp's entry, then its SM id
    smid = (head[:, 1] & 0xFFFFFFFF).astype(np.int64)
    words = slab[:, hdr:].copy().view("<u4")
    w, pos = np.nonzero(words)
    word = words[w, pos].astype(np.int64)
    site = (word & ((1 << bits) - 1)) - 1
    low = word & ~((1 << bits) - 1) & 0xFFFFFFFF
    e = entry[w]
    t = (e & ~0xFFFFFFFF) | low
    t += np.where(low < (e & 0xFFFFFFFF), 1 << 32, 0)
    ok = (site >= 0) & (site < len(meta["sites"]))  # the value word past a warp's last record can read as a record
    return w[ok], site[ok], t[ok], smid


def barriers(w, s, t, meta, nw):
    """A, E [warp, barrier] (ns) and the code-site sequence, from the arrive -> exit pairs (unpaired exits dropped)."""
    kinds = meta["sites"]
    isb = np.array([k["barrier"] >= 0 for k in kinds])
    kind = np.array([1 if k["kind"] == "exit" else 0 for k in kinds])
    bar = np.array([k["barrier"] for k in kinds])
    m = isb[s]
    w, s, t = w[m], s[m], t[m]
    pair = np.nonzero((w[1:] == w[:-1]) & (kind[s[1:]] == 1) & (kind[s[:-1]] == 0) & (bar[s[1:]] == bar[s[:-1]]))[0]
    keep = np.zeros(w.size, bool)
    keep[pair] = keep[pair + 1] = True
    w, s, t = w[keep], s[keep], t[keep]
    counts = np.bincount(w, minlength=nw)
    assert (counts == counts[0]).all(), sorted(set(counts.tolist()))[:5]
    nb = counts[0] // 2
    seq = bar[s[:2 * nb:2]]
    S = s.reshape(nw, 2 * nb)
    assert (bar[S[:, 0::2]] == seq).all(), "warps disagree on the barrier sequence"
    T = t.reshape(nw, 2 * nb)
    return T[:, 0::2], T[:, 1::2], seq, int((~keep).sum())


def gemm_records(w, s, t, meta, nw, max_tiles=4):
    """Per (warp, GEMM call): g_async, g_end and per tile g_tile / g_m0 / g_wnext (ns; nan if absent)."""
    kinds = [k["kind"] for k in meta["sites"]]
    code = {k: i for i, k in enumerate(kinds) if k.startswith("g_")}
    if not code:
        return None
    m = np.isin(s, list(code.values()))
    w, s, t = w[m], s[m], t[m]
    is_async = s == code["g_async"]
    # call index per record: number of g_async this warp has written so far, minus one
    first = np.r_[True, w[1:] != w[:-1]]
    c = np.cumsum(is_async)
    base = np.maximum.accumulate(np.where(first, c - is_async, 0))
    call = c - base - 1
    ncall = int(call.max()) + 1
    is_tile = s == code["g_tile"]
    key = w * ncall + call
    newkey = np.r_[True, key[1:] != key[:-1]]
    ct = np.cumsum(is_tile)
    tbase = np.maximum.accumulate(np.where(newkey, ct - is_tile, 0))
    tile = ct - tbase - 1
    out = {k: np.full((nw, ncall), np.nan) for k in ("g_async", "g_end")}
    for k in ("g_async", "g_end"):
        mm = s == code[k]
        out[k][w[mm], call[mm]] = t[mm]
    for k in ("g_tile", "g_m0", "g_wnext"):
        arr = np.full((nw, ncall, max_tiles), np.nan)
        mm = (s == code[k]) & (tile >= 0) & (tile < max_tiles)
        arr[w[mm], call[mm], tile[mm]] = t[mm]
        out[k] = arr
    out["tiles"] = np.isfinite(out["g_tile"]).sum(2)
    return out


def label_barriers(seq, A, G):
    """Phase each dynamic barrier closes. A GEMM call k (type GEMMS[k % 3]) lies between the last barrier whose warp-0
    arrival precedes its g_async and the next one: those are '>X' (pre) and 'X>' (post). In a layer, after 'A>' come the
    barriers closing B, C (attention) and D (o_proj), then '>F' closes E (residual); '>H' closes G; '>A' the residual
    after H (or the step tail)."""
    nb = len(seq)
    lab = ["init"] * nb
    a0 = A[0]
    ga = G["g_async"][0]
    for k in range(ga.size):
        x = GEMMS[k % 3]
        pre = int(np.searchsorted(a0, ga[k])) - 1
        if pre >= 0:
            lab[pre] = {"A": "res>A", "F": "E>F", "H": "G>H"}[x]
        if pre + 1 < nb:
            lab[pre + 1] = {"A": "A>B", "F": "F>G", "H": "H>res"}[x]
            if x == "A":
                for j, name in enumerate(("B>C", "C>D", "D>E"), 2):
                    if pre + j < nb and lab[pre + j] == "init":
                        lab[pre + j] = name
    # barriers between 'H>res' and the next '>A' (step tail, Euler update) keep their own name
    for i in range(1, nb):
        if lab[i] == "init" and lab[i - 1] in ("H>res", "tail"):
            lab[i] = "tail"
    return lab


def launch(npz, meta):
    nw = meta["num_warps"]
    w, s, t, smid = decode(npz, meta)
    A, E, seq, dropped = barriers(w, s, t, meta, nw)
    G = gemm_records(w, s, t, meta, nw)
    nc, nb = nw // WARPS_PER_CTA, A.shape[1]
    L = E.reshape(nc, WARPS_PER_CTA, nb)[:, 0, :]          # CTA release (warp 0's exit)
    Ac = A.reshape(nc, WARPS_PER_CTA, nb).max(1)           # CTA arrival (its last warp)
    return {"A": A, "E": E, "L": L, "Ac": Ac, "seq": seq, "G": G, "smid": smid[::WARPS_PER_CTA], "dropped": dropped,
            "lab": label_barriers(seq, A, G) if G is not None else ["?"] * nb}


def wait_split(r):
    """Per warp-barrier wait = intra (until its CTA's last warp) + cross-CTA (until the grid's last CTA) + mechanism
    (last arrival -> its CTA's release); per-label totals in us per warp per launch, and shares of the span."""
    A, L, Ac = r["A"], r["L"], r["Ac"]
    rel = np.repeat(L, WARPS_PER_CTA, 0)
    ac = np.repeat(Ac, WARPS_PER_CTA, 0)
    last = A.max(0)
    span = float(r["E"].max() - A.min())
    parts = {"intra_cta": ac - A, "cross_cta": last[None, :] - ac, "mechanism": rel - last[None, :]}
    lab = np.array(r["lab"])
    out = {"span_us": round(span / 1e3, 1)}
    for k, v in parts.items():
        out[k + "_pct_of_span"] = round(100 * float(v.sum(1).mean()) / span, 2)
    out["total_wait_pct_of_span"] = round(sum(out[k + "_pct_of_span"] for k in parts), 2)
    by = {}
    for name in dict.fromkeys(r["lab"]):
        m = lab == name
        by[name] = {"n": int(m.sum()), **{k: round(float(v[:, m].sum(1).mean()) / 1e3, 1) for k, v in parts.items()},
                    "arrive_spread_cta_p50_ns": round(float(np.median(Ac[:, m].max(0) - Ac[:, m].min(0))), 0)}
    out["by_phase_us_per_warp"] = by
    return out


def det_jitter(Acs, labs):
    """Acs [launch, cta, barrier]: CTA arrival minus that barrier's median CTA arrival. Per label: total variance, the
    part that repeats across launches (variance of the per-(cta, barrier) mean, less the noise it carries), the mean
    lateness of the last CTA vs what the repeating part alone predicts, and how often the last CTA is the same one."""
    n = Acs.shape[0]
    X = Acs - np.median(Acs, axis=1, keepdims=True)
    mean = X.mean(0)
    resid = X - mean
    lab = np.array(labs)
    out = {}
    for name in dict.fromkeys(labs):
        m = lab == name
        if m.sum() < 5:
            continue
        x, mu, rs = X[:, :, m], mean[:, m], resid[:, :, m]
        v_tot = float(x.var())
        v_noise = float(rs.var()) * n / max(n - 1, 1)
        v_det = max(float(mu.var()) - v_noise / n, 0.0)
        lastc = x.argmax(1)                                   # [launch, barrier]
        same = [(lastc[i] == lastc[j]).mean() for i in range(n) for j in range(i + 1, n)]
        out[name] = {"n_barriers": int(m.sum()), "sd_ns": round(v_tot ** .5, 0),
                     "repeatable_var_pct": round(100 * v_det / v_tot, 1) if v_tot else None,
                     "last_cta_late_ns_obs": round(float(x.max(1).mean()), 0),
                     "last_cta_late_ns_repeatable": round(float(mu.max(0).mean()), 0),
                     "last_cta_same_across_launches_pct": round(100 * float(np.mean(same)), 1) if same else None,
                     "uniform_pct": round(100 / x.shape[1], 2)}
    return out, mean, (X[0::2].mean(0), X[1::2].mean(0))


def gemm_stats(r):
    """MB3 per GEMM type: tiles per warp, exposed weight wait per tile, issue -> ready, and the GEMM's weight stream
    time vs the HBM floor; plus MB2's 'what was the last warp at the post-GEMM barrier doing'."""
    G = r["G"]
    if G is None:
        return {}
    nw, ncall = G["g_async"].shape
    nc = nw // WARPS_PER_CTA
    lab = np.array(r["lab"])
    out = {}
    for gi, x in enumerate(GEMMS):
        calls = np.arange(gi, ncall, 3)
        ga, ge = G["g_async"][:, calls], G["g_end"][:, calls]
        tl, m0, wn = G["g_tile"][:, calls], G["g_m0"][:, calls], G["g_wnext"][:, calls]
        tiles = G["tiles"][:, calls]
        # the A-tile CTA sync releases when the CTA's last warp arrives at it (g_async is that arrival)
        arel = np.repeat(np.nanmax(ga.reshape(nc, WARPS_PER_CTA, -1), 1), WARPS_PER_CTA, 0)
        w1 = m0[:, :, 0] - arel                              # tile 1: weights were issued before the barrier (WPRE)
        w2 = m0[:, :, 1] - tl[:, :, 1]                       # tile 2: from the loop head (after tile 1's stores)
        lat2 = m0[:, :, 1] - wn[:, :, 0]                     # tile 2 issue -> ready
        mma1 = np.where(tiles > 1, wn[:, :, 0], ge) - m0[:, :, 0]
        # weight stream window of a call: the first warp's arrival at the pre-barrier (WPRE issues the first tile's
        # loads right after it; without WPRE they wait for the release) ... the last warp's last weights ready
        pre = np.array([np.searchsorted(r["A"][0], ga[0, k]) - 1 for k in range(len(calls))])
        start = r["A"][:, pre].min(0) if (pre >= 0).all() else np.full(len(calls), np.nan)
        start_rel = r["L"][:, pre].min(0) if (pre >= 0).all() else np.full(len(calls), np.nan)
        last_ready = np.nanmax(np.where(np.isfinite(m0), m0, -np.inf).reshape(nw, len(calls), -1), axis=(0, 2))
        nbytes = tiles.sum(0) * TILE_BYTES
        win = (last_ready - start) / 1e3
        o = {"tiles_per_warp": {int(k): int(v) for k, v in zip(*np.unique(tiles[:, 0], return_counts=True))},
             "weight_bytes_per_call_mb": round(float(nbytes.mean()) / 1e6, 2),
             "tile1_wait_after_A_sync_ns": pct(w1), "tile2_wait_after_stores_ns": pct(w2),
             "tile2_issue_to_ready_ns": pct(lat2), "tile1_mma_and_stores_ns": pct(mma1),
             "warp_gemm_time_ns": pct(ge - ga), "stream_window_us": pct(win, (10, 50, 90)),
             "stream_window_from_release_us": pct((last_ready - start_rel) / 1e3, (10, 50, 90)),
             "stream_TBps_p50": round(float(np.nanmedian(nbytes / (win * 1e-6))) / 1e12, 2),
             "hbm_floor_us": HBM_FLOOR_US[x]}
        # the post-GEMM barrier: which warp arrives last, how many tiles it had, and its stages vs the median warp
        post = pre + 1
        if (post < lab.size).all():
            Ab = r["A"][:, post]                             # [warp, call]
            lw = Ab.argmax(0)
            k = np.arange(len(calls))
            seg = {"pre_gemm": ga - np.repeat(r["L"][:, pre], WARPS_PER_CTA, 0), "tile1_wait": w1, "tile1_work": mma1,
                   "tile2_wait": m0[:, :, 1] - wn[:, :, 0], "after_last_tile": ge - np.nanmax(m0, 2),
                   "gemm_end_to_arrive": Ab - ge}
            o["post_barrier"] = {
                "label": str(lab[post[0]]),
                "last_warp_tiles": {int(a): int(b) for a, b in zip(*np.unique(tiles[lw, k], return_counts=True))},
                "last_warp_segments_ns": {n: round(float(np.nanmedian(v[lw, k])), 0) for n, v in seg.items()},
                "median_warp_segments_ns": {n: round(float(np.nanmedian(v)), 0) for n, v in seg.items()},
                "last_warp_in_A_sync_late_cta_pct": round(100 * float(np.mean(
                    np.nanargmax(ga.reshape(nc, WARPS_PER_CTA, -1), 1)[lw // WARPS_PER_CTA, k] == lw % WARPS_PER_CTA)), 1)}
        out[x] = o
    return out


def main():
    run = sys.argv[1]
    metas = json.load(open(os.path.join(run, "xt_mb_meta.json")))
    summary = {"records": [json.loads(x) for x in open(os.path.join(run, "records.jsonl"))
                           if json.loads(x).get("kind") in ("equal", "timing")]}
    for x in summary["records"]:
        x.pop("all_ms", None)
    dets, imb = {}, {}
    for slot in metas:
        files = sorted(glob.glob(os.path.join(run, f"xt_{slot}_*.npz")), key=lambda p: int(re.findall(r"_(\d+)\.npz", p)[0]))
        if not files:
            continue
        meta = metas[slot]
        per, Acs, labs, smids = [], [], None, []
        for f in files:
            r = launch(f, meta)
            ws = wait_split(r)
            per.append({"file": os.path.basename(f), "dropped": r["dropped"], "barriers": len(r["seq"]), **ws,
                        "gemm": gemm_stats(r)})
            if r["G"] is not None and slot in ("base_x", "rot37_x"):
                imb.setdefault(slot, []).append(imbalance_launch(r, ROT.get(slot, 0)))
            Acs.append(r["Ac"])
            smids.append(r["smid"])
            labs = labs or r["lab"]
            print(slot, os.path.basename(f), ws["span_us"], ws["total_wait_pct_of_span"], flush=True)
        Acs = np.stack(Acs)
        dj, mean, halves = det_jitter(Acs, labs)
        dets[slot] = (mean, labs, halves)
        sm = np.stack(smids)
        summary[slot] = {"launches": len(files), "label_counts": {k: labs.count(k) for k in dict.fromkeys(labs)},
                         "cta_to_sm_same_all_launches": bool((sm == sm[0]).all()),
                         "first": per[0], "per_launch_wait": [{k: p[k] for k in ("span_us", "total_wait_pct_of_span",
                                                                                 "intra_cta_pct_of_span",
                                                                                 "cross_cta_pct_of_span",
                                                                                 "mechanism_pct_of_span")} for p in per],
                         "det_jitter": dj}
    # MB2 rotation test: does the repeatable lateness of a CTA follow its work (blockIdx + 37) or its SM (blockIdx)?
    if "base_x" in dets and "rot37_x" in dets:
        (mb, lb, (h0, h1)), (mr, lr, _) = dets["base_x"], dets["rot37_x"]
        rot = {}
        if lb == lr:
            lab = np.array(lb)
            for name in dict.fromkeys(lb):
                m = lab == name
                if m.sum() < 5:
                    continue
                b, r_ = mb[:, m], mr[:, m]
                work = np.roll(b, -37, axis=0)                 # base's CTA (c + 37) did the work rot37's CTA c does
                rot[name] = {"corr_same_sm": round(float(np.corrcoef(r_.ravel(), b.ravel())[0, 1]), 3),
                             "corr_same_work": round(float(np.corrcoef(r_.ravel(), work.ravel())[0, 1]), 3),
                             # the ceiling: base's even launches vs its odd ones
                             "corr_base_split_half": round(float(np.corrcoef(h0[:, m].ravel(), h1[:, m].ravel())[0, 1]), 3)}
            summary["rotation"] = rot
        else:
            summary["rotation"] = "label sequences differ"
    summary["imbalance"] = {slot: imbalance_summary(v, ROT.get(slot, 0)) for slot, v in imb.items()}
    # busy-only rotation check: over CTAs that hold a max-count warp in both builds, does the post-GEMM lateness still
    # follow the work? (the all-CTA correlation is dominated by which CTAs are busy at all)
    if "base_x" in dets and "rot37_x" in dets and dets["base_x"][1] == dets["rot37_x"][1]:
        (mb, lb, _), (mr, _, _) = dets["base_x"], dets["rot37_x"]
        lab = np.array(lb)
        busy_rot = {}
        for x in GEMMS:
            m = lab == SHAPES[x][3]
            cb = tile_model(x, 0)["count"].reshape(-1, WARPS_PER_CTA).max(1)
            cr = tile_model(x, 37)["count"].reshape(-1, WARPS_PER_CTA).max(1)
            top = max(cb.max(), cr.max())
            work = np.roll(mb[:, m], -37, axis=0)
            sw = cr == top                                   # rot CTA busy (its base twin c + 37 did the same work)
            ss = sw & (cb == top)                            # busy in both builds on the same SM
            busy_rot[x] = {"label": SHAPES[x][3], "ctas_same_work": int(sw.sum()), "ctas_same_sm": int(ss.sum()),
                           "corr_same_work": round(float(np.corrcoef(mr[:, m][sw].ravel(), work[sw].ravel())[0, 1]), 3),
                           "corr_same_sm": round(float(np.corrcoef(mr[:, m][ss].ravel(), mb[:, m][ss].ravel())[0, 1]), 3)}
        summary["rotation_busy_only"] = busy_rot
    with open(os.path.join(run, "xt_mb_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("wrote", os.path.join(run, "xt_mb_summary.json"))


if __name__ == "__main__":
    main()
