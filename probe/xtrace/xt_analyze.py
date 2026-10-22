"""CPU analysis of the Xtrace runs (xt_cuda.py: shim path, CUDA; xt_gwatch.py: gwatch runtime, CuTe / CUDA).

Per trace (xt_<clock>_<i>.npz from the shim, xt_<impl>_<clock>_<i>.npz from gwatch): gwatch's decode, then
  - completeness: records per warp, arrive / exit alternation, identical barrier-site sequence in every warp
  - consistency: per dynamic barrier, min(exit) - max(arrive) over the grid (negative = clocks disagree or a probe
    is misplaced; a grid barrier cannot release before its last arrival)
  - per barrier: exit spread inside a CTA (8 warps) and across the grid, release latency, last-arriving CTA
  - per warp: time between arrive and exit summed over the kernel, as a share of the kernel span
  - last-arriver persistence across consecutive barriers and across launches (MB2's question)
plus the timing / equality records, and the fitrace (job 8460) barrier numbers for comparison.
usage: XT_LD=... /raid/garv901/xtrace-venv/bin/python -m megabench.sota.pi05.probe.xtrace.xt_analyze RUN_DIR [FIT_DIR]"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
from gwatch.cuda.xtrace._schema import SassTraceSite

from . import splice as SP

WARPS_PER_CTA = 8
FIT_DIR = "megabench/sota/pi05/probe/runs/20261019T051256Z-fitrace"


def pct(a, qs=(50, 90, 99)):
    a = np.asarray(a, dtype=np.float64)
    return {f"p{q}": round(float(np.percentile(a, q)), 1) for q in qs} | {"max": round(float(a.max()), 1)} if a.size else {}


def decode(path, meta):
    raw = np.load(path)["raw"]
    sites = [SassTraceSite(s["insert_index"], warp_level=True) for s in meta["sites"]]
    rows = SP.decode(raw, meta["clock"], sites, meta["records"])
    w = np.fromiter((r["global_tid"] // 32 for r in rows), np.int64, len(rows))
    s = np.fromiter((r["site_id"] for r in rows), np.int64, len(rows))
    t = np.fromiter((r["timestamp"] for r in rows), np.int64, len(rows))
    sm = np.fromiter((r["smid"] for r in rows), np.int64, len(rows))
    o = np.argsort(w, kind="stable")  # keep each warp's slot order (the order it reached its sites)
    o = o[(s[o] >= 0) & (s[o] < len(meta["sites"]))]  # a probe's value word lands one slot past its record: if it is
    return w[o], s[o], t[o], sm[o]                    # the warp's last, gwatch reads it as a record of a bogus site


def analyze(path, meta):
    w, s, t, sm = decode(path, meta)
    if w.size == 0:
        return {"records": 0}
    nw = meta["num_warps"]
    kind = np.array([1 if x["kind"] == "exit" else 0 for x in meta["sites"]])
    bar = np.array([x["barrier"] for x in meta["sites"]])
    counts = np.bincount(w, minlength=nw)
    res = {"records": int(w.size), "records_per_warp": sorted(set(counts.tolist()))[:6],
           "timer_min_step_ns": None}
    # an exit site is the instruction after a barrier's closing BAR.SYNC; another path can join there (CUDA b12: warps
    # 1-7 once; CuTe b3: every warp once) and leave an exit with no arrive before it: keep only arrive -> exit pairs
    keep = np.zeros(w.size, bool)
    pair = np.nonzero((w[1:] == w[:-1]) & (kind[s[1:]] == 1) & (kind[s[:-1]] == 0) & (bar[s[1:]] == bar[s[:-1]]))[0]
    keep[pair] = keep[pair + 1] = True  # an arrive record is followed by one record, so pairs cannot overlap
    dropped = ~keep
    res["unpaired_dropped"] = {f"b{bar[k]}{'ae'[kind[k]]}": int(n) for k, n in
                               zip(*np.unique(s[dropped], return_counts=True))}
    w, s, t, sm = w[keep], s[keep], t[keep], sm[keep]
    counts = np.bincount(w, minlength=nw)
    cuts = np.r_[0, np.cumsum(counts)]
    seqs = [s[cuts[i]:cuts[i + 1]] for i in range(nw)]
    ok_alt = all(len(q) % 2 == 0 and (kind[q[0::2]] == 0).all() and (kind[q[1::2]] == 1).all()
                 and (bar[q[0::2]] == bar[q[1::2]]).all() for q in seqs)
    same = all(np.array_equal(q, seqs[0]) for q in seqs)
    res.update(alternating=bool(ok_alt), same_sequence_all_warps=bool(same), barriers_per_warp=len(seqs[0]) // 2)
    steps = np.diff(t[cuts[0]:cuts[1]])
    res["timer_min_step_ns"] = int(steps[steps > 0].min()) if (steps > 0).any() else None
    res["timer_steps_multiple_of_32ns_pct"] = round(100 * float((steps % 32 == 0).mean()), 2)
    if not (ok_alt and same):
        return res
    nb = len(seqs[0]) // 2
    T = t.reshape(nw, 2 * nb)
    A, E = T[:, 0::2], T[:, 1::2]
    smid = sm.reshape(nw, 2 * nb)[:, 0]
    span = int(E.max() - A.min())
    nc = nw // WARPS_PER_CTA
    Ec = E.reshape(nc, WARPS_PER_CTA, nb)
    Ac = A.reshape(nc, WARPS_PER_CTA, nb)
    # The closing CTA barrier is BAR.SYNC.DEFER_BLOCKING: a warp issues it and stalls only at the next instruction that
    # needs it, so the exit probe's timer read runs before the release in every warp but the one that arrives there
    # last, warp 0 (its thread 0 polls the grid counter; it is the CTA's latest exit in 99.99% of CTA-barriers). Warp
    # 0's exit is the CTA's release; another warp's "exit" is when it reached the closing barrier.
    L = E.reshape(nw // WARPS_PER_CTA, WARPS_PER_CTA, nb)[:, 0, :]
    release = L.min(0) - A.max(0)                      # per barrier: first CTA release after the last arrival
    last_cta = Ac.max(1).argmax(0)                     # per barrier: the CTA whose last warp arrived last
    wait = np.repeat(L, WARPS_PER_CTA, axis=0) - A  # arrival -> its CTA's release
    res.update(
        span_us=round(span / 1e3, 2),
        ctas_on_distinct_sms=int(len(set(smid[::WARPS_PER_CTA].tolist()))),
        release_ns=pct(release, (1, 50, 90)) | {"min": int(release.min()), "n_negative": int((release < 0).sum())},
        warp0_latest_exit_in_cta_pct=round(100 * float((Ec.argmax(1) == 0).mean()), 2),
        release_spread_grid_ns=pct(L.max(0) - L.min(0)),
        nonleader_idle_at_closing_bar_ns=pct((L[:, None, :] - Ec[:, 1:, :]).ravel()),
        arrive_spread_grid_ns=pct(A.max(0) - A.min(0)),
        wait_ns_per_warp_barrier=pct(wait.ravel()),
        barrier_share_of_span_pct=pct(100 * wait.sum(1) / span, (10, 50, 90)),
        work_between_barriers_ns=pct((A[:, 1:] - np.repeat(L, WARPS_PER_CTA, axis=0)[:, :-1]).ravel()),
        last_cta_same_as_prev_barrier_pct=round(100 * float((last_cta[1:] == last_cta[:-1]).mean()), 2),
        last_cta_top5=sorted(([int(c), int(n)] for c, n in zip(*np.unique(last_cta, return_counts=True))),
                             key=lambda x: -x[1])[:5],
        uniform_last_cta_expect_pct=round(100 / nc, 2))
    res["_last_cta"] = last_cta
    res["_site_seq"] = bar[seqs[0][0::2]]
    return res


def main():
    run = sys.argv[1]
    fit = sys.argv[2] if len(sys.argv) > 2 else FIT_DIR
    summary = {}
    metas = {}
    for f in glob.glob(os.path.join(run, "xt_meta*.json")):
        tag = os.path.basename(f)[len("xt_meta"):-len(".json")].lstrip("_") or "shim"
        metas[tag] = json.load(open(f))
    last = {}
    for path in sorted(glob.glob(os.path.join(run, "xt_*.npz"))):
        name = os.path.basename(path)[:-4]
        m = re.fullmatch(r"xt_(?:(cute|cuda)_)?(gpu|anchor)_(\d+)", name)
        if not m:
            continue
        impl, clock, i = m.group(1) or "shim", m.group(2), int(m.group(3))
        meta = metas[impl][f"xt_{clock}"]
        r = analyze(path, meta)
        lc = r.pop("_last_cta", None)
        r.pop("_site_seq", None)
        if lc is not None:
            key = (impl, clock)
            if key in last and len(last[key]) == len(lc):
                r["last_cta_same_as_prev_launch_pct"] = round(100 * float((last[key] == lc).mean()), 2)
            last[key] = lc
        summary[name] = r
        print(name, json.dumps(r))
    for f in glob.glob(os.path.join(run, "records*.jsonl")):
        recs = [json.loads(x) for x in open(f)]
        summary[os.path.basename(f)] = [{k: v for k, v in x.items() if k not in ("all_ms", "tb")} for x in recs
                                        if x.get("kind") in ("equal", "timing", "trace", "capture", "splice", "start",
                                                             "origin", "cubin", "load") or x.get("ok") is False]
    fs = os.path.join(fit, "summary.json")
    if os.path.exists(fs):
        f1 = json.load(open(fs)).get("fit1", {})
        summary["fitrace_fit1"] = {k: f1.get(k) for k in ("bar_per_warp", "timer_min_step_ns", "bar_exit_spread_in_cta_ns",
                                                          "bar_exit_spread_grid_ns")}
    with open(os.path.join(run, "xt_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("wrote", os.path.join(run, "xt_summary.json"))


if __name__ == "__main__":
    main()
