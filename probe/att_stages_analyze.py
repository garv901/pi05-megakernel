"""K2 attention sub-stages from probe/mk_gemm_stages.py traces (PI05_FITRACE=3: clock64 entries, CPU only).

Per trace (fit3_<i>.npz), warp 0 of each unit block (0..142), per layer. Gates as probe/gemm_stages_analyze.py (cal pairs, ns / cycle
spread, span vs event time, no negative interval) plus calls_ok: 180 units and 180 combines per block. Durations are within one SM:
  bar_to_unit (attention barrier exit -> unit entry), ld (unit entry -> Q (+ tail suffix K / V) landed), qk (QK^T + softmax) = qk_mma (warp 0's
  QK^T MMAs) + softmax (mask, max / sum exchange, P stores, CTA barriers), pv (P.V),
  store_rel (opart / stats stores + release), unit (barrier exit -> release), gap (release -> combine entry: the wg / wu L2 prefetch issue),
  cwait (combine entry -> the row group's 11 units released), comb (combine loads / math / attn store), to_bar (combine end -> D's barrier
  exit), phase (attention barrier exit -> D's barrier exit). Split into prefix units (chunks 0..9) and tail units (chunk 10: prefix 960..967 +
  the 50 suffix keys). crit: per layer, the block with the longest barrier -> release (the row-group critical unit), its stage split.
usage: python -m megabench.sota.pi05.probe.att_stages_analyze RUN_DIR"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

M32 = 1 << 32
EV = ["BAR", "G0", "WISS", "ADONE", "SYNC", "M0", "MMA", "ST", "WNEXT", "GEND", "CALG", "CALC", "D0",
      "AU0", "ALD", "AQK", "APV", "AREL", "AC0", "ACW", "ACEND", "AMM"]
E = {n: i for i, n in enumerate(EV)}
NWARPS, LAYERS, NRG, NCH = 8, 180, 13, 11
KEYS = ["bar_to_unit", "ld", "qk", "qk_mma", "softmax", "pv", "store_rel", "unit", "gap", "cwait", "comb", "to_bar", "phase"]


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def analyze_trace(path):
    z = np.load(path)
    event_ns = float(z["event_ms"]) * 1e6
    w, k, tag, t = z["warp"], z["k"], z["tag"], z["t"].astype(np.int64)
    ev = ((tag >> 2) & 0x3FF).astype(int)
    order = np.lexsort((k, w))
    w, ev, t = w[order], ev[order], t[order]
    cuts = np.flatnonzero(np.diff(w)) + 1
    gates, cal_k, spans, neg, counts = {"cal_pairs": True}, [], [], 0, set()
    rows = []
    for a0, b0 in zip(np.r_[0, cuts], np.r_[cuts, w.size]):
        gw, e, tt = int(w[a0]), ev[a0:b0], t[a0:b0]
        if not (len(e) >= 4 and e[0] == E["CALG"] and e[1] == E["CALC"] and e[-2] == E["CALG"] and e[-1] == E["CALC"]):
            gates["cal_pairs"] = False
            continue
        g0, c0, g1, c1 = tt[0], tt[1], tt[-2], tt[-1]
        kk = ((g1 - g0) % M32) / ((c1 - c0) % M32)
        cal_k.append(kk)
        spans.append(((c1 - c0) % M32) * kk)
        e, tt = e[2:-2], tt[2:-2]
        ns = ((tt - c0) % M32) * kk
        neg += int((np.diff(ns) < 0).sum())
        blk, wp = divmod(gw, NWARPS)
        if wp != 0 or blk >= NRG * NCH:
            continue
        u_idx, c_idx = np.flatnonzero(e == E["AU0"]), np.flatnonzero(e == E["AC0"])
        counts.add((len(u_idx), len(c_idx)))
        bars = np.flatnonzero(e == E["BAR"])
        kc = NCH - 1 - blk // NRG
        for layer, (a, c) in enumerate(zip(u_idx, c_idx)):
            pos, end = {}, None
            for j in range(a, len(e)):
                n = EV[e[j]]
                if n in ("AU0", "ALD", "AMM", "AQK", "APV", "AREL", "AC0", "ACW", "ACEND") and n not in pos:
                    pos[n] = ns[j]
                if n == "ACEND":
                    end = j
                    break
            if end is None:
                continue
            pb, nb = bars[bars < a], bars[bars > end]
            if not pb.size or not nb.size or len(pos) < 8:
                continue
            if "AMM" not in pos:
                continue
            tb, tn = ns[pb[-1]], ns[nb[0]]
            rows.append({"blk": blk, "tail": kc == NCH - 1, "layer": layer,
                         "bar_to_unit": pos["AU0"] - tb, "ld": pos["ALD"] - pos["AU0"], "qk": pos["AQK"] - pos["ALD"],
                         "qk_mma": pos["AMM"] - pos["ALD"], "softmax": pos["AQK"] - pos["AMM"], "pv": pos["APV"] - pos["AQK"], "store_rel": pos["AREL"] - pos["APV"], "unit": pos["AREL"] - tb,
                         "gap": pos["AC0"] - pos["AREL"], "cwait": pos["ACW"] - pos["AC0"], "comb": pos["ACEND"] - pos["ACW"],
                         "to_bar": tn - pos["ACEND"], "phase": tn - tb})
    kk = np.array(cal_k)
    gates["ns_per_cycle"] = {"median": float(np.median(kk)), "spread_pct": float(100 * (kk.max() / kk.min() - 1))}
    gates["cal_ok"] = bool(gates["ns_per_cycle"]["spread_pct"] < 0.5)
    sp = np.array(spans)
    gates["span_ok"] = bool(0.98 < sp.min() / event_ns and sp.max() / event_ns <= 1.0)
    gates["negative_intervals"] = neg
    gates["unit_counts"] = [list(map(int, c)) for c in sorted(counts)]
    gates["calls_ok"] = counts == {(LAYERS, LAYERS)}
    gates["all_ok"] = bool(gates["cal_pairs"] and gates["cal_ok"] and gates["span_ok"] and neg == 0 and gates["calls_ok"])
    res = {"event_ms": event_ns / 1e6, "gates": gates, "rows": len(rows)}
    for name, sel in (("prefix", [r for r in rows if not r["tail"]]), ("tail", [r for r in rows if r["tail"]])):
        res[name] = {key: {"p50_us": q([r[key] for r in sel], 50) / 1e3, "p90_us": q([r[key] for r in sel], 90) / 1e3} for key in KEYS}
    crit = {}
    for r in rows:
        if r["layer"] not in crit or r["unit"] > crit[r["layer"]]["unit"]:
            crit[r["layer"]] = r
    cl = list(crit.values())
    res["crit"] = {key: q([r[key] for r in cl], 50) / 1e3 for key in KEYS}
    res["crit"]["tail_frac"] = float(np.mean([r["tail"] for r in cl])) if cl else float("nan")
    return res


def main():
    d = sys.argv[1]
    allr = {os.path.basename(p): analyze_trace(p) for p in sorted(glob.glob(os.path.join(d, "fit3_*.npz")))}
    with open(os.path.join(d, "att_stages.json"), "w") as f:
        json.dump(allr, f, indent=1)
    for name, r in allr.items():
        print(f"== {name}  event {r['event_ms']:.3f} ms  rows {r['rows']}  gates {json.dumps(r['gates'])}")
        for part in ("prefix", "tail"):
            print(f"  {part:6s} " + "  ".join(f"{k} {r[part][k]['p50_us']:.3f}/{r[part][k]['p90_us']:.3f}" for k in KEYS))
        print("  crit   " + "  ".join(f"{k} {v:.3f}" for k, v in r["crit"].items()))


if __name__ == "__main__":
    main()
