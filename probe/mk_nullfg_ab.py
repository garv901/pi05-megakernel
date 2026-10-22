"""L0 in-situ ground truth for F+G: megakernel default vs the dev-only -DPI05_NULL_FG build (every barrier / stamp slot kept,
F's GEMM and G's reduce+gelu*up skipped). ABAB interleaved (order flips each round), L2 flushed before each launch, cudaEvent time
of the UNSTAMPED launch (no_timing empty). F+G work cost per layer = (median_default - median_nullfg) / 180.
Also records block-0 globaltimer-stamped launches of both builds (F / G phase means, for the attribution cross-check only).
Env: PROBE_PREFIX (968), PROBE_OUT, ROUNDS (40)."""
import json, os, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, pre_flip, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch
from .mk_analyze import analyze

ROUNDS = int(os.environ.get("ROUNDS", "40"))


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    out = out_dir_from_env("nullfg")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    V = {"default": (), "nullfg": ("-DPI05_NULL_FG",), "default_c64": ("-DPI05_TS_CLOCK64",),
         "nullfg_c64": ("-DPI05_NULL_FG", "-DPI05_TS_CLOCK64")}  # *_c64: block-0 clock64 stamps (cycles), stamped launches only
    arms = {v: build_mk_arm(geo, inputs, "k64", extra=pre_flip(x)) for v, x in V.items()}
    empty = {v: arms[v].no_timing for v in V}
    stamp = {v: torch.zeros(4096, dtype=torch.int64, device="cuda") for v in V}
    fl = l2_flush()
    for a in arms.values():
        for _ in range(3):
            a.launch()
    torch.cuda.synchronize()
    t = {v: [] for v in ("default", "nullfg")}
    ts = {v: [] for v in V}
    st = {v: [] for v in V}
    for r in range(ROUNDS):
        order = ["default", "nullfg"] if r % 2 == 0 else ["nullfg", "default"]
        for v in order:
            a = arms[v]
            a.no_timing = empty[v]
            t[v].append(timed(a, fl))
        if r % 4 == 0:
            for v in (order + ["default_c64", "nullfg_c64"] if r % 8 == 0 else ["nullfg_c64", "default_c64"] + order):
                a = arms[v]
                a.no_timing = stamp[v]
                stamp[v].zero_()
                ts[v].append(timed(a, fl))
                st[v].append(stamp[v].cpu().numpy().copy())
    res = {}
    from .phases import mk_slot, L, NSTEP
    last = mk_slot(NSTEP - 1, L - 1, 9)
    # ns per cycle of the megakernel: globaltimer span (long, >= 100 us) of the gt arm / clock64 span of its c64 twin
    nspc = {}
    for b in ("default", "nullfg"):
        gt_span = statistics.median(float(s[last] - s[0]) for s in st[b])
        c_span = statistics.median(float(s[last] - s[0]) for s in st[b + "_c64"])
        nspc[b] = gt_span / c_span
    for v in ("default", "nullfg"):
        res[v + "_c64"] = {"nspc": nspc[v]}
        ph = []
        for s in st[v + "_c64"]:
            ph.append(analyze(((s - s[0]) * nspc[v]).astype(np.int64))["phases_us_per_layer"])
        res[v + "_c64"]["F_us"] = float(np.mean([p["F_gu"]["mean"] for p in ph]))
        res[v + "_c64"]["G_us"] = float(np.mean([p["G_act"]["mean"] for p in ph]))
    res["mk_clock_dev_default_vs_nullfg"] = nspc["default"] / nspc["nullfg"] - 1.0
    for v in ("default", "nullfg"):
        res[v] = {"med_ms": statistics.median(t[v]), "p10_ms": float(np.percentile(t[v], 10)),
                  "p90_ms": float(np.percentile(t[v], 90)), "n": len(t[v]),
                  "stamped_med_ms": statistics.median(ts[v])}
        ph = [analyze(s)["phases_us_per_layer"] for s in st[v]]
        res[v]["F_us"] = float(np.mean([p["F_gu"]["mean"] for p in ph]))
        res[v]["G_us"] = float(np.mean([p["G_act"]["mean"] for p in ph]))
    d = (res["default"]["med_ms"] - res["nullfg"]["med_ms"]) * 1e3 / 180
    pairs = [(a - b) * 1e3 / 180 for a, b in zip(t["default"], t["nullfg"])]
    res["L0_delta_us_per_layer"] = d
    res["L0_delta_paired_p10_p90"] = [float(np.percentile(pairs, 10)), float(np.percentile(pairs, 90))]
    res["stamp_FG_default_us"] = res["default"]["F_us"] + res["default"]["G_us"]
    res["stamp_FG_delta_us"] = res["stamp_FG_default_us"] - res["nullfg"]["F_us"] - res["nullfg"]["G_us"]
    rec("nullfg_ab", **res)
    np.save(os.path.join(out, "event_ms_default.npy"), np.array(t["default"]))
    np.save(os.path.join(out, "event_ms_nullfg.npy"), np.array(t["nullfg"]))
    print("L0 (default-nullfg)/180 us/layer:", d)


main()
