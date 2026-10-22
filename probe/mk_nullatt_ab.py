"""L0 in-situ ground truth for the attention phase (C + C3): megakernel vs the dev-only -DPI05_NULL_ATT build (every barrier, every counter
release / wait, every L2 prefetch and every stamp slot kept; C's loads / MMAs / softmax / stores, the pre-B K/V prefill and the C3 combine
skipped). Two pairs: the FG_UMMA build (the current best, primary) and the plain default build. ABAB interleaved (order flips each round), L2
flushed before each launch, cudaEvent time of the UNSTAMPED launch (no_timing empty). Attention cost per layer = (median_x - median_nullatt_x) / 180.
Also records block-0 globaltimer-stamped launches (C_attn phase mean per arm; attribution cross-check only).
Env: PROBE_PREFIX (968), PROBE_OUT, ROUNDS (40)."""
import json, os, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch
from .mk_analyze import analyze

ROUNDS = int(os.environ.get("ROUNDS", "40"))
V = {"umma": ("-DPI05_FG_UMMA=1",), "umma_nullatt": ("-DPI05_FG_UMMA=1", "-DPI05_NULL_ATT"),
     "default": (), "default_nullatt": ("-DPI05_NULL_ATT",)}
PAIRS = (("umma", "umma_nullatt"), ("default", "default_nullatt"))


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    out = out_dir_from_env("nullatt")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    arms = {v: build_mk_arm(geo, inputs, "k64", extra=x) for v, x in V.items()}
    for v, a in arms.items():
        print("config", v, "|", a.ext.build_config(), flush=True)
        rec("config", variant=v, config=a.ext.build_config())
    empty = {v: arms[v].no_timing for v in V}
    stamp = {v: torch.zeros(4096, dtype=torch.int64, device="cuda") for v in V}
    fl = l2_flush()
    for a in arms.values():
        for _ in range(3):
            a.launch()
    torch.cuda.synchronize()
    t = {v: [] for v in V}
    st = {v: [] for v in V}
    rng = np.random.default_rng(5)
    names = list(V)
    for r in range(ROUNDS):
        if r % 2 == 0:
            order_prev = [str(x) for x in rng.permutation(names)]  # random order, then its reverse (ABBA-style)
            order = order_prev
        else:
            order = order_prev[::-1]
        for v in order:
            a = arms[v]
            a.no_timing = empty[v]
            t[v].append(timed(a, fl))
        if r % 4 == 0:
            for v in order[::-1]:
                a = arms[v]
                a.no_timing = stamp[v]
                stamp[v].zero_()
                timed(a, fl)
                st[v].append(stamp[v].cpu().numpy().copy())
    res = {}
    for v in V:
        ph = [analyze(s)["phases_us_per_layer"] for s in st[v]]
        res[v] = {"med_ms": statistics.median(t[v]), "p10_ms": float(np.percentile(t[v], 10)), "p90_ms": float(np.percentile(t[v], 90)),
                  "n": len(t[v]), "C_attn_stamp_us": float(np.mean([p["C_attn"]["mean"] for p in ph])),
                  "phases_us": {k: float(np.mean([p[k]["mean"] for p in ph])) for k in ph[0]}}
    for x, n in PAIRS:
        d = (res[x]["med_ms"] - res[n]["med_ms"]) * 1e3 / 180
        pairs = [(a - b) * 1e3 / 180 for a, b in zip(t[x], t[n])]
        res[f"L0_{x}_us_per_layer"] = d
        res[f"L0_{x}_paired_p10_p90"] = [float(np.percentile(pairs, 10)), float(np.percentile(pairs, 90))]
        res[f"stamp_phase_delta_{x}_us"] = {k: res[x]["phases_us"][k] - res[n]["phases_us"][k] for k in res[x]["phases_us"]}  # block-0 gt stamps, attribution only
        res[f"stamp_C_{x}_delta_us"] = res[x]["C_attn_stamp_us"] - res[n]["C_attn_stamp_us"]
        print(f"L0 ({x} - {n})/180 us/layer: {d:.4f}  paired p10/p90 {res[f'L0_{x}_paired_p10_p90']}  (e2e {res[x]['med_ms']:.4f} vs {res[n]['med_ms']:.4f} ms)")
    res["L0_delta_us_per_layer"] = res["L0_umma_us_per_layer"]  # primary: the FG_UMMA build
    rec("nullatt_ab", **res)
    for v in V:
        np.save(os.path.join(out, f"event_ms_{v}.npy"), np.array(t[v]))


main()
