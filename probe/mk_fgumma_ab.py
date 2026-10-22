"""PI05_FG_UMMA (tcgen05 F/G) vs default: correctness (MegaBench fixture.band, 3 seeds) + timing.
Timed e2e arms (default, umma, pad) are default flags (+ the one knob); nullfg / umma_nullfg are L0-difference diagnostics only.
Arms: default, umma (-DPI05_FG_UMMA=1), pad (default + -DPI05_SMEM_PAD: smem raised to the UMMA footprint only), nullfg (default -DPI05_NULL_FG),
umma_nullfg (-DPI05_FG_UMMA=1 -DPI05_NULL_FG: E + E->F barrier kept, F work and F->G barrier dropped). Timing is ABAB-style (random arm order per round),
L2 flushed before each unstamped launch, cudaEvent ms. L0 per layer = (median_x - median_null_x) * 1e3 / 180.
Env: PROBE_PREFIX (968), PROBE_OUT, ROUNDS (40), SEEDS."""
import json, os, random, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, pre_flip, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch
from .. import fixture as FX

ROUNDS = int(os.environ.get("ROUNDS", "40"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7301,8917,42811").split(",")]
V = {"default": (), "umma": ("-DPI05_FG_UMMA=1",), "pad": ("-DPI05_SMEM_PAD",), "nullfg": ("-DPI05_NULL_FG",),
     "umma_nullfg": ("-DPI05_FG_UMMA=1", "-DPI05_NULL_FG")}


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    out = out_dir_from_env("fgumma")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    arms = {v: build_mk_arm(geo, inputs, "k64", extra=pre_flip(x)) for v, x in V.items()}
    for v, a in arms.items():
        print("config", v, "|", a.ext.build_config(), flush=True)
        rec("config", variant=v, config=a.ext.build_config())
    case = FX.make_case(geo)
    allok = True
    for n, seed in enumerate(SEEDS):
        if n:
            fresh = FX.make_inputs(geo, seed, "cuda")
            for k, v in fresh.items():
                inputs[k].copy_(v)
            del fresh
            torch.cuda.synchronize()
        want = FX.reference(geo, inputs)["actions"]
        exact = FX.oracle(geo, inputs)["actions"]
        for v in ("default", "umma", "pad"):
            a = arms[v]
            a.launch(); torch.cuda.synchronize()
            got = a.actions.clone()
            a.launch(); torch.cuda.synchronize()
            rep = float((a.actions - got).abs().max())
            d = FX.band(want, got, exact, case)
            d["repeat_max_abs"] = rep
            to = arms[v].ext.fg_timeouts() if v == "umma" else 0
            d["mbar_timeouts"] = to
            allok &= bool(d["pass"]) and to == 0
            rec("correctness", variant=v, seed=seed, **d)
            print("correctness", v, seed, d, flush=True)
    rec("correctness_all", ok=allok)
    # restore seed-0 inputs for timing (any inputs fine); warm all arms
    fl = l2_flush()
    for a in arms.values():
        for _ in range(3):
            a.launch()
    torch.cuda.synchronize()
    t = {v: [] for v in V}
    rng = random.Random(1)
    for r in range(ROUNDS):
        order = list(V)
        rng.shuffle(order)
        for v in order:
            t[v].append(timed(arms[v], fl))
    res = {v: {"med_ms": statistics.median(t[v]), "p10_ms": float(np.percentile(t[v], 10)), "p90_ms": float(np.percentile(t[v], 90)), "n": len(t[v])} for v in V}
    for v in V:
        np.save(os.path.join(out, f"event_ms_{v}.npy"), np.array(t[v]))
    res["L0_default_us_per_layer"] = (res["default"]["med_ms"] - res["nullfg"]["med_ms"]) * 1e3 / 180
    res["L0_umma_us_per_layer"] = (res["umma"]["med_ms"] - res["umma_nullfg"]["med_ms"]) * 1e3 / 180
    res["L0_pad_us_per_layer_vs_default_total"] = (res["pad"]["med_ms"] - res["default"]["med_ms"]) * 1e3 / 180
    res["umma_minus_default_us_per_layer"] = (res["umma"]["med_ms"] - res["default"]["med_ms"]) * 1e3 / 180
    res["umma_nullfg_minus_nullfg_us_per_layer"] = (res["umma_nullfg"]["med_ms"] - res["nullfg"]["med_ms"]) * 1e3 / 180
    res["mbar_timeouts_after_timing"] = int(arms["umma"].ext.fg_timeouts())
    rec("fgumma_ab", **res)
    print(json.dumps(res, indent=1))


main()
