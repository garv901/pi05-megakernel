"""PI05_D_UMMA (tcgen05 D o_proj on the H design, 8 or 16 K slices) vs the default (k64 with PI05_H_UMMA, job 9423): correctness (fixture.band, 3 seeds, mbarrier
timeouts, repeat diff) + interleaved timing. Arms (explicit flags, so the meaning survives a default flip): default (-DPI05_D_UMMA=0), d16 / d8 (-DPI05_D_UMMA=16 / 8), ctl
(-DPI05_D_UMMA=0 -DPI05_SO_CTL=1: an unused define, the same code as default in a different .so; the .so-to-.so noise control). Timing: random arm order per round, L2 flushed before each
launch, cudaEvent ms. Env: PROBE_OUT, ROUNDS (40), SEEDS."""
import json, os, random, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch
from .. import fixture as FX

ROUNDS = int(os.environ.get("ROUNDS", "40"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7301,8917,42811").split(",")]
V = {"default": ("-DPI05_D_UMMA=0",), "d16": ("-DPI05_D_UMMA=16",), "d8": ("-DPI05_D_UMMA=8",), "ctl": ("-DPI05_D_UMMA=0", "-DPI05_SO_CTL=1")}


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    out = out_dir_from_env("dumma")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    arms = {v: build_mk_arm(geo, inputs, "k64", extra=x) for v, x in V.items()}
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
        base = None
        for v in ("default", "ctl", "d16", "d8"):
            a = arms[v]
            a.launch(); torch.cuda.synchronize()
            got = a.actions.clone()
            a.launch(); torch.cuda.synchronize()
            rep = float((a.actions - got).abs().max())
            d = FX.band(want, got, exact, case)
            d["repeat_max_abs"] = rep
            d["finite"] = bool(torch.isfinite(got).all())
            if v == "default":
                base = got
            d["max_abs_vs_default"] = float((got - base).abs().max())
            to = int(a.ext.fg_timeouts()) + int(a.ext.att_timeouts())
            d["mbar_timeouts"] = to
            allok &= bool(d["pass"]) and to == 0 and d["finite"]
            rec("correctness", variant=v, seed=seed, **d)
            print("correctness", v, seed, d, flush=True)
    rec("correctness_all", ok=allok)
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
    for v in ("d16", "d8"):
        res[f"{v}_minus_default_us_per_layer"] = (res[v]["med_ms"] - res["default"]["med_ms"]) * 1e3 / 180
        res[f"{v}_vs_default_pct"] = (res[v]["med_ms"] / res["default"]["med_ms"] - 1) * 100
    res["ctl_vs_default_pct"] = (res["ctl"]["med_ms"] / res["default"]["med_ms"] - 1) * 100
    res["mbar_timeouts_after_timing"] = sum(int(arms[v].ext.fg_timeouts()) + int(arms[v].ext.att_timeouts()) for v in ("d16", "d8"))
    rec("dumma_ab", **res)
    print(json.dumps(res, indent=1))


main()
