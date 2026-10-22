"""PI05_ATT_K2 (K2 attention port) vs the current best build: correctness (MegaBench fixture.band, 3 seeds) + interleaved event timing + in-situ L0.
Arms: default; fgu (-DPI05_FG_UMMA=1, current best); fgu_att (+ -DPI05_ATT_K2=1, the target build); att (-DPI05_ATT_K2=1 alone);
*_nullatt = the same build + -DPI05_NULL_ATT (dev-only: attention work skipped, barriers / counters kept; L0 diagnostics, never e2e arms).
In-situ attention L0 per layer = (median_x - median_x_nullatt) * 1e3 / 180. Timing is ABAB-style (random arm order per round), L2 flushed before each
launch, cudaEvent ms. Correctness arms: default, fgu, fgu_att, att (band + repeat + mbarrier timeouts of the build's TMA phases).
Env: PROBE_PREFIX (968), PROBE_OUT, ROUNDS (40), SEEDS."""
import json, os, random, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, pre_flip, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch
from .. import fixture as FX

ROUNDS = int(os.environ.get("ROUNDS", "40"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7301,8917,42811").split(",")]
FGU, ATT, NUL = "-DPI05_FG_UMMA=1", "-DPI05_ATT_K2=1", "-DPI05_NULL_ATT"
V = {"default": (), "fgu": (FGU,), "fgu_att": (FGU, ATT), "att": (ATT,),
     "nullatt": (NUL,), "fgu_nullatt": (FGU, NUL), "fgu_att_nullatt": (FGU, ATT, NUL), "att_nullatt": (ATT, NUL)}
CORR = ("default", "fgu", "fgu_att", "att")


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def timeouts(v, arm):
    n = 0
    if "fgu" in v:
        n += int(arm.ext.fg_timeouts())
    if "att" in v and "nullatt" not in v:
        n += int(arm.ext.att_timeouts())
    return n


def main():
    out = out_dir_from_env("attk2")
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
        for v in CORR:
            a = arms[v]
            a.launch(); torch.cuda.synchronize()
            got = a.actions.clone()
            a.launch(); torch.cuda.synchronize()
            rep = float((a.actions - got).abs().max())
            d = FX.band(want, got, exact, case)
            d["repeat_max_abs"] = rep
            d["mbar_timeouts"] = timeouts(v, a)
            allok &= bool(d["pass"]) and d["mbar_timeouts"] == 0
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
    L0 = lambda a, b: (res[a]["med_ms"] - res[b]["med_ms"]) * 1e3 / 180
    res["L0_default_us_per_layer"] = L0("default", "nullatt")
    res["L0_fgu_us_per_layer"] = L0("fgu", "fgu_nullatt")
    res["L0_fgu_att_us_per_layer"] = L0("fgu_att", "fgu_att_nullatt")
    res["L0_att_us_per_layer"] = L0("att", "att_nullatt")
    res["fgu_att_minus_fgu_ms"] = res["fgu_att"]["med_ms"] - res["fgu"]["med_ms"]
    res["att_minus_default_ms"] = res["att"]["med_ms"] - res["default"]["med_ms"]
    res["fgu_nullatt_minus_fgu_att_nullatt_us_per_layer"] = L0("fgu_nullatt", "fgu_att_nullatt")
    res["mbar_timeouts_after_timing"] = {v: timeouts(v, arms[v]) for v in V}
    rec("attk2_ab", **res)
    print(json.dumps(res, indent=1))


main()
