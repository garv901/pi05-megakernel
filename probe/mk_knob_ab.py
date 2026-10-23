"""Generic knob A/B on the current default build (FG_UMMA + ATT_K2): correctness (MegaBench fixture.band, 3 seeds, repeat,
mbarrier timeouts) + interleaved event timing (random arm order per round, L2 flushed before each launch, cudaEvent ms).
ARMS (env, JSON): {"name": {"extra": ["-D..."], "fuse": int|null, "diag": bool}, ...}; the first arm is the baseline.
diag arms (e.g. NULL_* builds) are timed only (no correctness gate). Builds with equal extra share nothing (one .so per arm).
Env: PROBE_PREFIX (968), PROBE_OUT, ROUNDS (40), SEEDS, ARMS."""
import json, os, random, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch
from .. import fixture as FX

ROUNDS = int(os.environ.get("ROUNDS", "40"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "7301,8917,42811").split(",")]
ARMS = json.loads(os.environ["ARMS"])


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def timeouts(arm):
    n = 0
    for f in ("fg_timeouts", "att_timeouts"):
        if hasattr(arm.ext, f):
            n += int(getattr(arm.ext, f)())
    return n


def main():
    out = out_dir_from_env(os.environ.get("TAG", "knob-ab"))
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    arms = {v: build_mk_arm(geo, inputs, "k64", extra=tuple(c.get("extra", ())), fuse=c.get("fuse")) for v, c in ARMS.items()}
    for v, a in arms.items():
        print("config", v, "|", a.ext.build_config(), "| fuse", a.fuse, flush=True)
        rec("config", variant=v, config=a.ext.build_config(), fuse=a.fuse)
    corr = [v for v, c in ARMS.items() if not c.get("diag")]
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
        for v in corr:
            a = arms[v]
            a.launch(); torch.cuda.synchronize()
            got = a.actions.clone()
            a.launch(); torch.cuda.synchronize()
            d = FX.band(want, got, exact, case)
            d["repeat_max_abs"] = float((a.actions - got).abs().max())
            d["mbar_timeouts"] = timeouts(a)
            if base is None:
                base = got
            d["vs_base_max_abs"] = float((got - base).abs().max())
            allok &= bool(d["pass"]) and d["mbar_timeouts"] == 0
            rec("correctness", variant=v, seed=seed, **d)
            print("correctness", v, seed, {k: d[k] for k in ("pass", "repeat_max_abs", "mbar_timeouts", "vs_base_max_abs")}, flush=True)
    rec("correctness_all", ok=allok)
    print("correctness_all", allok, flush=True)
    fl = l2_flush()
    for a in arms.values():
        for _ in range(3):
            a.launch()
    torch.cuda.synchronize()
    t = {v: [] for v in arms}
    rng = random.Random(1)
    for r in range(ROUNDS):
        order = list(arms)
        rng.shuffle(order)
        for v in order:
            t[v].append(timed(arms[v], fl))
    b0 = next(iter(arms))
    res = {v: {"med_ms": statistics.median(t[v]), "p10_ms": float(np.percentile(t[v], 10)), "p90_ms": float(np.percentile(t[v], 90)), "n": len(t[v])} for v in arms}
    for v in arms:
        res[v]["vs_base_pct"] = 100.0 * (res[v]["med_ms"] - res[b0]["med_ms"]) / res[b0]["med_ms"]
        res[v]["vs_base_us_per_layer"] = (res[v]["med_ms"] - res[b0]["med_ms"]) * 1e3 / 180
        np.save(os.path.join(out, f"event_ms_{v}.npy"), np.array(t[v]))
    res["mbar_timeouts_after_timing"] = {v: timeouts(arms[v]) for v in arms}
    rec("knob_ab", **res)
    print(json.dumps(res, indent=1))


main()
