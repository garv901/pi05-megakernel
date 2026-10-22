"""Per-phase A/B from the block-0 barrier-exit stamps of the plain (benchmarked) binary: the stamp store is runtime-gated,
so no rebuild. Variants interleaved per round in a fresh random order each round (random.Random(SEED), the per-round order is recorded as a
"round_order" record: a fixed order measured the first variant +60-90 us), L2 flushed before each launch.
Env: PROBE_PREFIX, MK_VARIANTS (comma list, e.g. k64,k64-nolh). Writes records.jsonl (round orders, per-variant phase means +
stamped / unstamped launch times) and stamps_<variant>.npy."""
import os, random, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch
from .mk_analyze import analyze
from .phases import PHASES

ROUNDS, PER_ROUND, SEED = 10, 3, 20261019


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    out = out_dir_from_env("stamps-ab")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    vs = os.environ.get("MK_VARIANTS", "k64,k64-nolh").split(",")
    arms = {v: build_mk_arm(geo, inputs, v) for v in vs}
    off = {v: arms[v].no_timing for v in vs}
    on = {v: torch.zeros(4096, dtype=torch.int64, device="cuda") for v in vs}
    fl = l2_flush()
    for a in arms.values():
        for _ in range(3):
            a.launch()
    torch.cuda.synchronize()
    t_on, t_off, stamps = ({v: [] for v in vs} for _ in range(3))
    rng = random.Random(SEED)
    for r in range(ROUNDS):
        order = list(vs)
        rng.shuffle(order)
        rec("round_order", round=r, order=order)
        for v in order:
            a = arms[v]
            a.no_timing = off[v]
            t_off[v] += [timed(a, fl) for _ in range(PER_ROUND)]
            a.no_timing = on[v]
            for _ in range(PER_ROUND):
                on[v].zero_()
                t_on[v].append(timed(a, fl))
                stamps[v].append(on[v].cpu().numpy().copy())
    for v in vs:
        arr = np.stack(stamps[v])
        np.save(os.path.join(out, f"stamps_{v}.npy"), arr)
        per = [analyze(s) for s in arr]
        ph = {p: float(np.mean([x["phases_us_per_layer"][p]["mean"] for x in per])) for p in PHASES}
        rec("phases", variant=v, prefix=geo.prefix, phase_mean_us=ph,
            tail_us_per_step=float(np.mean([x["tail_us_per_step"] for x in per])),
            prologue_us=float(np.mean([x["prologue_us"] for x in per])),
            last_stamp_us=float(np.mean([x["last_stamp_us"] for x in per])),
            stamped_ms=statistics.median(t_on[v]), unstamped_ms=statistics.median(t_off[v]), n=len(per))
    print("out_dir", out)


main()
