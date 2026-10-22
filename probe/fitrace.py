"""FlashInfer-profiler port (mk/pi05_mk.cu PI05_FITRACE): bitwise check, overhead A/B and one trace per mode.

Builds the plain k64 arm and the PI05_FITRACE=1 (no fence) / =2 (FlashInfer's __threadfence_block) arms. Each traced binary
is timed with its buffer unbound (code present, stores off: runtime-gated) and bound (stores on). Variants are interleaved in
a fresh random order per round (random.Random(SEED)), L2 flushed before each launch. Then one bound launch per mode is
decoded on the GPU and saved as fit_<mode>.npz (warp, k, tag, t; nonzero entries only) for probe/fitrace_analyze.py.
Env: PROBE_PREFIX (default 128 in common; the sbatch sets 968), MK_VARIANT (default k64), PROBE_OUT."""
from __future__ import annotations

import os
import random
import statistics

import numpy as np
import torch

from .common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch

ROUNDS, PER_ROUND, SEED = 10, 3, 20261019
NWARPS, FIT_CAP = 8, 32768  # mk/pi05_mk.cu
MODES = {"fit1": ("-DPI05_FITRACE=1",), "fit2": ("-DPI05_FITRACE=2",)}


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def decode(buf: torch.Tensor, grid: int) -> dict:
    """Nonzero entries of the FIT area (after the 4096 tstamp slots) as numpy arrays."""
    fit = buf[4096:]
    hdr = int(fit[0].item())
    body = fit[1:1 + FIT_CAP * grid * NWARPS].view(FIT_CAP, grid * NWARPS)
    nz = body != 0
    k, w = nz.nonzero(as_tuple=True)
    raw = body[k, w]
    tag = (raw & 0xFFFFFFFF).to(torch.int64)
    t = (raw >> 32) & 0xFFFFFFFF
    last_k = int(k.max().item()) if k.numel() else -1
    return {"warp": w.cpu().numpy().astype(np.int32), "k": k.cpu().numpy().astype(np.int32),
            "tag": tag.cpu().numpy().astype(np.uint32), "t": t.cpu().numpy().astype(np.uint32),
            "hdr_blocks": hdr & 0xFFFFFFFF, "hdr_groups": hdr >> 32, "last_k": last_k,
            "tstamp": buf[:4096].cpu().numpy()}


def main():
    out = out_dir_from_env("fitrace")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    variant = os.environ.get("MK_VARIANT", "k64")
    plain = build_mk_arm(geo, inputs, variant)
    traced = {m: build_mk_arm(geo, inputs, variant, extra=f) for m, f in MODES.items()}
    grid = torch.cuda.get_device_properties(0).multi_processor_count
    nbuf = 4096 + 1 + FIT_CAP * grid * NWARPS
    off = torch.empty(0, dtype=torch.int64, device="cuda")
    bufs = {m: torch.zeros(nbuf, dtype=torch.int64, device="cuda") for m in MODES}
    fl = l2_flush()

    # bitwise: traced binaries, bound and unbound, against the plain build
    plain.launch(); torch.cuda.synchronize()
    ref = plain.outputs()["actions"].clone()
    for m, a in traced.items():
        for bound in (False, True):
            a.no_timing = bufs[m] if bound else off
            a.launch(); torch.cuda.synchronize()
            rec("equal", mode=m, bound=bound, equal=bool(torch.equal(a.outputs()["actions"], ref)))

    # overhead A/B
    runs = {"plain": (plain, off)}
    for m, a in traced.items():
        runs[m + "-off"] = (a, off)
        runs[m + "-on"] = (a, bufs[m])
    for a, b in runs.values():
        a.no_timing = b
        for _ in range(3):
            a.launch()
    torch.cuda.synchronize()
    times = {v: [] for v in runs}
    rng = random.Random(SEED)
    for r in range(ROUNDS):
        order = list(runs)
        rng.shuffle(order)
        rec("round_order", round=r, order=order)
        for v in order:
            a, b = runs[v]
            a.no_timing = b
            for _ in range(PER_ROUND):
                if b.numel():
                    b.zero_()
                times[v].append(timed(a, fl))
    base = statistics.median(times["plain"])
    for v, ts in times.items():
        med = statistics.median(ts)
        rec("timing", variant=v, median_ms=med, min_ms=min(ts), max_ms=max(ts), n=len(ts),
            vs_plain_pct=100.0 * (med - base) / base)

    # one trace per mode
    for m, a in traced.items():
        a.no_timing = bufs[m]
        bufs[m].zero_()
        timed(a, fl)
        d = decode(bufs[m], grid)
        np.savez_compressed(os.path.join(out, f"fit_{m}.npz"), **d)
        rec("trace", mode=m, entries=int(d["k"].size), last_k=d["last_k"], cap=FIT_CAP,
            hdr_blocks=int(d["hdr_blocks"]), hdr_groups=int(d["hdr_groups"]),
            per_warp_min=int(np.bincount(d["warp"]).min()), per_warp_max=int(np.bincount(d["warp"]).max()))
    print("out_dir", out)


main()
