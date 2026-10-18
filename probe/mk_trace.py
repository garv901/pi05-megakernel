"""Megakernel per-CTA barrier trace (build -DPI05_PTRACE): arrive/exit globaltimer of every CTA at
every barrier, plus block-0 tstamp.  Writes ptrace.npz and a traced-vs-untraced timing record."""
import os, statistics, sys
import numpy as np
import torch
from .common import Recorder, build_mk_arm, iso_times, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch

NB = 1700  # barrier slots allocated (1635 used)


def main():
    out = out_dir_from_env("mktrace")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    variant = os.environ.get("MK_VARIANT", "k16")
    flags = tuple(os.environ.get("MK_FLAGS", "").split())
    plain = build_mk_arm(geo, inputs, variant, extra=flags)
    traced = build_mk_arm(geo, inputs, variant, extra=flags + ("-DPI05_PTRACE",))
    n = 148
    traced.no_timing = torch.zeros(4096 + NB * n * 2, dtype=torch.int64, device="cuda")
    fl = l2_flush()
    for o in (plain, traced):
        for _ in range(3):
            o.launch()
    torch.cuda.synchronize()
    tp, tt = [], []
    keep = []
    for r in range(10):
        tp += iso_times(type("L", (), {"launch": lambda self: plain.launch()})(), 3, fl)
        for _ in range(3):
            fl.zero_()
            traced.no_timing.zero_()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); traced.launch(); b.record(); b.synchronize()
            tt.append(a.elapsed_time(b))
            keep.append(traced.no_timing.cpu().numpy().copy())
    rec("traced_vs_plain", variant=variant, plain_median_ms=statistics.median(tp), traced_median_ms=statistics.median(tt),
        ratio=statistics.median(tt) / statistics.median(tp), plain=tp, traced=tt)
    sel = keep[-5:]
    arr = np.stack(sel)  # [launch, 4096 + NB*n*2]
    t0 = arr[:, 0:1]
    stamps = arr[:, :1635]
    pt = arr[:, 4096:4096 + NB * n * 2].reshape(len(sel), NB, n, 2)
    # make relative int32 ns
    rel = np.where(pt > 0, pt - t0[:, :, None, None], -10**15).astype(np.int64)
    np.savez_compressed(os.path.join(out, "ptrace.npz"), stamps=(stamps - t0).astype(np.int64), ptrace=rel,
                        t0=t0[:, 0], variant=variant)
    rec("saved", shape=list(rel.shape), nonzero_frac=float((pt > 0).mean()))
    print("out_dir", out)


main()
