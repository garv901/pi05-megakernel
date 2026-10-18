"""Traced-vs-untraced overhead and a non-ncu PFTRACE+PTRACE capture (build -DPI05_PTRACE -DPI05_PFTRACE).
Writes pftrace.npz (stamps, ptrace, pf; all relative to stamp[0]) under $PROBE_OUT."""
import os, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, iso_times, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch

NB, G = 1700, 148
PT_N = 4096 + NB * G * 2
PF_N = 180 * 4 * G * 8 * 2
TOTAL = PT_N + PF_N


def timed(arm, fl, n):
    out, bufs = [], []
    for _ in range(n):
        fl.zero_()
        if hasattr(arm, "no_timing") and arm.no_timing.numel() > 0:
            arm.no_timing.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); arm.launch(); b.record(); b.synchronize()
        out.append(a.elapsed_time(b))
        if arm.no_timing.numel() > 0:
            bufs.append(arm.no_timing.cpu().numpy().copy())
    return out, bufs


def main():
    out = out_dir_from_env("mkpftrace")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    plain = build_mk_arm(geo, inputs, "k16")
    pt = build_mk_arm(geo, inputs, "k16", extra=("-DPI05_PTRACE",))
    both = build_mk_arm(geo, inputs, "k16", extra=("-DPI05_PTRACE", "-DPI05_PFTRACE"))
    pt.no_timing = torch.zeros(PT_N, dtype=torch.int64, device="cuda")
    both.no_timing = torch.zeros(TOTAL, dtype=torch.int64, device="cuda")
    fl = l2_flush()
    for o in (plain, pt, both):
        for _ in range(3):
            o.launch()
    torch.cuda.synchronize()
    tp, t1, t2, keep = [], [], [], []
    for r in range(10):          # interleaved rounds
        tp += timed(plain, fl, 3)[0]
        t1 += timed(pt, fl, 3)[0]
        t, b = timed(both, fl, 3)
        t2 += t; keep += b
    med = statistics.median
    rec("pftrace_overhead", plain_ms=tp, ptrace_ms=t1, ptrace_pftrace_ms=t2, plain_median=med(tp), ptrace_median=med(t1),
        both_median=med(t2), ratio_ptrace=med(t1) / med(tp), ratio_both=med(t2) / med(tp))
    arr = np.stack(keep[-6:])
    t0 = arr[:, 0:1]
    pt_ = arr[:, 4096:PT_N].reshape(len(arr), NB, G, 2)
    pf = arr[:, PT_N:TOTAL].reshape(len(arr), 180, 4, G, 8, 2)
    rel = lambda x: np.where(x > 0, x - t0.reshape((-1,) + (1,) * (x.ndim - 1)), -10**15).astype(np.int64)
    np.savez_compressed(os.path.join(out, "pftrace.npz"), stamps=(arr[:, :1635] - t0).astype(np.int64),
                        ptrace=rel(pt_), pf=rel(pf), t0=t0[:, 0])
    print("out_dir", out)


main()
