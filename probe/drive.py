"""Profiling driver: python -m megabench.sota.pi05.probe.drive ARM [--reps N] [--warm W] [--flush]
ARM: mk[:k16] | graph-pdl | graph-nopdl | compile (torch-compile; set PROBE_PREFIX / PROBE_TUNED for non-P128).  Launches inside cudaProfilerStart/Stop (for nsys / ncu
--profile-from-start off).  --mk-flags "-DX=1 ..." adds -D flags to the megakernel build."""
import argparse, os, sys
import torch
from .common import Launcher, l2_flush, make_geo_inputs, setup_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--flush", action="store_true")
    ap.add_argument("--mk-flags", default="")
    ap.add_argument("--tstamp", action="store_true")
    ap.add_argument("--tstamp-size", type=int, default=4096)
    ap.add_argument("--save-tstamp", default="")
    ap.add_argument("--range-all", action="store_true", help="one profiler range around all reps (nsys)")
    ap.add_argument("--chrome-trace", default="", help="write a torch.profiler chrome trace (Perfetto-readable) of --reps launches")
    a = ap.parse_args()
    setup_torch()
    geo, inputs = make_geo_inputs(int(os.environ.get("PROBE_PREFIX", "128")))
    kw = {}
    if a.arm.startswith("mk"):
        kw = dict(tstamp=a.tstamp, extra=tuple(a.mk_flags.split()), tstamp_size=a.tstamp_size)
    if a.arm == "compile":
        from ..compile_arm import CompileArm
        L = type("L", (), {})()
        L.obj = CompileArm(geo, inputs, "cuda")
        L.obj.set_variant("max-autotune")
        L.launch = L.obj.launch
    else:
        L = Launcher(a.arm, geo, inputs, **kw)
    fl = l2_flush() if a.flush else None
    for _ in range(a.warm):
        L.launch()
    torch.cuda.synchronize()
    if a.chrome_trace:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.reps):
                if fl is not None:
                    fl.zero_()
                    torch.cuda.synchronize()
                L.launch()
                torch.cuda.synchronize()
        prof.export_chrome_trace(a.chrome_trace)
        print("drive trace done", a.arm, flush=True)
        return
    cu = torch.cuda.cudart()
    if a.range_all:
        cu.cudaProfilerStart()
    for _ in range(a.reps):
        if fl is not None:
            fl.zero_()
            torch.cuda.synchronize()
        if not a.range_all:
            cu.cudaProfilerStart()   # start/stop per rep: the L2 flush stays outside the profiled range
        L.launch()
        torch.cuda.synchronize()
        if not a.range_all:
            cu.cudaProfilerStop()
    if a.range_all:
        cu.cudaProfilerStop()
    if a.save_tstamp and a.tstamp:
        import numpy as np
        ts = L.obj.no_timing.cpu().numpy()
        # "{ts}" in the name is replaced by stamp[1] (epoch ns): lets ncu --replay-mode application keep one file per replay pass
        np.save(a.save_tstamp.replace("{ts}", str(int(ts[1]))), ts)
    print("drive done", a.arm, flush=True)


main()
