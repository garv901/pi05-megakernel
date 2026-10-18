"""Stage 0: feasibility + baseline timings (run on a GPU node via slurm/stage0.sbatch)."""
import json, os, statistics, sys, time
import numpy as np
import torch
from .common import (Launcher, Recorder, build_graph_arm, build_mk_arm, iso_times, l2_flush,
                     make_geo_inputs, out_dir_from_env, setup_torch)
sys.path.insert(0, "/raid/garv901/vibe-megakernel")
from megabench.sota.timing import ClockLogger, hbm_copy_bw


def main():
    out = out_dir_from_env("stage0")
    rec = Recorder(out)
    setup_torch()
    clk = ClockLogger(0)
    rec("env", torch=torch.__version__, gpu=torch.cuda.get_device_name(), clock=clk.snapshot(),
        job=os.environ.get("SLURM_JOB_ID"))
    rec("hbm_copy_bw", **hbm_copy_bw())
    geo, inputs = make_geo_inputs()
    t0 = time.time()
    g = build_graph_arm(geo, inputs)
    rec("build_graph", seconds=time.time() - t0)
    mk = build_mk_arm(geo, inputs, "k16")
    mkt = build_mk_arm(geo, inputs, "k16", tstamp=True)
    rec("build_mk", ok=True)
    fl = l2_flush()
    # correctness smoke: tuned graph vs megakernel actions
    g.set_variant("graph-pdl"); g.launch(); mk.launch(); torch.cuda.synchronize()
    rec("smoke_agree", max_abs_graph_vs_mk=float((g.outputs()["actions"] - mk.outputs()["actions"]).abs().max()))
    arms = {"graph-pdl": g, "graph-nopdl": g, "mk": mk, "mk-tstamp": mkt}
    res = {k: [] for k in arms}
    for r in range(5):
        for name, o in arms.items():
            if name.startswith("graph"):
                o.set_variant(name)
            L = type("L", (), {"launch": lambda self, o=o: o.launch()})()
            for _ in range(2):
                L.launch()
            torch.cuda.synchronize()
            res[name] += iso_times(L, 10, fl)
        rec("round", r=r, clock=clk.snapshot())
    for k, v in res.items():
        rec("timing", arm=k, median_ms=statistics.median(v), min_ms=min(v), max_ms=max(v), n=len(v), samples=v)
    ts = mkt.no_timing.cpu().numpy()
    np.save(os.path.join(out, "tstamp_raw.npy"), ts)
    nz = int((ts > 0).sum())
    rec("tstamp", nonzero=nz, first=int(ts[0]), last_nonzero_idx=int(np.nonzero(ts)[0].max()),
        span_us=float(ts[np.nonzero(ts)[0].max()] - ts[0]) / 1e3)
    print("out_dir", out)


main()
