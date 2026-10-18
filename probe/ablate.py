"""Stage 4: megakernel ablations (timings, L2-flushed, interleaved rounds). Each config = (name, variant, -D flags, fuse)."""
import os, statistics, sys
import torch
from .common import Recorder, build_mk_arm, iso_times, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch

CONFIGS = [
    ("base", "k16", (), None),
    ("nopf", "k16", ("-DPI05_ABL_NOPF",), None),
    ("wpre0", "k16", ("-DPI05_WPRE=0",), None),
    ("wpre0_nopf", "k16", ("-DPI05_WPRE=0", "-DPI05_ABL_NOPF"), None),
    ("fuse16_allbarrier", "k16", (), 16),
    ("fuse19_AB", "k16", (), 19),
    ("fuse22_FG", "k16", (), 22),
    ("fuse23_AB_FG", "k16", (), 23),
    ("fuse26_backoff", "k16", (), 26),
    ("barrier2", "k16", ("-DPI05_BARRIER=2",), None),
    ("k32", "k32", (), None),
]

def main():
    out = out_dir_from_env("stage4"); rec = Recorder(out); setup_torch()
    geo, inputs = make_geo_inputs()
    fl = l2_flush()
    arms = {}
    ref = None
    for name, var, extra, fuse in CONFIGS:
        try:
            a = build_mk_arm(geo, inputs, var, extra=extra, fuse=fuse)
            a.launch(); torch.cuda.synchronize()
            act = a.outputs()["actions"].clone()
            if ref is None: ref = act
            rec("build", name=name, ok=True, max_abs_vs_base=float((act - ref).abs().max()))
            arms[name] = a
        except Exception as e:
            rec("build", name=name, ok=False, err=repr(e)[:300])
    res = {k: [] for k in arms}
    for r in range(5):
        for name, a in arms.items():
            L = type("L", (), {"launch": lambda self, a=a: a.launch()})()
            for _ in range(2): L.launch()
            torch.cuda.synchronize()
            res[name] += iso_times(L, 10, fl)
    for k, v in res.items():
        rec("timing", name=k, median_ms=statistics.median(v), min_ms=min(v), max_ms=max(v), n=len(v), samples=v)
    print("out_dir", out)
main()
