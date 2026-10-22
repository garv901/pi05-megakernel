"""Xtrace step 1, CUDA k64: run the shipped extension with its kernel swapped for Xtrace-spliced copies (offline splice,
probe/xtrace/splice.py) through the LD_PRELOAD shim coop_shim.so, which interposes cudaLaunchCooperativeKernel.

Slots: reload = the shipped cubin re-loaded through the shim (isolates module-load / launch-path effects),
xt_gpu / xt_anchor = the spliced cubins. (1) torch.equal of each slot's actions vs the plain launch, and the shim's
launch count; (2) L2-flushed timing, plain / reload / xt_gpu / xt_anchor in a fresh random order per round;
(3) raw trace buffers (zeroed, one launch each) to xt_<clock>_<i>.npz for CPU decode (xt_analyze.py, gwatch venv).
Run: LD_PRELOAD=runs/xtrace-cpu/coop_shim.so python -m megabench.sota.pi05.probe.xtrace.xt_cuda
Env: XT_DIR (spliced cubins + json), PROBE_PREFIX (sbatch: 968), PROBE_OUT."""
from __future__ import annotations

import ctypes
import json
import os
import random
import statistics

import numpy as np
import torch

from .bank4 import upload
from ..common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch

ROUNDS, PER_ROUND, SEED = 10, 3, 20261019
TRACES = {"gpu": 3, "anchor": 2}


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    xt_dir = os.environ.get("XT_DIR", "runs/xtrace-cpu")
    out = out_dir_from_env("xtrace-cuda")
    rec = Recorder(out)
    setup_torch()
    shim = ctypes.CDLL(None)  # the LD_PRELOADed shim's symbols are global
    shim.xt_set.argtypes = [ctypes.c_int, ctypes.c_ulonglong]
    shim.xt_count.restype = ctypes.c_ulonglong
    shim.xt_set_extra.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong)]

    geo, inputs = make_geo_inputs()
    arm = build_mk_arm(geo, inputs, "k64")
    arm.launch(); torch.cuda.synchronize()  # loads the extension's module (and the runtime) before cuModuleLoadData
    ref = arm.outputs()["actions"].clone()

    metas, slots, bufs, tables = {}, {}, {}, {}
    for name, path in (("reload", f"{xt_dir}/pi05_mk.sm_100.cubin"), ("xt_gpu", f"{xt_dir}/cuda/xtrace_gpu.cubin"),
                       ("xt_anchor", f"{xt_dir}/cuda/xtrace_anchor.cubin")):
        s = shim.xt_load(path.encode(), b"_Z15pi05_megakernel6Params")
        rec("load", slot=name, path=path, ret=s)
        assert 0 <= s < 8, (name, s)
        slots[name] = s
        if name.startswith("xt_"):
            metas[name] = json.load(open(path[:-len(".cubin")] + ".json"))
            bufs[name] = torch.zeros(metas[name]["total_bytes"], dtype=torch.uint8, device="cuda")
            tables[name], extra = upload(metas[name])
            assert shim.xt_set_extra(s, len(extra), (ctypes.c_ulonglong * max(1, len(extra)))(*extra)) == 0
        else:
            bufs[name] = torch.zeros(16, dtype=torch.uint8, device="cuda")

    def use(name):
        shim.xt_set(slots.get(name, -1), bufs[name].data_ptr() if name in bufs else 0)

    for name in ["plain"] + list(slots):
        use(name)
        c0 = shim.xt_count()
        arm.actions.zero_()
        arm.launch(); torch.cuda.synchronize()
        rec("equal", slot=name, equal=bool(torch.equal(arm.outputs()["actions"], ref)),
            shim_launches=int(shim.xt_count() - c0))

    fl = l2_flush()
    variants = ["plain"] + list(slots)
    times = {v: [] for v in variants}
    rng = random.Random(SEED)
    for v in variants:
        use(v)
        for _ in range(3):
            arm.launch()
    torch.cuda.synchronize()
    for _ in range(ROUNDS):
        order = list(variants)
        rng.shuffle(order)
        for v in order:
            use(v)
            times[v] += [timed(arm, fl) for _ in range(PER_ROUND)]
    base = statistics.median(times["plain"])
    for v in variants:
        m = statistics.median(times[v])
        rec("timing", slot=v, median_ms=m, min_ms=min(times[v]), over_plain=m / base - 1, n=len(times[v]),
            all_ms=times[v])

    for clock, n in TRACES.items():
        name = f"xt_{clock}"
        use(name)
        for i in range(n):
            bufs[name].zero_()
            ms = timed(arm, fl)
            raw = bufs[name].cpu().numpy()
            np.savez_compressed(os.path.join(out, f"xt_{clock}_{i}.npz"), raw=raw)
            rec("trace", clock=clock, i=i, ms=ms, nonzero_bytes=int(np.count_nonzero(raw)))
    use("plain")
    with open(os.path.join(out, "xt_meta.json"), "w") as f:
        json.dump(metas, f, default=str)
    print("out_dir", out)


main()
