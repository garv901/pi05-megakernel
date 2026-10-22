"""MB2 / MB3 on the shipped CUDA k64 kernel and three same-source variants, Xtrace-spliced offline (splice.py ... gemm:
grid-barrier arrive / exit plus five gemm_phase sites) and launched in place of the k64 extension's kernel through the
LD_PRELOAD shim coop_shim.so (same Params; the variants differ only in -D knobs).

Builds (runs/mb23-cpu/build.sh): base = shipped flags; rot37 = PI05_TILE_ROT=37 (GEMM tiles on other SMs, MB2);
lh1 = PI05_LOAD_HOIST=1 and wpre0 = PI05_WPRE=0 (next-tile weight issue before the stores / first-tile issue after the
barrier, MB3). Slots: each build plain (<name>) and spliced (<name>_x).
(1) torch.equal of each slot's actions vs the extension's launch; (2) L2-flushed timing, every slot and the extension
(plain) in a fresh random order per round; (3) traces, L2 flushed, launch order shuffled per round: base_x and rot37_x
TRACE_ROUNDS times, lh1_x and wpre0_x the first TRACE_SHORT rounds. A trace is saved trimmed (each warp's slab up to
the deepest record any warp wrote) as xt_<slot>_<i>.npz {slab, nw, slab_bytes, total}; xt_mb_analyze.py decodes them.
Run: LD_PRELOAD=runs/xtrace-cpu/coop_shim.so python -m megabench.sota.pi05.probe.xtrace.xt_mb
Env: XT_DIR (default runs/mb23-cpu), PROBE_PREFIX (sbatch: 968), PROBE_OUT, XT_TRACE_ROUNDS (20), XT_TRACE_SHORT (5)."""
from __future__ import annotations

import ctypes
import glob
import json
import os
import random
import statistics

import numpy as np
import torch

from .bank4 import upload
from ..common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch

ROUNDS, PER_ROUND, SEED = 10, 3, 20261020
BUILDS = ("base", "rot37", "lh1", "wpre0")
KERNEL = b"_Z15pi05_megakernel6Params"


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    xt_dir = os.environ.get("XT_DIR", "runs/mb23-cpu")
    n_long = int(os.environ.get("XT_TRACE_ROUNDS", "20"))
    n_short = int(os.environ.get("XT_TRACE_SHORT", "5"))
    out = out_dir_from_env("xt-mb")
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

    slots, bufs, metas, keep = {}, {}, {}, []
    for b in BUILDS:
        spliced = glob.glob(f"{xt_dir}/{b}/xtrace_gpu*_gemm.cubin")
        assert len(spliced) == 1, (b, spliced)
        for name, path in ((b, f"{xt_dir}/{b}.cubin"), (b + "_x", spliced[0])):
            s = shim.xt_load(path.encode(), KERNEL)
            rec("load", slot=name, path=path, ret=s)
            assert 0 <= s < 8, (name, s)
            slots[name] = s
            if name.endswith("_x"):
                metas[name] = json.load(open(path[:-len(".cubin")] + ".json"))
                bufs[name] = torch.zeros(metas[name]["total_bytes"], dtype=torch.uint8, device="cuda")
                table, extra = upload(metas[name])
                keep.append(table)
                assert shim.xt_set_extra(s, len(extra), (ctypes.c_ulonglong * max(1, len(extra)))(*extra)) == 0

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
    for r in range(ROUNDS):
        order = list(variants)
        rng.shuffle(order)
        rec("order", round=r, order=order)
        for v in order:
            use(v)
            times[v] += [timed(arm, fl) for _ in range(PER_ROUND)]
    base = statistics.median(times["plain"])
    for v in variants:
        m = statistics.median(times[v])
        rec("timing", slot=v, median_ms=m, min_ms=min(times[v]), over_plain=m / base - 1, n=len(times[v]),
            all_ms=times[v])

    count = {}
    for r in range(n_long):
        order = ["base_x", "rot37_x"] + (["lh1_x", "wpre0_x"] if r < n_short else [])
        rng.shuffle(order)
        for name in order:
            m, buf = metas[name], bufs[name]
            use(name)
            buf.zero_()
            ms = timed(arm, fl)
            nw, slab, hdr = m["num_warps"], m["warp_slab_bytes"], m["header_bytes"]
            v = buf[:nw * slab].view(nw, slab)
            words = v[:, hdr:].contiguous().view(torch.int32)
            used = hdr + 4 * (int((words != 0).sum(1).max().item()) + 2)  # + the value word past the last record
            i = count.get(name, 0)
            count[name] = i + 1
            np.savez(os.path.join(out, f"xt_{name}_{i}.npz"), slab=v[:, :used].cpu().numpy(), nw=nw,
                     slab_bytes=slab, total=m["total_bytes"])
            rec("trace", slot=name, i=i, round=r, ms=ms, used_bytes_per_warp=used)
    use("plain")
    with open(os.path.join(out, "xt_mb_meta.json"), "w") as f:
        json.dump(metas, f, default=str)
    print("out_dir", out)


main()
