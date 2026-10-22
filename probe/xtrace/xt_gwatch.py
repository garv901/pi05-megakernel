"""Xtrace step 2: trace a megakernel through gwatch's own runtime (CUPTI interception + launch swap), which is the only
way into the CuTe k64 arm (cooperative cuLaunchKernelEx from libcute_dsl_runtime, a JIT cubin; no LD_PRELOAD seam).

XT_IMPL=cute (default) or cuda (cross-check of the cooperative-launch swap on the arm step 1 runs through the shim).
In this process: gwatch.cuda.start() before anything loads, the plain CUDA k64 arm as reference, the XT_IMPL arm.
Each step is recorded with its outcome so a failing stage still leaves evidence:
(1) capture the launch, find the kernel's cubin, splice it in-process with splice.splice (instruction count checked
against the offline splice), load it and resolve the origin / trace kernel definitions;
(2) torch.equal of plain and traced actions vs the CUDA reference;
(3) L2-flushed timing: cuda plain / impl plain / impl traced, random order per round;
(4) raw trace buffers (zeroed, one launch each) to xt_<impl>_<clock>_<i>.npz.
Run in the gwatch venv: LD_LIBRARY_PATH=<cu12 cudart + cupti> /raid/garv901/xtrace-venv/bin/python -m ...xt_gwatch
Env: XT_IMPL, XT_CLOCKS (gpu,anchor), XT_KEEP (splice.py keep tag; default every site), PROBE_PREFIX (sbatch: 968), PROBE_OUT."""
from __future__ import annotations

import json
import os
import random
import statistics
import traceback

import numpy as np
import torch

import gwatch.cuda
import gwatch.cuda.binary as gb
from gwatch.cuda.profile import ProfileContext

from . import splice as SP
from .bank4 import upload
from ..common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch

ROUNDS, PER_ROUND, SEED = 10, 3, 20261019
OFFLINE_N_INSTR = {"cuda": 19448, "cute": 14408}
TRACES = {"gpu": 3, "anchor": 2}


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    impl = os.environ.get("XT_IMPL", "cute")
    clocks = os.environ.get("XT_CLOCKS", "gpu,anchor").split(",")
    out = out_dir_from_env(f"xtrace-gwatch-{impl}")
    rec = Recorder(out, f"records_{impl}.jsonl")

    def step(name, fn, **extra):
        try:
            r = fn()
            rec("step", step=name, ok=True, **extra)
            return r
        except Exception as e:
            rec("step", step=name, ok=False, err=repr(e), tb=traceback.format_exc()[-2000:], **extra)
            raise

    setup_torch()
    rec("start", impl=impl, started=bool(gwatch.cuda.start()))
    ctx = ProfileContext()
    geo, inputs = make_geo_inputs()
    ref_arm = build_mk_arm(geo, inputs, "k64")
    if impl == "cute":
        from ...cute.arm import CuteArm
        arm = CuteArm(geo, "k64", "cuda")
        arm.bind(inputs)
    else:
        arm = build_mk_arm(geo, inputs, "k64")
    ref_arm.launch(); torch.cuda.synchronize()
    ref = ref_arm.outputs()["actions"].clone()

    # (1) capture
    ctx.start_tracing_kernel_launch()
    arm.launch()
    ctx.sync_stream()
    kernels = ctx.stop_tracing_kernel_launch()
    names = [k.definition.mangled_prototype for k in kernels]
    match = sorted([k for k in kernels if "pi05_megakernel" in k.definition.mangled_prototype],
                   key=lambda k: len(k.definition.mangled_prototype))
    rec("capture", n_kernels=len(kernels), names=[n[:120] for n in names], matched=len(match))
    if not match:
        return
    kernel = match[0]
    kname = kernel.definition.mangled_prototype
    arch = kernel.definition.params.arch_version
    dims = ((kernel.grid_dim_x, kernel.grid_dim_y, kernel.grid_dim_z),
            (kernel.block_dim_x, kernel.block_dim_y, kernel.block_dim_z))
    cubins = [c for c in ctx.get_list_binary_cubin() if kname in c.get_map_kernel_def()]
    cubins = [c for c in cubins if c.params.arch_version == arch] or cubins
    rec("cubin", kernel=kname[:160], arch=str(arch), grid=dims[0], block=dims[1], n_cubins=len(cubins))
    assert dims == (SP.GRID, SP.BLOCK), dims

    origin = step("origin_kerneldef", lambda: ctx.get_kernel_def_sass_by_name(kname))
    rec("origin", found=origin is not None, regs=getattr(getattr(origin, "params", None), "num_registers", None))
    traced, bufs, metas, extras = {}, {}, {}, {}
    for clock in clocks:
        def build():
            cub = cubins[0].clone()
            _, image, schema, _, meta = SP.splice(cub, clock, keep=SP.parse_keep(os.environ.get("XT_KEEP", "")))
            path = os.path.join(out, f"xtrace_{impl}_{clock}.cubin")
            with open(path, "wb") as f:
                f.write(image)
            loaded = gb.Cubin()
            loaded.fill(path)
            loaded.parse()
            ctx.load_binary_cubin(loaded)
            tks = sorted(ctx.load_kerneldef_sass(loaded, [kname]), key=lambda k: len(k.mangled_prototype))
            return meta, tks[0]
        meta, tk = step(f"splice_load_{clock}", build)
        meta["n_instr_offline"] = OFFLINE_N_INSTR[impl]
        rec("splice", clock=clock, n_instr=meta["n_instr"], n_instr_match=meta["n_instr"] == OFFLINE_N_INSTR[impl],
            sites=len(meta["sites"]), total_bytes=meta["total_bytes"], trace_regs=tk.params.num_registers)
        metas[clock], traced[clock] = meta, tk
        bufs[clock] = torch.zeros(meta["total_bytes"], dtype=torch.uint8, device="cuda")
        extras[clock] = upload(meta)  # (table tensor kept alive, its slot addresses)
    with open(os.path.join(out, f"xt_meta_{impl}.json"), "w") as f:
        json.dump({f"xt_{c}": m for c, m in metas.items()}, f, default=str)

    def use(v):
        ctx.unregister_trace_kernel_sass(origin) if use.cur else None
        use.cur = None
        if v.startswith("xt_"):
            c = v[3:]
            ctx.register_trace_kernel_sass(origin, traced[c], param_values=extras[c][1] + [bufs[c].data_ptr()])
            use.cur = v
    use.cur = None

    # (2) bitwise
    for v in ["plain"] + [f"xt_{c}" for c in traced]:
        step(f"use_{v}", lambda: use(v))
        arm.outputs()["actions"].zero_()
        try:
            arm.launch(); torch.cuda.synchronize()
            rec("equal", slot=v, equal=bool(torch.equal(arm.outputs()["actions"], ref)),
                written_bytes=int(bufs[v[3:]].count_nonzero().item()) if v != "plain" else None)
        except Exception as e:
            rec("equal", slot=v, equal=None, err=repr(e))
            raise
    use("plain")

    # (3) timing; ref_arm is not swapped (only `origin` is registered), so it normalizes cross-process drift
    fl = l2_flush()
    variants = ["ref", "plain"] + [f"xt_{c}" for c in traced]
    arms = {v: (ref_arm if v == "ref" else arm) for v in variants}
    times = {v: [] for v in variants}
    rng = random.Random(SEED)
    for v in variants:
        use(v)
        for _ in range(3):
            arms[v].launch()
    torch.cuda.synchronize()
    for _ in range(ROUNDS):
        order = list(variants)
        rng.shuffle(order)
        for v in order:
            use(v)
            times[v] += [timed(arms[v], fl) for _ in range(PER_ROUND)]
    use("plain")
    base, refm = statistics.median(times["plain"]), statistics.median(times["ref"])
    for v in variants:
        m = statistics.median(times[v])
        rec("timing", impl=impl, slot=v, median_ms=m, min_ms=min(times[v]), over_plain=m / base - 1,
            over_ref=m / refm, n=len(times[v]), all_ms=times[v])

    # (4) traces
    for c in traced:
        use(f"xt_{c}")
        for i in range(TRACES.get(c, 1)):
            bufs[c].zero_()
            ms = timed(arm, fl)
            raw = bufs[c].cpu().numpy()
            np.savez_compressed(os.path.join(out, f"xt_{impl}_{c}_{i}.npz"), raw=raw)
            rec("trace", impl=impl, clock=c, i=i, ms=ms, nonzero_bytes=int(np.count_nonzero(raw)))
    use("plain")
    print("out_dir", out)


main()
