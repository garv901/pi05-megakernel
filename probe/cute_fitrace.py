"""CuTe side of the FlashInfer-profiler port (cute/fit_marks.py): one process per PI05_FIT mode (the mode is read at import).

In this process: the plain CUDA k64 arm (reference, same process) and the CuTe k64 arm built with the current PI05_FIT.
(1) torch.equal of the CuTe actions vs the CUDA actions; (2) N L2-flushed launches of each, interleaved CUDA / CuTe in a
fresh random order per round, recorded as the CuTe / CUDA median ratio (normalizes cross-process drift between modes);
(3) PI05_FIT 1/2: one launch with the zeroed `fit` region, decoded to fit_cute<mode>.npz (same format as probe/fitrace.py).
Env: PI05_FIT, PROBE_PREFIX (sbatch: 968), PROBE_OUT."""
from __future__ import annotations

import os
import random
import statistics

import numpy as np
import torch

from .common import Recorder, build_mk_arm, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch

ROUNDS, PER_ROUND, SEED = 10, 3, 20261019


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def main():
    from ..cute import fit_marks as FM
    from ..cute.arm import CuteArm
    from ..cute.config import workspace_layout

    out = out_dir_from_env("cute-fitrace")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    cuda = build_mk_arm(geo, inputs, "k64")
    cute = CuteArm(geo, "k64", "cuda")
    cute.bind(inputs)
    fl = l2_flush()
    cuda.launch(); cute.launch(); torch.cuda.synchronize()
    rec("equal", mode=FM.MODE, equal=bool(torch.equal(cute.outputs()["actions"], cuda.outputs()["actions"])))

    arms = {"cuda": cuda, "cute": cute}
    times = {v: [] for v in arms}
    rng = random.Random(SEED)
    for _ in range(3):
        cuda.launch(); cute.launch()
    torch.cuda.synchronize()
    for r in range(ROUNDS):
        order = list(arms)
        rng.shuffle(order)
        for v in order:
            times[v] += [timed(arms[v], fl) for _ in range(PER_ROUND)]
    mc, mu = statistics.median(times["cute"]), statistics.median(times["cuda"])
    rec("timing", mode=FM.MODE, cute_ms=mc, cuda_ms=mu, ratio=mc / mu, n=len(times["cute"]))

    if FM.MODE in (1, 2):
        off, size = workspace_layout(cute.knobs)["fit"]
        region = cute.ws[off:off + size].view(torch.int64)
        region.zero_()
        timed(cute, fl)
        nwg = FM.NBLOCKS * FM.NWARPS_
        body = region[1:1 + FM.CAP * nwg].view(FM.CAP, nwg)
        k, w = (body != 0).nonzero(as_tuple=True)
        raw = body[k, w]
        np.savez_compressed(os.path.join(out, f"fit_cute{FM.MODE}.npz"),
                            warp=w.cpu().numpy().astype(np.int32), k=k.cpu().numpy().astype(np.int32),
                            tag=(raw & 0xFFFFFFFF).cpu().numpy().astype(np.uint32),
                            t=((raw >> 32) & 0xFFFFFFFF).cpu().numpy().astype(np.uint32),
                            hdr=int(region[0].item()), last_k=int(k.max().item()) if k.numel() else -1,
                            tstamp=np.zeros(4096, np.int64))
        rec("trace", mode=FM.MODE, entries=int(k.numel()), last_k=int(k.max().item()) if k.numel() else -1,
            hdr=hex(int(region[0].item())))
    print("out_dir", out)


main()
