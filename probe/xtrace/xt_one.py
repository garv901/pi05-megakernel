"""One launch of one spliced CUDA k64 cubin through coop_shim.so (bisecting a faulting probe; an illegal address kills the
context, so one process per cubin). Prints a JSON line: equal vs the plain launch, error, records written.
Run: LD_PRELOAD=runs/xtrace-cpu/coop_shim.so python -m megabench.sota.pi05.probe.xtrace.xt_one X.cubin
Env: PROBE_PREFIX (sbatch: 968); XT_WATCH=<s>: if the launch has not finished after that long, copy the trace buffer out
on a side stream (copy engines still run beside a deadlocked kernel) and print where every warp stopped (XT_STUCK)."""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time

import torch

from .bank4 import upload
from ..common import build_mk_arm, make_geo_inputs, setup_torch


def stuck(meta, buf, host):
    """Where the warps of a hung launch stopped: groups of warps by (records written, site id of the last two words).
    A record is a 32-bit word whose low bits hold site id + 1; each probe also stores a value word one slot past its
    record, which the next record overwrites, so the last non-zero word can be that value rather than a record."""
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        host.copy_(buf, non_blocking=True)
    side.synchronize()
    slab, hdr, nw = meta["warp_slab_bytes"], meta["header_bytes"], meta["num_warps"]
    mask = (1 << meta["site_id_bits"]) - 1
    rec = host[:slab * nw].view(nw, slab)[:, hdr:].contiguous().view(torch.int32).view(nw, -1)
    n = (rec != 0).sum(1).tolist()
    name = {i: f"b{t['barrier']}{t['kind'][0]}" for i, t in enumerate(meta["sites"])}
    groups = {}
    for w in range(nw):
        tail = tuple(name.get((int(rec[w, k].item()) & mask) - 1, "?") for k in range(max(0, n[w] - 2), n[w]))
        groups.setdefault((n[w], tail), []).append(w)
    return sorted([[k[0], list(k[1]), len(v), v[:6]] for k, v in groups.items()], key=lambda g: (g[0], g[1]))


def main():
    path = sys.argv[1]
    meta = json.load(open(path[:-len(".cubin")] + ".json"))
    setup_torch()
    shim = ctypes.CDLL(None)
    shim.xt_set.argtypes = [ctypes.c_int, ctypes.c_ulonglong]
    shim.xt_set_extra.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ulonglong)]
    geo, inputs = make_geo_inputs()
    arm = build_mk_arm(geo, inputs, "k64")
    arm.launch(); torch.cuda.synchronize()
    ref = arm.outputs()["actions"].clone()
    slot = shim.xt_load(path.encode(), meta["kernel"].encode())
    buf = torch.zeros(meta["total_bytes"], dtype=torch.uint8, device="cuda")
    table, extra = upload(meta)
    assert shim.xt_set_extra(slot, len(extra), (ctypes.c_ulonglong * max(1, len(extra)))(*extra)) == 0
    res = {"cubin": path.split("/")[-1], "slot": slot}
    host = torch.empty(buf.numel(), dtype=torch.uint8, pin_memory=True)
    shim.xt_set(slot, buf.data_ptr())
    arm.actions.zero_()
    try:
        arm.launch()
        watch = float(os.environ.get("XT_WATCH", "0"))
        if watch:
            done = torch.cuda.Event()
            done.record()
            t0 = time.time()
            while not done.query() and time.time() - t0 < watch:
                time.sleep(0.5)
            if not done.query():
                print("XT_STUCK " + json.dumps(stuck(meta, buf, host)), flush=True)
        torch.cuda.synchronize()
        res.update(ok=True, equal=bool(torch.equal(arm.outputs()["actions"], ref)),
                   nonzero_bytes=int(buf.count_nonzero().item()))
    except Exception as e:
        res.update(ok=False, err=repr(e)[:200])
    res["sites"] = [f"b{s['barrier']}{s['kind'][0]}" for s in meta["sites"]]
    print("XT_ONE " + json.dumps(res), flush=True)


main()
