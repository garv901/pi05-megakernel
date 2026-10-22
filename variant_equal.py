"""Bitwise check of the gemm_phase load hoist (and any other variant suffix): actions of every entry must equal CUDA k64
(torch.equal), per seed.

  python -m megabench.sota.pi05.variant_equal [--arms megakernel] [--variants k64,k64-nolh] [--seeds 7301,8917,42811] [--prefix 968]

Entries: every --variants entry on every --arms arm (CUDA arm = MegakernelArm, cute arm = CuteArm), all at the same inputs per
seed; the reference is always CUDA k64 (built even if it is not in the list). Prints EQUAL / DIFF max_abs=... per (arm/variant,
seed), then `RESULT: PASS|FAIL`; exit 1 on any DIFF or error. Needs a GPU (run through slurm)."""
from __future__ import annotations

import argparse
import sys

import torch

from . import fixture as FX
from .megakernel import MegakernelArm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="megakernel")
    ap.add_argument("--variants", default="k64,k64-nolh")
    ap.add_argument("--seeds", default="7301,8917,42811")
    ap.add_argument("--prefix", type=int, default=968)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    arms, variants = a.arms.split(","), a.variants.split(",")
    geo = FX.Geometry(prefix=a.prefix)
    entries = {"megakernel/k64": MegakernelArm(geo, "k64", a.device)}  # reference first
    if "megakernel" in arms:
        for v in variants:
            entries.setdefault(f"megakernel/{v}", MegakernelArm(geo, v, a.device))
    if "cute" in arms:
        from .cute.arm import CuteArm
        for v in variants:
            entries[f"cute/{v}"] = CuteArm(geo, v, a.device)
    bad = 0
    for seed in (int(s) for s in a.seeds.split(",")):
        inputs = FX.make_inputs(geo, seed, a.device)
        want = None
        for label, arm in entries.items():
            arm.bind(inputs)
            arm.launch()
            torch.cuda.synchronize()
            got = arm.outputs()["actions"].clone()
            if want is None:  # first entry is CUDA k64
                want = got
            if torch.equal(want, got):
                print(f"{label} seed={seed}: EQUAL", flush=True)
            else:
                bad += 1
                print(f"{label} seed={seed}: DIFF max_abs={float((want - got).abs().max()):.6g}", flush=True)
    print("RESULT:", "FAIL" if bad else "PASS", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
