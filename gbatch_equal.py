"""Bitwise check of the stage G load batching: actions of every variant must equal CUDA k64 (torch.equal), per seed.

  python -m megabench.sota.pi05.gbatch_equal [--arms megakernel,cute] [--seeds 7301,8917,42811] [--prefix 968]

Variants: CUDA k64, k64-gb1, k64-gb2, k64-gn, k64-gb1-gn, k64-gb2-gn and (``--arms`` includes cute) the CuTe k64, k64-gb1, k64-gb2, all at the same inputs per
seed. Prints EQUAL / DIFF max_abs=... per (arm/variant, seed); exit 1 on any DIFF or error. Needs a GPU (run through slurm)."""
from __future__ import annotations

import argparse
import sys

import torch

from . import fixture as FX
from .megakernel import MegakernelArm

CUTE_VARIANTS = ("k64", "k64-gb1", "k64-gb2")
VARIANTS = (*CUTE_VARIANTS, "k64-gn", "k64-gb1-gn", "k64-gb2-gn")  # CUDA; gn is CUDA only


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", default="megakernel,cute")
    ap.add_argument("--seeds", default="7301,8917,42811")
    ap.add_argument("--prefix", type=int, default=968)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    arms = a.arms.split(",")
    geo = FX.Geometry(prefix=a.prefix)
    # The reference is always CUDA k64, even with --arms cute.
    entries = {"megakernel/k64": MegakernelArm(geo, "k64", a.device)}
    if "megakernel" in arms:
        for v in VARIANTS[1:]:
            entries[f"megakernel/{v}"] = MegakernelArm(geo, v, a.device)
    if "cute" in arms:
        from .cute.arm import CuteArm
        for v in CUTE_VARIANTS:
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
