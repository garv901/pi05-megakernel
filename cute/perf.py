"""Stage ladder timing (phase 6): CUDA vs CuTe launch time of stops at the cumulative ladder pro, A, B, C3, D, E, F, G, H, I
(step 0 layer 0), I at layer 1 (one full layer more) and the full chunk, so that the per-stage deltas show where the CuTe port is
slower. Same inputs, CUDA and CuTe launches interleaved (CUDA, CuTe, CUDA, ...), median of --iters launches after 3 warmups, CUDA
events, L2 not flushed.

  python -m megabench.sota.pi05.cute.perf --prefix 968 --att-keys 64 [--seed S --iters N --stages pro,A,... --fuse F --wpre W
                                           --d-wpt D --host-zero --out DIR]
Prints one `LADDER {json}` line per stop and a table of the per-stage deltas (entry minus the previous entry of the ladder; the
first entry is its own time) for CUDA, CuTe and CuTe - CUDA. Builds the CUDA extension per stop (cached by torch) and compiles the
CuTe kernel per stop (about 20 s each)."""

import argparse
import json
import os
import statistics
import time
import sys

from .config import FUSE_ABL, Knobs
from .parity import DEFAULT_D_WPT, DEFAULT_WPRE, STAGES, cuda_arm, cute_arm, make_inputs

WARMUPS = 3
# entry label -> (stage, layer): the ladder; "I@1" is stop I of step 0 layer 1, "full" the whole chunk (no stop)
LADDER = tuple((s, s, 0) for s in STAGES) + (("I@1", "I", 1), ("full", None, 0))
LABELS = tuple(e[0] for e in LADDER)


def ladder(stages=None):
    """List of (label, stage, layer) for the requested labels (default: the whole ladder, in ladder order)."""
    if stages is None:
        return list(LADDER)
    want = list(stages.split(",")) if isinstance(stages, str) else list(stages)
    bad = [w for w in want if w not in LABELS]
    if bad:
        raise ValueError(f"unknown stages {bad}, have {LABELS}")
    return [e for e in LADDER if e[0] in want]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", type=int, default=968)
    ap.add_argument("--att-keys", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--stages", default=",".join(LABELS), help=f"comma list of {LABELS}")
    ap.add_argument("--fuse", type=int, default=18)
    ap.add_argument("--wpre", type=int, default=DEFAULT_WPRE)
    ap.add_argument("--d-wpt", type=int, default=DEFAULT_D_WPT)
    ap.add_argument("--host-zero", action="store_true", help="CuTe arm zeroes cnt..bar from the host before each launch (phase 5 path)")
    ap.add_argument("--out", help="directory for ladder_p<P>_k<K>.json")
    args = ap.parse_args(argv)
    try:
        args.ladder = ladder(args.stages)
    except ValueError as e:
        ap.error(str(e))
    if args.fuse & FUSE_ABL:  # no stop A with FUSE_ABL (A and B are one phase); the CUDA arm runs fuse & 31
        args.ladder = [e for e in args.ladder if e[1] != "A"]
    if args.iters < 1:
        ap.error("--iters must be positive")
    args.knobs = Knobs(prefix=args.prefix, att_keys=args.att_keys, fuse=args.fuse, wpre=args.wpre, d_wpt=args.d_wpt,
                       host_zero=args.host_zero)
    return args


def deltas(rows):
    """rows: [{'label', 'cuda_ms', 'cute_ms'}, ...] in ladder order -> adds cuda_delta_us, cute_delta_us, diff_us (CuTe - CUDA)."""
    prev_a = prev_b = 0.0
    for r in rows:
        r["cuda_delta_us"] = (r["cuda_ms"] - prev_a) * 1e3
        r["cute_delta_us"] = (r["cute_ms"] - prev_b) * 1e3
        r["diff_us"] = r["cute_delta_us"] - r["cuda_delta_us"]
        prev_a, prev_b = r["cuda_ms"], r["cute_ms"]
    return rows


def table(rows) -> str:
    lines = [f"{'stop':>6} {'CUDA ms':>9} {'CuTe ms':>9} | {'d CUDA us':>10} {'d CuTe us':>10} {'CuTe-CUDA':>10}"]
    for r in rows:
        lines.append(f"{r['label']:>6} {r['cuda_ms']:9.4f} {r['cute_ms']:9.4f} | {r['cuda_delta_us']:10.1f} {r['cute_delta_us']:10.1f} "
                     f"{r['diff_us']:10.1f}")
    return "\n".join(lines)


def time_entry(knobs: Knobs, inputs, stage, layer, iters: int) -> dict:
    """Median launch time (ms) of the CUDA and the CuTe arm stopped after `stage` at step 0 `layer` (stage None: the whole chunk)."""
    import torch
    stop = None if stage is None else (stage, 0, layer)
    arms = {"cuda": cuda_arm(knobs, stop), "cute": cute_arm(knobs, stop)}
    for arm in arms.values():
        arm.bind(inputs)
        arm.ws.zero_()
        arm.actions.zero_()
    for _ in range(WARMUPS):
        for arm in arms.values():
            arm.launch()
    torch.cuda.synchronize()
    ms = {"cuda": [], "cute": []}
    for _ in range(iters):
        for name, arm in arms.items():  # interleaved: CUDA, CuTe, CUDA, ...
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            arm.launch()
            b.record()
            torch.cuda.synchronize()
            ms[name].append(a.elapsed_time(b))
    for arm in arms.values():
        arm.close()
    return {"cuda_ms": statistics.median(ms["cuda"]), "cute_ms": statistics.median(ms["cute"]),
            "cuda_min_ms": min(ms["cuda"]), "cute_min_ms": min(ms["cute"])}


def main(argv=None) -> int:
    args = parse_args(argv)
    inputs = make_inputs(args.knobs, args.seed)
    rows = []
    for label, stage, layer in args.ladder:
        t0 = time.time()
        row = {"label": label, "stage": stage, "layer": layer, **time_entry(args.knobs, inputs, stage, layer, args.iters)}
        rows.append(row)
        print("LADDER " + json.dumps({"prefix": args.prefix, "att_keys": args.att_keys, "fuse": args.fuse, "wpre": args.wpre,
                                      "d_wpt": args.d_wpt, "host_zero": args.host_zero, "seed": args.seed, "iters": args.iters,
                                      "elapsed_s": round(time.time() - t0, 1), **row}), flush=True)
    deltas(rows)
    print(table(rows))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"ladder_p{args.prefix}_k{args.att_keys}.json"), "w") as f:
            json.dump({"prefix": args.prefix, "att_keys": args.att_keys, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
