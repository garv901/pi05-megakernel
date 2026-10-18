"""Profiling driver of the CuTe pi0.5 megakernel (CPU-importable; the run needs a GPU).

  python -m megabench.sota.pi05.cute.profile [--prefix 968 --att-keys 64 --seed 7301 --n-steps 10 --warmups 2 --launches 1
        --mode {plain,iket,ncu} --save-actions PATH --out DIR]
  python -m megabench.sota.pi05.cute.profile --compare A.pt B.pt        (exit 0 iff the saved action tensors are byte-equal)

Builds the CuTe arm with the factories of perf.py / parity.py (make_inputs, cute_arm), sets arm.n_steps, binds (JIT compile), runs
the warmups, synchronizes, then the measured launches.
  plain / iket: every launch timed with CUDA events; prints one `PROFILE {json}` line. `iket` means the process runs under
                `run-iket` (CUTE_DSL_COMPILER_OPT=iket, the kernel then carries the range markers of iket_marks.py); the driver
                code is the same, the mode is only recorded.
  ncu:          the measured launches are wrapped in cudaProfilerStart / Stop (`ncu --profile-from-start off`).
The kernel's mangled entry name is printed (`ENTRY name`) so that the ncu -k filter can be checked."""

import argparse
import hashlib
import json
import os
import sys
import time

from .config import Knobs
from .parity import make_inputs, cute_arm

ENV_KEYS = ("CUTE_DSL_COMPILER_OPT", "PI05_IKET_CTA_STRIDE", "PI05_IKET_SYNC", "CUTE_DSL_LINEINFO", "CUTE_DSL_KEEP",
            "CUTE_DSL_DUMP_DIR")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prefix", type=int, default=968)
    ap.add_argument("--att-keys", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7301)
    ap.add_argument("--fuse", type=int, default=18, help="Knobs.fuse (50 = FUSE_ABL, the fused A + B phase)")
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--launches", type=int, default=1)
    ap.add_argument("--mode", choices=("plain", "iket", "ncu"), default="plain")
    ap.add_argument("--save-actions", help="torch.save the CPU copy of the actions tensor after the last launch")
    ap.add_argument("--out", help="directory for profile.json (the PROFILE line)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two saved action tensors and exit")
    args = ap.parse_args(argv)
    if args.compare is None:
        if not 1 <= args.n_steps <= 10:
            ap.error("--n-steps must be in 1..10")
        if args.warmups < 0 or args.launches < 1:
            ap.error("--warmups >= 0 and --launches >= 1 required")
    return args


def compare(a_path, b_path) -> int:
    import torch
    a, b = torch.load(a_path), torch.load(b_path)
    same = a.shape == b.shape and a.dtype == b.dtype and a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes()
    diff = float((a.double() - b.double()).abs().max()) if a.shape == b.shape else float("nan")
    print(f"COMPARE {'PASS' if same else 'FAIL'} {a_path} {b_path} max_abs_diff={diff}")
    return 0 if same else 1


def entry_name(arm):
    """Mangled name of the kernel entry of the compiled object (the PTX .entry), or None."""
    try:
        return next(iter(arm.fn.kernel_info))
    except Exception:
        return None


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    import torch
    knobs = Knobs(prefix=args.prefix, att_keys=args.att_keys, fuse=args.fuse)
    inputs = make_inputs(knobs, args.seed)
    arm = cute_arm(knobs)
    arm.n_steps = args.n_steps
    t0 = time.time()
    arm.bind(inputs)  # JIT compile (under run-iket: with the markers)
    compile_s = time.time() - t0
    print("ENTRY", entry_name(arm), flush=True)
    arm.ws.zero_()
    arm.actions.zero_()
    for _ in range(args.warmups):
        arm.launch()
    torch.cuda.synchronize()
    ms = []
    if args.mode == "ncu":
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.launches):
            arm.launch()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    else:
        for _ in range(args.launches):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            arm.launch()
            b.record()
            torch.cuda.synchronize()
            ms.append(a.elapsed_time(b))
    torch.cuda.synchronize()
    actions = arm.actions.detach().cpu().contiguous()
    if args.save_actions:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_actions)), exist_ok=True)
        torch.save(actions, args.save_actions)
    rec = {"mode": args.mode, "prefix": args.prefix, "att_keys": args.att_keys, "fuse": args.fuse, "seed": args.seed, "n_steps": args.n_steps,
           "warmups": args.warmups, "launches": args.launches, "env": {k: os.environ.get(k) for k in ENV_KEYS},
           "compile_s": round(compile_s, 1), "launch_ms": ms, "entry": entry_name(arm),
           "actions_sha256": hashlib.sha256(actions.numpy().tobytes()).hexdigest()}
    print("PROFILE " + json.dumps(rec), flush=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "profile.json"), "w") as f:
            json.dump(rec, f, indent=1)
    arm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
