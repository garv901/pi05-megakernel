"""Parity harness of the CuTe port: run the CUDA megakernel (optionally stopped after a stage) and compare workspace buffers.

  run_cuda(knobs, stop, seed) -> {buffer name: uint8 tensor}    stop = None | (stage, step, layer), stage in STAGES
  run_cute(...)                                                 stops pro, A, B (phase 2)
  compare(a, b, names) -> {name: report}                        byte compare; mismatches get count, first index, max ulp
CLI (GPU):  python -m megabench.sota.pi05.cute.parity --selfcheck --prefix 968 --att-keys 64 [--stages pro,A,B,C3,I,full]
  python -m megabench.sota.pi05.cute.parity --compare --stages pro [--mask-tail N]   (CUDA vs CuTe)
CLI (CPU):  python -m megabench.sota.pi05.cute.parity --layout-check [--prefix P --att-keys K]"""

import argparse
import dataclasses
import json
import os
import sys

from .config import FUSE_ABL, LAYOUT_DTYPE, LAYOUT_ORDER, NSTEP, L, Knobs, workspace_layout

STAGES = ("pro", "A", "B", "C3", "D", "E", "F", "G", "H", "I")
DEFAULT_STAGES = ("pro", "A", "B", "C3", "I", "full")
BUFFERS = tuple(LAYOUT_ORDER) + ("actions",)
DEFAULT_WPRE, DEFAULT_D_WPT = 15, 2
# buffers a stop at the stage must have written (non-vacuity of the byte compare); later stages extend B's set
EXPECTED = {"pro": ("temb", "tmid", "cond", "mod", "rope", "h", "ssp", "xbuf")}
EXPECTED["A"] = EXPECTED["pro"] + ("part",)
EXPECTED["B"] = EXPECTED["A"] + ("q", "ksuf", "vsuf")
EXPECTED["C3"] = EXPECTED["B"] + ("stats", "opart", "attn")  # + "cnt" when fuse bit 2 (CC3) is set, see expected()


# D..I rewrite part / h / ssp / act of their own stage, which the earlier layers (or the stage's own input) already made
# non-zero; the expected set is what the stage itself writes (non-zero in the CUDA dump at the stop)
EXPECTED["D"] = ("part",)
EXPECTED["E"] = ("h", "ssp")
EXPECTED["F"] = ("part",)
EXPECTED["G"] = ("act",)
EXPECTED["H"] = ("part",)
EXPECTED["I"] = ("h", "ssp")
EXPECTED["full"] = ("actions",)


def expected(stage: str, knobs: Knobs) -> tuple:
    names = EXPECTED.get(stage, ())
    return names + ("cnt",) if stage == "C3" and knobs.fuse & 2 else names


def cuda_fuse(knobs: Knobs) -> int:
    """Fuse word of the CUDA kernel: it has bits 0..4 only; FUSE_ABL (32) is CuTe-only (--fuse 50 -> CuTe 50, CUDA 18)."""
    return knobs.fuse & (FUSE_ABL - 1)


def is_abl(knobs: Knobs) -> bool:
    return bool(knobs.fuse & FUSE_ABL)


def drop_stage_a(knobs: Knobs, stages):
    """FUSE_ABL has no stop A (A and B are one phase): the stage is skipped (a note goes to stderr)."""
    stages = list(stages)
    if is_abl(knobs) and "A" in stages:
        print("NOTE: stop A skipped with FUSE_ABL (A and B are one phase)", file=sys.stderr)
        stages.remove("A")
    return stages


def stop_flags(stop):
    if stop is None:
        return ()
    stage, step, layer = stop
    return (f"-DPI05_STOP_PH={STAGES.index(stage)}", f"-DPI05_STOP_STEP={int(step)}", f"-DPI05_STOP_LAYER={int(layer)}")


def build_flags(knobs: Knobs, stop=None):
    extra = list(stop_flags(stop))
    if knobs.wpre != DEFAULT_WPRE:
        extra.append(f"-DPI05_WPRE={knobs.wpre}")
    if knobs.d_wpt != DEFAULT_D_WPT:
        extra.append(f"-DPI05_D_WPT={knobs.d_wpt}")
    return tuple(extra)


def load_ext(knobs: Knobs, stop=None):
    from ..megakernel import load
    return load(knobs.prefix, knobs.att_keys, build_flags(knobs, stop))


def make_inputs(knobs: Knobs, seed: int = 0, mask_tail: int = 0) -> dict:
    """Fixture inputs; mask_tail zeroes the last N entries of prefix_mask (nvalid < P, RoPE positions change)."""
    from ..fixture import Geometry, make_inputs as fixture_inputs
    inputs = fixture_inputs(Geometry(prefix=knobs.prefix), seed)
    if mask_tail:
        inputs["prefix_mask"][-mask_tail:] = False
    return inputs


def _dump(arm, knobs: Knobs) -> dict:
    import torch
    lay = workspace_layout(knobs)
    out = {n: arm.ws[lay[n][0]:lay[n][0] + lay[n][1]].clone() for n in LAYOUT_ORDER}
    out["actions"] = arm.actions.contiguous().view(torch.uint8).flatten().clone()
    return out


def cuda_arm(knobs: Knobs, stop=None):
    """MegakernelArm of the CUDA extension (stop = None | (stage, step, layer)); not bound."""
    from ..fixture import Geometry
    from ..megakernel import MegakernelArm
    return MegakernelArm(Geometry(prefix=knobs.prefix), f"k{knobs.att_keys}", "cuda", build_flags(knobs, stop), cuda_fuse(knobs))


def cute_arm(knobs: Knobs, stop=None):
    """CuteArm (stop = None | (stage, step, layer)); not bound."""
    from ..fixture import Geometry
    from .arm import CuteArm
    from .kernel import Stop
    return CuteArm(Geometry(prefix=knobs.prefix), f"k{knobs.att_keys}", "cuda", knobs, Stop(*stop) if stop else Stop())


def run_cuda(knobs: Knobs, stop=None, seed: int = 0, mask_tail: int = 0, inputs=None, n_steps=None, n_layers=None) -> dict:
    import torch
    from ..fixture import Geometry
    from ..megakernel import MegakernelArm
    geo = Geometry(prefix=knobs.prefix)
    arm = MegakernelArm(geo, f"k{knobs.att_keys}", "cuda", build_flags(knobs, stop), cuda_fuse(knobs))
    arm.geo = dataclasses.replace(geo, steps=n_steps or geo.steps, layers=n_layers or geo.layers)  # runtime truncation
    arm.bind(inputs if inputs is not None else make_inputs(knobs, seed, mask_tail))
    arm.ws.zero_()
    arm.actions.zero_()
    arm.launch()
    torch.cuda.synchronize()
    out = _dump(arm, knobs)
    arm.close()
    return out


def cnt_bar_nonzero(arm, knobs: Knobs) -> int:
    """Number of nonzero bytes in the cnt and bar regions of a CuteArm workspace (0 after a launch with the in-kernel reset)."""
    lay = workspace_layout(knobs)
    return int(sum((arm.ws[lay[n][0]:lay[n][0] + lay[n][1]] != 0).sum() for n in ("cnt", "bar")))


def run_cute(knobs: Knobs, stop=None, seed: int = 0, mask_tail: int = 0, inputs=None, n_steps=None, n_layers=None) -> dict:
    """Same return format as run_cuda; stop is the same (stage, step, layer) tuple (None = everything implemented)."""
    import torch
    from ..fixture import Geometry
    from .arm import CuteArm
    from .kernel import Stop
    geo = Geometry(prefix=knobs.prefix)
    arm = CuteArm(geo, f"k{knobs.att_keys}", "cuda", knobs, Stop(*stop) if stop else Stop())
    arm.n_steps, arm.n_layers = n_steps or NSTEP, n_layers or L
    arm.bind(inputs if inputs is not None else make_inputs(knobs, seed, mask_tail))
    arm.ws.zero_()
    arm.actions.zero_()
    arm.launch()
    torch.cuda.synchronize()
    out = _dump(arm, knobs)
    out["_cnt_bar_nonzero"] = cnt_bar_nonzero(arm, knobs)  # not a buffer: the reset check (see compare_stages)
    arm.close()
    return out


def _ordered(u, bits):
    import torch
    m = (1 << (bits - 1)) - 1
    mag = (u & m).to(torch.int64)
    return torch.where((u >> (bits - 1)) & 1 != 0, -mag, mag)


def _elems(t, dt):
    import torch
    if dt == "bf16":
        return t.view(torch.int16).to(torch.int32) & 0xFFFF, 16
    return t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF, 32


def compare(a: dict, b: dict, names=BUFFERS) -> dict:
    """Per buffer: equal flag; on mismatch the byte count, first element index and max ulp (unsigned-integer difference for u32)."""
    import torch
    rep = {}
    for n in names:
        x, y = a[n], b[n]
        if x.numel() != y.numel():
            rep[n] = {"equal": False, "error": f"size {x.numel()} vs {y.numel()}"}
            continue
        if torch.equal(x, y):
            rep[n] = {"equal": True}
            continue
        dt = LAYOUT_DTYPE.get(n, "f32")
        ex, bits = _elems(x.contiguous(), dt)
        ey, _ = _elems(y.contiguous(), dt)
        bad = (ex != ey).nonzero().flatten()
        if dt == "u32":
            ulp = (ex - ey).abs()
        else:
            ulp = (_ordered(ex, bits) - _ordered(ey, bits)).abs()
        rep[n] = {"equal": False, "bytes_differ": int((x != y).sum()), "elements_differ": int(bad.numel()),
                  "first_index": int(bad[0]), "max_ulp": int(ulp.max())}
    return rep


def check_layout(knobs: Knobs) -> list:
    """workspace_layout vs the extension's layout_offsets() / workspace_bytes(); returns the list of differences."""
    ext = load_ext(knobs)
    off = list(ext.layout_offsets())
    lay = workspace_layout(knobs)
    names = [n for n in LAYOUT_ORDER if n != "bar"]
    diffs = []
    if len(off) != len(names):
        diffs.append(f"layout_offsets has {len(off)} entries, expected {len(names)}")
    for n, o in zip(names, off):
        if lay[n][0] != o:
            diffs.append(f"{n}: config {lay[n][0]} ext {o}")
    if lay["total"][1] != ext.workspace_bytes():
        diffs.append(f"total: config {lay['total'][1]} ext {ext.workspace_bytes()}")
    return diffs


def selfcheck(knobs: Knobs, stages, step: int, layer: int, seed: int) -> dict:
    res = {}
    base = None
    for st in drop_stage_a(knobs, stages):
        stop = None if st == "full" else (st, step, layer)
        a = run_cuda(knobs, stop, seed)
        b = run_cuda(knobs, stop, seed)
        rep = compare(a, b)
        bad = {n: r for n, r in rep.items() if not r["equal"]}
        res[st] = {"deterministic": not bad, "mismatches": bad}
        if st == "full":
            base = a
    if base is not None:  # a stop at the very last stage of the run falls through to the natural end
        last = run_cuda(knobs, ("I", NSTEP - 1, L - 1), seed)
        rep = compare(base, last, ("actions",))
        res["stop_at_end_equals_full"] = rep["actions"]["equal"]
    return res


def time_stop(knobs: Knobs, seed: int, mask_tail: int = 0, stage: str = "pro", layer: int = 0) -> dict:
    """Wall time (CUDA events, median of 20 after 3 warmups) of a launch stopped after `stage` (step 0, `layer`), CUDA vs CuTe;
    informative only."""
    import statistics
    import torch
    from ..fixture import Geometry
    from ..megakernel import MegakernelArm
    from .arm import CuteArm
    from .kernel import Stop
    geo = Geometry(prefix=knobs.prefix)
    inputs = make_inputs(knobs, seed, mask_tail)
    full = stage == "full"  # the whole chunk: no stop in either kernel
    arms = {"cuda": MegakernelArm(geo, f"k{knobs.att_keys}", "cuda", build_flags(knobs, None if full else (stage, 0, layer)), cuda_fuse(knobs)),
            "cute": CuteArm(geo, f"k{knobs.att_keys}", "cuda", knobs, Stop() if full else Stop(stage, 0, layer))}
    res = {}
    for name, arm in arms.items():
        arm.bind(inputs)
        arm.ws.zero_()
        for _ in range(3):
            arm.launch()
        ms = []
        for _ in range(20):  # back to back, L2 not flushed (each launch zeroes its own counters)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            arm.launch()
            b.record()
            torch.cuda.synchronize()
            ms.append(a.elapsed_time(b))
        res[name + "_ms_median"] = statistics.median(ms)
        arm.close()
    return res


def time_pro(knobs: Knobs, seed: int, mask_tail: int = 0) -> dict:
    return time_stop(knobs, seed, mask_tail, "pro")


def compare_stages(knobs: Knobs, stages, step: int, layer: int, seed: int, mask_tail: int, n_steps=None, n_layers=None) -> dict:
    """CUDA vs CuTe at each stage on the same inputs; every workspace buffer except `bar` (epoch numbering differs) and, with the
    in-kernel reset (not knobs.host_zero), `cnt` (zeroed on exit; use --host-zero to compare it). With the reset the CuTe cnt / bar
    words must all be zero after the launch (`counters_zero`, part of `equal`)."""
    inputs = make_inputs(knobs, seed, mask_tail)
    names = tuple(n for n in BUFFERS if n != "bar" and (knobs.host_zero or n != "cnt"))
    res = {}
    for st in drop_stage_a(knobs, stages):
        stop = None if st == "full" else (st, step, layer)
        ref = run_cuda(knobs, stop, seed, mask_tail, inputs, n_steps, n_layers)
        got = run_cute(knobs, stop, seed, mask_tail, inputs, n_steps, n_layers)
        cmp_names = names
        if is_abl(knobs) and st in ("B", "C3"):  # FUSE_ABL never writes the A partials; D / F / H overwrite `part` and still compare
            cmp_names = tuple(n for n in names if n != "part")
        rep = compare(ref, got, cmp_names)
        nz = got["_cnt_bar_nonzero"]
        counters_zero = knobs.host_zero or nz == 0  # host_zero leaves the counters of the stopped run in place
        frac = {n: float((ref[n] != 0).float().mean()) for n in cmp_names}  # fraction of nonzero bytes of the CUDA dump
        vacuous = [n for n in expected(st, knobs) if n in frac and frac[n] == 0.0]
        res[st] = {"equal": all(r["equal"] for r in rep.values()) and counters_zero, "counters_zero": counters_zero,
                   "cnt_bar_nonzero_bytes": nz, "buffers": {n: r for n, r in rep.items() if not r["equal"]},
                   "nonzero_frac": frac, "all_zero_but_expected_written": vacuous}
        if vacuous:
            print(f"WARNING stage {st}: buffers expected to be written are all zero in the CUDA dump: {vacuous}", file=sys.stderr)
    return res


def check_reset(knobs: Knobs, seed: int, mask_tail: int = 0, launches: int = 2) -> dict:
    """Back-to-back CuTe `full` launches on one bound arm without any host zeroing in between (in-kernel reset): the counters must
    be zero after each launch and every launch must reproduce the first one (all buffers except cnt / bar, and the actions)."""
    import torch
    inputs = make_inputs(knobs, seed, mask_tail)
    arm = cute_arm(knobs)
    arm.bind(inputs)  # zeroes cnt..bar once
    names = tuple(n for n in BUFFERS if n not in ("cnt", "bar"))
    first, res = None, {"launches": []}
    for i in range(launches):
        arm.launch()
        torch.cuda.synchronize()
        out = _dump(arm, knobs)
        nz = cnt_bar_nonzero(arm, knobs)
        rep = {"counters_zero": nz == 0, "cnt_bar_nonzero_bytes": nz}
        if first is None:
            first = out
        else:
            bad = {n: r for n, r in compare(first, out, names).items() if not r["equal"]}
            rep["equals_first"] = not bad
            rep["mismatches"] = bad
        res["launches"].append(rep)
    arm.close()
    res["ok"] = all(r["counters_zero"] and r.get("equals_first", True) for r in res["launches"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--layout-check", action="store_true")
    ap.add_argument("--compare", action="store_true", help="CUDA vs CuTe at --stages")
    ap.add_argument("--timing", action="store_true", help="CUDA vs CuTe wall time of a launch stopped after the first of --stages")
    ap.add_argument("--fuse", type=int, default=18, help="Knobs.fuse of the CuTe kernel; the CUDA kernel gets fuse & 31 (19 = counter hand-off A->B, 50 = FUSE_ABL: CuTe 50, CUDA 18)")
    ap.add_argument("--wpre", type=int, default=DEFAULT_WPRE, help="Knobs.wpre (-DPI05_WPRE for the CUDA build)")
    ap.add_argument("--d-wpt", type=int, default=DEFAULT_D_WPT, help="Knobs.d_wpt (-DPI05_D_WPT for the CUDA build)")
    ap.add_argument("--host-zero", action="store_true", help="Knobs.host_zero: the host zeroes cnt..bar before each CuTe launch (A/B)")
    ap.add_argument("--check-reset", action="store_true", help="two back-to-back CuTe full launches: counters zero, 2nd == 1st")
    ap.add_argument("--n-steps", type=int, default=None, help="runtime step count of both kernels (debug truncation)")
    ap.add_argument("--n-layers", type=int, default=None, help="runtime layer count of both kernels (debug truncation)")
    ap.add_argument("--mask-tail", type=int, default=0, help="zero the last N prefix_mask entries")
    ap.add_argument("--prefix", type=int, default=968)
    ap.add_argument("--att-keys", type=int, default=64)
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    knobs = Knobs(prefix=args.prefix, att_keys=args.att_keys, fuse=args.fuse, wpre=args.wpre, d_wpt=args.d_wpt,
                  host_zero=args.host_zero)
    if args.check_reset:
        res = {"prefix": args.prefix, "att_keys": args.att_keys, "fuse": args.fuse, "seed": args.seed, "mask_tail": args.mask_tail,
               **check_reset(knobs, args.seed, args.mask_tail)}
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if args.layout_check:
        diffs = check_layout(knobs)
        print("layout equal" if not diffs else "\n".join(diffs))
        return 1 if diffs else 0
    if args.timing:
        stage = args.stages.split(",")[0] if "--stages" in (argv or sys.argv) else "pro"
        res = {"prefix": args.prefix, "att_keys": args.att_keys, "fuse": args.fuse, "stage": stage, "layer": args.layer,
               **time_stop(knobs, args.seed, args.mask_tail, stage, args.layer)}
        print("TIMING " + json.dumps(res))
        return 0
    if args.compare:
        res = {"prefix": args.prefix, "att_keys": args.att_keys, "fuse": args.fuse, "wpre": args.wpre, "d_wpt": args.d_wpt, "seed": args.seed,
               "mask_tail": args.mask_tail, "host_zero": args.host_zero, "step": args.step, "layer": args.layer,
               "stages": compare_stages(knobs, args.stages.split(","), args.step, args.layer, args.seed, args.mask_tail,
                                         args.n_steps, args.n_layers)}
        res["ok"] = all(v["equal"] for v in res["stages"].values())
        print(json.dumps(res, indent=1))
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, f"parity_compare_p{args.prefix}_k{args.att_keys}_s{args.seed}_m{args.mask_tail}_f{args.fuse}_st{args.step}_w{args.wpre}_d{args.d_wpt}_h{int(args.host_zero)}_l{args.layer}_{args.stages.replace(',', '')}.json"), "w") as f:
                json.dump(res, f, indent=1)
        return 0 if res["ok"] else 1
    if not args.selfcheck:
        ap.error("choose --selfcheck, --compare, --check-reset or --layout-check")
    res = {"prefix": args.prefix, "att_keys": args.att_keys, "step": args.step, "layer": args.layer,
           "stages": selfcheck(knobs, args.stages.split(","), args.step, args.layer, args.seed)}
    ok = all(v["deterministic"] for v in res["stages"].values() if isinstance(v, dict)) and \
        res["stages"].get("stop_at_end_equals_full", True)
    res["ok"] = bool(ok)
    print(json.dumps(res, indent=1))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"parity_selfcheck_p{args.prefix}_k{args.att_keys}.json"), "w") as f:
            json.dump(res, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
