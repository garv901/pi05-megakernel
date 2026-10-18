"""pi0.5 action chunk: ported megakernel vs SOTA multi-kernel baselines on one GPU.

    python -m megabench.sota.pi05 --prefix 968 --tag p968

Arms (each one complete 10-step action chunk per launch):
  megakernel/k{64,32,16}   ported single-launch kernel, attention key-chunk sweep
  cute/k{64,32,16}         the CuTe DSL port of it (cute/; opt-in: --arms megakernel,cute)
  sota-graph/graph-pdl     FlashInfer + autotuned GEMMs + fused Triton glue, one CUDA graph, PDL
  sota-graph/graph-nopdl   same graph without PDL
  sota-graph/eager         same kernels launched eagerly (launch-overhead reference)
  torch-compile/max-autotune   Inductor max-autotune step, chunk in one CUDA graph
  eager-ref/bf16           the MegaBench PyTorch reference (eager, what grades report)

Correctness: every arm on every seed, against the FP32 oracle with the trusted
MegaBench band (2x the BF16 reference's own error); inputs are checked for mutation.
Timing: arms interleaved in shuffled order per round. ``steady`` = back-to-back
launches / N; ``isolated`` = one launch after an L2 flush (2x L2 zero-fill), so all
weights and the prefix K/V come from HBM.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

import torch

from . import fixture as FX
from .fixture import Geometry

HBM_NOMINAL = 8.0e12
TC_PEAK = 2.25e15
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"


class Recorder:
    def __init__(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "records.jsonl")
        self.out_dir = out_dir

    def __call__(self, kind: str, **rec) -> None:
        rec = {"kind": kind, "utc": dt.datetime.utcnow().isoformat(timespec="seconds"), **rec}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"[{kind}] " + json.dumps(rec, default=str)[:400], flush=True)


# --------------------------------------------------------------------------- SOL
def sol(geo: Geometry) -> dict:
    """HBM floor with the 594 MB+ expert re-streamed every step (it cannot stay in L2)."""
    L, W, I, S, T = geo.layers, geo.hidden, geo.intermediate, geo.steps, geo.horizon
    qd, kvd = geo.q_heads * geo.head_dim, geo.kv_heads * geo.head_dim
    layer_params = W * (qd + 2 * kvd) + qd * W + 3 * W * I
    expert = L * layer_params * 2
    fp32_once = (L * 2 + 1) * 3 * W * W * 4 + 2 * W * W * 4 + 2 * W * geo.action_dim * 4
    kv = L * geo.prefix * kvd * 2 * 2
    bytes_stream = S * (expert + kv) + fp32_once
    bytes_once = expert + kv + fp32_once
    keys = geo.prefix + T
    flops = S * (2 * T * L * layer_params + L * 4 * T * qd * keys)
    return {"expert_weight_bytes": expert, "fp32_cond_weight_bytes": fp32_once,
            "prefix_kv_bytes": kv, "bytes_restream": bytes_stream, "bytes_read_once": bytes_once,
            "flops": flops,
            "floor_restream_ms": bytes_stream / HBM_NOMINAL * 1e3,
            "floor_read_once_ms": max(bytes_once / HBM_NOMINAL, flops / TC_PEAK) * 1e3,
            "tensor_floor_ms": flops / TC_PEAK * 1e3}


def measure_copy_bw(device) -> float:
    n = 1 << 30
    a = torch.empty(n, dtype=torch.uint8, device=device)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        b.copy_(a)
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e-3)
    del a, b
    return 2 * n / statistics.median(ts)


def env_info() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID")}
    for mod in ("flashinfer", "triton"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:  # noqa: BLE001
            info[mod] = _err(e)
    try:
        q = "clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active"
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "-i", dev, f"--query-gpu={q}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi"] = _err(e)
    return info


# --------------------------------------------------------------------------- arms
class Entry:
    """One (arm, variant) under test."""

    def __init__(self, arm: str, variant: str, obj, prepare=None) -> None:
        self.arm, self.variant, self.obj = arm, variant, obj
        self.prepare = prepare or (lambda: None)
        self.label = f"{arm}/{variant}"

    def launch(self) -> None:
        self.obj.launch()

    def actions(self) -> torch.Tensor:
        return self.obj.outputs()["actions"]


class EagerRef:
    def __init__(self, geo, inputs) -> None:
        self.geo, self.inputs, self.out = geo, inputs, None

    def launch(self) -> None:
        self.out = FX.reference(self.geo, self.inputs)["actions"]

    def outputs(self) -> dict:
        return {"actions": self.out}


def build_arms(args, geo, inputs, device, rec) -> list[Entry]:
    entries: list[Entry] = []
    arms = args.arms.split(",")
    if "megakernel" in arms:
        from . import megakernel as MK
        for v in args.mk_variants.split(","):
            try:
                t0 = time.time()
                mk = MK.MegakernelArm(geo, v, device)
                mk.bind(inputs)
                rec("build", arm="megakernel", variant=v, seconds=time.time() - t0, config=mk.config)
                entries.append(Entry("megakernel", v, mk))
            except Exception as e:  # noqa: BLE001
                rec("error", arm="megakernel", variant=v, stage="build", error=_err(e),
                    trace=traceback.format_exc()[-2000:])
    if "cute" in arms:  # CuTe DSL port of the megakernel (cute/); same variants and paths as the CUDA arm, not a default arm
        from .cute.arm import CuteArm
        for v in args.mk_variants.split(","):
            try:
                t0 = time.time()
                ca = CuteArm(geo, v, device)
                ca.bind(inputs)  # compiles here (cute.compile), never inside a timed region
                rec("build", arm="cute", variant=v, seconds=time.time() - t0, config=ca.config)
                entries.append(Entry("cute", v, ca))
            except Exception as e:  # noqa: BLE001
                rec("error", arm="cute", variant=v, stage="build", error=_err(e), trace=traceback.format_exc()[-2000:])
    if "sota-graph" in arms:
        from .sota_graph import SotaGraphArm, VARIANTS
        try:
            t0 = time.time()
            sg = SotaGraphArm(geo, inputs, device)
            rec("build", arm="sota-graph", seconds=time.time() - t0, config=sg.config())
            for v in VARIANTS:
                def prep(v=v, sg=sg):
                    sg.set_variant(v)
                try:
                    prep()
                    entries.append(Entry("sota-graph", v, sg, prep))
                except Exception as e:  # noqa: BLE001
                    rec("error", arm="sota-graph", variant=v, stage="capture", error=_err(e),
                        trace=traceback.format_exc()[-2000:])
        except Exception as e:  # noqa: BLE001
            rec("error", arm="sota-graph", stage="build", error=_err(e),
                trace=traceback.format_exc()[-2000:])
    if "torch-compile" in arms:
        from .compile_arm import CompileArm
        try:
            t0 = time.time()
            ca = CompileArm(geo, inputs, device)
            ca.set_variant("max-autotune")
            rec("build", arm="torch-compile", seconds=time.time() - t0, config=ca.config())
            entries.append(Entry("torch-compile", "max-autotune", ca))
        except Exception as e:  # noqa: BLE001
            rec("error", arm="torch-compile", stage="build", error=_err(e),
                trace=traceback.format_exc()[-2000:])
    if "eager-ref" in arms:
        entries.append(Entry("eager-ref", "bf16", EagerRef(geo, inputs)))
    return entries


def refresh(entries: list[Entry]) -> None:
    seen = set()
    for e in entries:
        if id(e.obj) in seen:
            continue
        seen.add(id(e.obj))
        if hasattr(e.obj, "refresh_weights"):
            e.obj.refresh_weights()


# --------------------------------------------------------------------------- phases
def correctness(args, geo, inputs, entries, rec, device) -> dict[str, bool]:
    case = FX.make_case(geo)
    ok = {e.label: True for e in entries}
    for n, seed in enumerate(int(s) for s in args.seeds.split(",")):
        if n:
            fresh = FX.make_inputs(geo, seed, device)
            for k, v in fresh.items():
                inputs[k].copy_(v)
            del fresh
            refresh(entries)
            torch.cuda.synchronize()
        want = FX.reference(geo, inputs)["actions"]
        exact = FX.oracle(geo, inputs)["actions"]
        snap = FX.snapshot(inputs)
        for e in entries:
            try:
                e.prepare()
                e.launch()
                torch.cuda.synchronize()
                got = e.actions().clone()
                e.launch()  # idempotence: a second launch must give the same answer
                torch.cuda.synchronize()
                repeat = float((e.actions() - got).abs().max())
                d = FX.band(want, got, exact, case)
                d["repeat_max_abs"] = repeat  # recorded; split-K / atomics may be nondeterministic
                ok[e.label] &= d["pass"]
                rec("correctness", arm=e.arm, variant=e.variant, seed=seed, **d)
            except Exception as ex:  # noqa: BLE001
                ok[e.label] = False
                rec("error", arm=e.arm, variant=e.variant, stage="correctness", seed=seed,
                    error=_err(ex), trace=traceback.format_exc()[-2000:])
        mutated = [k for k, v in FX.snapshot(inputs).items() if v != snap[k]]
        rec("mutation_check", seed=seed, mutated=mutated)
        if mutated:
            for k in ok:
                ok[k] = False
    return ok


def _flush(buf: torch.Tensor) -> None:
    buf.zero_()


def timing(args, entries, rec, device) -> dict[str, dict]:
    l2 = torch.cuda.get_device_properties(device).L2_cache_size
    flush = torch.empty(2 * l2, dtype=torch.uint8, device=device)
    samples = {e.label: {"steady": [], "isolated": []} for e in entries}
    rng = random.Random(args.seed_order)
    for r in range(args.rounds):
        order = entries[:]
        rng.shuffle(order)
        for e in order:
            e.prepare()
            slow = e.arm == "eager-ref"
            n_steady = 2 if slow else args.steady
            n_iso = 1 if slow else args.iters
            for _ in range(1 if slow else 3):
                e.launch()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(n_steady):
                e.launch()
            e1.record()
            e1.synchronize()
            samples[e.label]["steady"].append(e0.elapsed_time(e1) / n_steady)
            for _ in range(n_iso):
                _flush(flush)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                e.launch()
                b.record()
                b.synchronize()
                samples[e.label]["isolated"].append(a.elapsed_time(b))
        rec("round", round=r, order=[e.label for e in order])
    summary = {}
    for e in entries:
        s = samples[e.label]
        summary[e.label] = {
            mode: {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v),
                   "n": len(v)} for mode, v in s.items()}
        rec("timing", arm=e.arm, variant=e.variant, **summary[e.label], samples=s)
    return summary


def kernel_counts(entries, rec) -> dict[str, dict]:
    from torch.profiler import ProfilerActivity, profile
    out = {}
    for e in entries:
        if e.arm == "eager-ref":
            continue
        try:
            e.prepare()
            e.launch()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                e.launch()
                torch.cuda.synchronize()
            names: dict[str, int] = {}
            busy = 0.0
            for ev in prof.events():
                if ev.device_type == torch.autograd.DeviceType.CUDA and \
                        not ev.name.startswith(("Memcpy", "Memset", "cudaGraph")):
                    names[ev.name] = names.get(ev.name, 0) + 1
                    busy += ev.device_time
            n = sum(names.values())
            top = sorted(names.items(), key=lambda kv: -kv[1])[:12]
            out[e.label] = {"kernels": n, "busy_ms": busy / 1e3,
                            "top": [(k[:90], v) for k, v in top]}
            rec("kernels", arm=e.arm, variant=e.variant, **out[e.label])
        except Exception as ex:  # noqa: BLE001
            rec("error", arm=e.arm, variant=e.variant, stage="profile", error=_err(ex))
    return out


def write_summary(out_dir, geo, floor, bw, ok, tim, kern) -> str:
    lines = [f"# pi0.5 action chunk, B=1, prefix KV={geo.prefix}, horizon {geo.horizon}, "
             f"{geo.steps} steps x {geo.layers} layers", "",
             f"HBM floor (expert re-streamed each step): {floor['floor_restream_ms']:.3f} ms at 8 TB/s, "
             f"{floor['bytes_restream'] / bw * 1e3:.3f} ms at measured copy BW {bw / 1e12:.2f} TB/s; "
             f"read-once floor (sol_info style) {floor['floor_read_once_ms']:.3f} ms.", "",
             "| arm / variant | correct | steady ms | isolated (L2 flushed) ms | kernels/chunk "
             "| x restream floor (isolated) |", "|---|---|---|---|---|---|"]
    for label, t in sorted(tim.items(), key=lambda kv: kv[1]["isolated"]["median_ms"]):
        k = kern.get(label, {}).get("kernels", "")
        iso = t["isolated"]["median_ms"]
        lines.append(f"| {label} | {'pass' if ok.get(label) else 'FAIL'} | "
                     f"{t['steady']['median_ms']:.3f} | {iso:.3f} | {k} | "
                     f"{iso / floor['floor_restream_ms']:.2f}x |")
    good = {k: v for k, v in tim.items() if ok.get(k)}
    mk = {k: v for k, v in good.items() if k.startswith("megakernel/")}
    base = {k: v for k, v in good.items() if not k.startswith(("megakernel/", "cute/", "eager"))
            and "/eager" not in k}
    if mk and base:
        for mode in ("steady", "isolated"):
            bm = min(mk, key=lambda k: mk[k][mode]["median_ms"])
            bb = min(base, key=lambda k: base[k][mode]["median_ms"])
            r = base[bb][mode]["median_ms"] / mk[bm][mode]["median_ms"]
            lines.append("")
            lines.append(f"**{mode}: best megakernel {bm} {mk[bm][mode]['median_ms']:.3f} ms vs best "
                         f"baseline {bb} {base[bb][mode]['median_ms']:.3f} ms -> megakernel speedup "
                         f"{r:.2f}x**")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text)
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--prefix", type=int, default=968)
    p.add_argument("--arms", default="megakernel,sota-graph,torch-compile,eager-ref")
    p.add_argument("--mk-variants", default="k64,k32,k16")
    p.add_argument("--seeds", default="7301,8917,42811")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--iters", type=int, default=10, help="isolated samples per arm per round")
    p.add_argument("--steady", type=int, default=20, help="back-to-back launches per round")
    p.add_argument("--seed-order", type=int, default=0)
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--tag", default="run")
    args = p.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out_dir = os.path.join(RUNS, dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + args.tag)
    rec = Recorder(out_dir)
    device = "cuda"
    geo = Geometry(prefix=args.prefix)
    floor = sol(geo)
    rec("env", **env_info(), args=vars(args))
    bw = measure_copy_bw(device)
    rec("sol", geometry=geo.__dict__, measured_copy_bw=bw, **floor)
    seeds = [int(s) for s in args.seeds.split(",")]
    inputs = FX.make_inputs(geo, seeds[0], device)
    entries = build_arms(args, geo, inputs, device, rec)
    ok = correctness(args, geo, inputs, entries, rec, device)
    rec("correctness_summary", ok=ok)
    tim = timing(args, entries, rec, device)
    kern = {} if args.no_profile else kernel_counts(entries, rec)
    text = write_summary(out_dir, geo, floor, bw, ok, tim, kern)
    print(text)
    print("out_dir", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
