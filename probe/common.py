"""Shared plumbing for the pi0.5 gap probes: recorder, arm construction, phase timing helpers."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time

import torch

from .. import fixture as FX
from ..fixture import Geometry

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
TUNED_JSON = os.environ.get("PROBE_TUNED") or os.path.join(HERE, "tuned.json")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Recorder:
    def __init__(self, out_dir: str, name: str = "records.jsonl") -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, name)

    def __call__(self, kind: str, **rec) -> None:
        rec = {"kind": kind, "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **rec}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"[{kind}] " + json.dumps(rec, default=str)[:300], flush=True)


def out_dir_from_env(tag: str) -> str:
    d = os.environ.get("PROBE_OUT") or os.path.join(RUNS, utc() + "-" + tag)
    os.makedirs(d, exist_ok=True)
    return d


def setup_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_geo_inputs(prefix: int = int(os.environ.get("PROBE_PREFIX", "128")), seed: int = 7301):
    geo = Geometry(prefix=prefix)
    return geo, FX.make_inputs(geo, seed, "cuda")


def build_graph_arm(geo, inputs):
    """SotaGraphArm with the persisted tuned config (no 14-minute autotune)."""
    from ..sota_graph import Attention, Gemm, SotaGraphArm
    with open(TUNED_JSON) as f:
        cfg = json.load(f)
    gemms = {t: Gemm(b) for t, b in cfg["gemm"].items()}
    a = Attention(cfg["attn"], geo, "cuda", inputs["prefix_mask"])
    # tgv tactic (tile / split-K) is chosen by an in-process autotune pass per shape
    import flashinfer
    T = geo.horizon
    arm = SotaGraphArm(geo, inputs, "cuda", tuned={"gemm": gemms, "attn": a, "record": {"persisted": cfg}})
    g = torch.Generator(device="cuda").manual_seed(5)
    for tag, w in (("qkv", arm.w_qkv[0]), ("o", inputs["wo"][0]), ("gu", arm.w_gu[0]), ("down", inputs["wd"][0])):
        if gemms[tag].backend == "cublas":
            continue
        n, k = w.shape
        x = torch.randn(T, k, generator=g, device="cuda").to(torch.bfloat16)
        out = torch.empty(T, n, dtype=torch.bfloat16, device="cuda")
        with flashinfer.autotune(True):
            gemms[tag](x, w, out, True)
        gemms[tag](x, w, out, True)
    torch.cuda.synchronize()
    return arm


def pre_flip(extra) -> tuple:
    """Extra -D flags of a build named relative to the pre-job-9134 default (PI05_FG_UMMA and PI05_ATT_K2 off): each
    of the two is turned off unless ``extra`` sets it, so ``()`` is the old default build and ``("-DPI05_FG_UMMA=1",)``
    the old umma build. The A/B drivers written before the default flip build through this to keep their meaning."""
    off = tuple(f"-DPI05_{m}=0" for m in ("FG_UMMA", "ATT_K2") if not any(x.startswith(f"-DPI05_{m}=") for x in extra))
    return (*off, *extra)


def build_mk_arm(geo, inputs, variant: str = "k16", tstamp: bool = False, extra=(), fuse=None, tstamp_size: int = 4096):
    from .. import megakernel as MK
    arm = MK.MegakernelArm(geo, variant, "cuda", extra=tuple(extra), fuse=fuse)
    arm.bind(inputs)
    if tstamp:
        arm.no_timing = torch.zeros(tstamp_size, dtype=torch.int64, device="cuda")
    return arm


class Launcher:
    """Uniform launch()/prepare for drive scripts."""

    def __init__(self, arm_name: str, geo, inputs, **kw):
        self.name = arm_name
        if arm_name.startswith("mk"):
            variant = arm_name.split(":")[1] if ":" in arm_name else "k16"
            self.obj = build_mk_arm(geo, inputs, variant, **kw)
        else:
            self.obj = build_graph_arm(geo, inputs)
            self.obj.set_variant(arm_name)  # "graph-pdl" | "graph-nopdl"

    def launch(self):
        self.obj.launch()


def iso_times(launcher, n: int, flush=None) -> list[float]:
    """ms per launch after an L2 flush (the bench.py 'isolated' protocol)."""
    out = []
    for _ in range(n):
        if flush is not None:
            flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        launcher.launch()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b))
    return out


def l2_flush(dev="cuda"):
    sz = torch.cuda.get_device_properties(dev).L2_cache_size
    return torch.empty(2 * sz, dtype=torch.uint8, device=dev)
