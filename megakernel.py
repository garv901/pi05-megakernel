"""The ported pi0.5 megakernel (mk/pi05_mk.cu) as a benchmark arm.

One cooperative launch per action chunk. Each variant is a separate extension
build: the prefix length is the compile-time ``PI05_PREFIX`` and the attention
key-chunk size ``PI05_ATT_KEYS`` (64 in the graded P=968 build) is swept,
because at P=128 the 64-key chunking leaves 7 x 3 = 21 attention units for 148 SMs.
The launch wrapper is the run's ``submission.py`` minus its fixed-geometry check.
"""

from __future__ import annotations

import os

import torch

from .fixture import Geometry

_HERE = os.path.dirname(os.path.abspath(__file__))
# Same epilogue handoff bits as the graded submission (_FUSE = 18).
FUSE = 18
ORDER = ("action_in_weight", "action_in_bias", "time_in_weight", "time_in_bias",
         "time_out_weight", "time_out_bias", "action_out_weight", "action_out_bias",
         "wq", "wk", "wv", "wo", "wg", "wu", "wd",
         "attn_norm_weight", "attn_norm_bias", "ffn_norm_weight", "ffn_norm_bias",
         "final_norm_weight", "final_norm_bias", "noise", "prefix_k", "prefix_v", "prefix_mask")
ATT_KEYS = (64, 32, 16)


def parse_variant(variant: str) -> dict:
    """``k64`` -> {att_keys: 64, load_hoist: 0, hoist_off: False, spread_off: False}; optional suffixes, each at most once:
    ``lh`` = Knobs.load_hoist 1 (CuTe: gemm_phase next-tile weight load hoisted above the partial stores), ``nolh`` = CUDA
    built with PI05_LOAD_HOIST=0 (not with lh), ``nosp`` = CUDA built with PI05_GEMM_SPREAD=0 (slot-major gemm_phase tasks),
    ``nopb`` = CUDA built with PI05_PART_BF16=0 (fp32 split-K partials, the pre-job-8802 build).
    ``fgu`` = CUDA built with PI05_FG_UMMA=1 (tcgen05 cta_group::2 F+G, cooperative cluster-2 launch, sm_100a). CUDA arm only:
    the CuTe arm ignores it (cute/ is untouched).
    The CUDA kernel also uses bf16 partials by default (PI05_PART_BF16 default 1), so ``k64-nopb`` is that A/B control.
    The CUDA kernel hoists and spreads by default (both macros default 1), so for the CUDA arm ``k64`` and ``k64-lh`` are the
    same build and ``k64-nolh`` / ``k64-nosp`` are the A/B controls. The CuTe arm reads att_keys and load_hoist only.
    E.g. ``k64-nosp``. Shared by the CUDA and CuTe arms so they cannot drift."""
    keys, *suffixes = variant.split("-")
    if not keys.startswith("k") or not keys[1:].isdigit():
        raise ValueError(f"bad variant {variant!r}: expected k<att_keys>[-lh|-nolh][-nosp][-nopb][-fgu]")
    out = {"att_keys": int(keys[1:]), "load_hoist": 0, "hoist_off": False, "spread_off": False, "part_bf16_off": False, "fgu": False}
    if len(set(suffixes)) != len(suffixes):
        raise ValueError(f"duplicate variant suffix in {variant!r}")
    if "lh" in suffixes and "nolh" in suffixes:
        raise ValueError(f"lh and nolh together in {variant!r}")
    for suffix in suffixes:
        if suffix == "lh":
            out["load_hoist"] = 1
        elif suffix == "nolh":
            out["hoist_off"] = True
        elif suffix == "nosp":
            out["spread_off"] = True
        elif suffix == "nopb":
            out["part_bf16_off"] = True
        elif suffix == "fgu":
            out["fgu"] = True
        else:
            raise ValueError(f"bad variant suffix {suffix!r} in {variant!r}: expected lh, nolh, nosp, nopb or fgu")
    return out


def variants() -> tuple[str, ...]:
    return tuple(f"k{k}" for k in ATT_KEYS)


def load(prefix: int, att_keys: int, extra: tuple = ()):
    from torch.utils.cpp_extension import load as load_ext
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0"
    name = f"pi05_mk_p{prefix}_k{att_keys}"
    if extra:  # probe-only extra -D flags (ablations); default build is unchanged
        import hashlib
        name += "_x" + hashlib.sha1(" ".join(extra).encode()).hexdigest()[:8]
    build = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp"), name)
    os.makedirs(build, exist_ok=True)
    # -DPI05_FG_UMMA=1 (tcgen05 F/G) needs sm_100a: build for 10.0a only, for that build only (the default build's flags are unchanged).
    umma = "-DPI05_FG_UMMA=1" in extra
    saved = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if umma:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    try:
        return load_ext(name=name, sources=[os.path.join(_HERE, "mk", "pi05_mk.cu")],
                        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo",
                                           f"-DPI05_PREFIX={prefix}", f"-DPI05_ATT_KEYS={att_keys}", *extra],
                        build_directory=build, verbose=False)
    finally:
        if umma:
            if saved is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = saved


class MegakernelArm:
    name = "megakernel"

    def __init__(self, geo: Geometry, variant: str, device: str, extra: tuple = (), fuse: int | None = None) -> None:
        if (geo.horizon, geo.action_dim, geo.layers, geo.steps) != (50, 32, 18, 10):
            raise ValueError("the megakernel is specialized for horizon 50, action 32, "
                             "18 layers, 10 steps")
        self.variant = variant
        self.fuse = FUSE if fuse is None else fuse
        pv = parse_variant(variant)
        self.att_keys, self.load_hoist = pv["att_keys"], 0 if pv["hoist_off"] else 1
        if not self.load_hoist and "-DPI05_LOAD_HOIST=0" not in extra:  # the plain build hoists (PI05_LOAD_HOIST default 1)
            extra = (*extra, "-DPI05_LOAD_HOIST=0")
        self.spread = not pv["spread_off"]
        if not self.spread and "-DPI05_GEMM_SPREAD=0" not in extra:  # the plain build spreads (PI05_GEMM_SPREAD default 1)
            extra = (*extra, "-DPI05_GEMM_SPREAD=0")
        self.part_bf16 = not pv["part_bf16_off"]
        if not self.part_bf16 and "-DPI05_PART_BF16=0" not in extra:  # the plain build uses bf16 partials (PI05_PART_BF16 default 1)
            extra = (*extra, "-DPI05_PART_BF16=0")
        self.fgu = pv["fgu"]
        if self.fgu and "-DPI05_FG_UMMA=1" not in extra:
            extra = (*extra, "-DPI05_FG_UMMA=1")
        self.geo = geo
        self.ext = load(geo.prefix, self.att_keys, extra)
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.ext.setup(self.device.index if self.device.index is not None
                           else torch.cuda.current_device())
            self.ws = torch.empty(self.ext.workspace_bytes(), dtype=torch.uint8,
                                  device=self.device)
            self.no_timing = torch.empty(0, dtype=torch.int64, device=self.device)
        self.actions = torch.empty((geo.horizon, geo.action_dim), dtype=torch.float32,
                                   device=self.device)
        self.tensors: list[torch.Tensor] | None = None
        self.config = {"att_keys": self.att_keys, "load_hoist": self.load_hoist, "spread": int(self.spread), "part_bf16": int(self.part_bf16), "fg_umma": int(self.fgu), "fuse": self.fuse, "prefix": geo.prefix,
                       "launches_per_chunk": 1}

    def bind(self, inputs: dict) -> None:
        tensors = []
        for name in ORDER:
            value = inputs[name]
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            tensors.append(value)
        self.tensors = tensors

    def launch(self) -> None:
        self.ext.launch(self.tensors, self.actions, self.ws, self.geo.steps, self.geo.layers,
                        self.no_timing, self.fuse)

    def outputs(self) -> dict:
        return {"actions": self.actions}

    def close(self) -> None:
        self.tensors = None
