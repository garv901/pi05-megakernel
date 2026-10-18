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
    """``k64`` -> {att_keys: 64, g_batch: 0, g_noinline: 0}; suffixes after ``k<N>`` in any order, each at most once:
    ``gb<0|1|2>`` = stage G load batching (PI05_G_BATCH / Knobs.g_batch), ``gn`` = stage G as a noinline function
    (PI05_G_NOINLINE; CUDA only). E.g. ``k64-gb1``, ``k64-gn``, ``k64-gb1-gn``. Shared by the CUDA and CuTe arms so they
    cannot drift."""
    keys, *suffixes = variant.split("-")
    if not keys.startswith("k") or not keys[1:].isdigit():
        raise ValueError(f"bad variant {variant!r}: expected k<att_keys>[-gb<0|1|2>][-gn]")
    out = {"att_keys": int(keys[1:]), "g_batch": 0, "g_noinline": 0}
    seen = set()
    for suffix in suffixes:
        kind = "gn" if suffix == "gn" else "gb" if suffix in ("gb0", "gb1", "gb2") else None
        if kind is None:
            raise ValueError(f"bad variant suffix {suffix!r} in {variant!r}: expected gb0, gb1, gb2 or gn")
        if kind in seen:
            raise ValueError(f"duplicate variant suffix {suffix!r} in {variant!r}")
        seen.add(kind)
        if kind == "gn":
            out["g_noinline"] = 1
        else:
            out["g_batch"] = int(suffix[2:])
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
    return load_ext(name=name, sources=[os.path.join(_HERE, "mk", "pi05_mk.cu")],
                    extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo",
                                       f"-DPI05_PREFIX={prefix}", f"-DPI05_ATT_KEYS={att_keys}", *extra],
                    build_directory=build, verbose=False)


class MegakernelArm:
    name = "megakernel"

    def __init__(self, geo: Geometry, variant: str, device: str, extra: tuple = (), fuse: int | None = None) -> None:
        if (geo.horizon, geo.action_dim, geo.layers, geo.steps) != (50, 32, 18, 10):
            raise ValueError("the megakernel is specialized for horizon 50, action 32, "
                             "18 layers, 10 steps")
        self.variant = variant
        self.fuse = FUSE if fuse is None else fuse
        pv = parse_variant(variant)
        self.att_keys, self.g_batch, self.g_noinline = pv["att_keys"], pv["g_batch"], pv["g_noinline"]
        if self.g_batch:  # gb0 is the plain build (same extension name)
            extra = (*extra, f"-DPI05_G_BATCH={self.g_batch}")
        if self.g_noinline:
            extra = (*extra, "-DPI05_G_NOINLINE=1")
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
        self.config = {"att_keys": self.att_keys, "g_batch": self.g_batch, "g_noinline": self.g_noinline, "fuse": self.fuse, "prefix": geo.prefix,
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
