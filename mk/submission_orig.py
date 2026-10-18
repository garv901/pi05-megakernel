"""pi0.5 action chunk as one persistent cooperative CUDA kernel (see pi05_mk.cu)."""

from __future__ import annotations

import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXPECTED = {'batch': 1, 'images': 3, 'image_tokens': 256, 'text_tokens': 200, 'layers': 18,
             'q_heads': 8, 'kv_heads': 1, 'head_dim': 256, 'expert_hidden': 1024,
             'expert_intermediate': 4096, 'horizon': 50, 'action_dim': 32, 'denoise_steps': 10}
# Epilogues handed off through arrival counters instead of grid barriers
# (bits: 1 QKV reduce+RoPE, 2 attention combine, 4 gelu*up, 8 spin backoff) and 16 = grid-parallel
# residual with the adaRMS norm applied in the next GEMM's A staging.
_FUSE = 18
_ORDER = ("action_in_weight", "action_in_bias", "time_in_weight", "time_in_bias",
          "time_out_weight", "time_out_bias", "action_out_weight", "action_out_bias",
          "wq", "wk", "wv", "wo", "wg", "wu", "wd",
          "attn_norm_weight", "attn_norm_bias", "ffn_norm_weight", "ffn_norm_bias",
          "final_norm_weight", "final_norm_bias", "noise", "prefix_k", "prefix_v", "prefix_mask")


def _build_dir() -> str:
    for base in (os.path.join(_HERE, ".torch_extensions"), "/tmp"):
        path = os.path.join(base, "pi05_mk_build")
        try:
            os.makedirs(path, exist_ok=True)
            if os.access(path, os.W_OK):
                return path
        except OSError:
            continue
    raise RuntimeError("no writable build directory")


def load_extension():
    from torch.utils.cpp_extension import load
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return load(name="pi05_mk", sources=[os.path.join(_HERE, "pi05_mk.cu")],
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
                build_directory=_build_dir(), verbose=False)


def build(case: dict):
    params = case["params"]
    for key, value in _EXPECTED.items():
        if params[key] != value:
            raise ValueError(f"kernel is specialized for {key}={value}, got {params[key]}")
    ext = load_extension()
    state: dict = {}

    def _prepare(device: torch.device):
        if state.get("device") != device:
            with torch.cuda.device(device):
                ext.setup(device.index if device.index is not None else torch.cuda.current_device())
                state["ws"] = torch.empty(ext.workspace_bytes(), dtype=torch.uint8, device=device)
                state["no_timing"] = torch.empty(0, dtype=torch.int64, device=device)
            state["device"] = device

    def run(inputs: dict, n_steps: int = 10, n_layers: int = 18, timing=None, fuse: int = _FUSE) -> dict:
        tensors = []
        for name in _ORDER:
            value = inputs[name]
            if not value.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            tensors.append(value)
        device = tensors[0].device
        _prepare(device)
        actions = torch.empty((50, 32), dtype=torch.float32, device=device)
        ext.launch(tensors, actions, state["ws"], n_steps, n_layers,
                   state["no_timing"] if timing is None else timing, fuse)
        return {"actions": actions}

    run.ext = ext
    run.state = state
    return run
