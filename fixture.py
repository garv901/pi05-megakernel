"""pi0.5 action-chunk fixture at a chosen prefix (KV) length, plus the grading band.

Inputs follow the MegaBench pi0.5 task (``reference/pi05.py``, vendored from the
trusted snapshot): the action expert's weights, noise and the per-layer BF16
post-RoPE prefix K/V. The prefix K/V comes from the task's Gemma-2B ``prefill``
math over ``prefix`` fully valid tokens with unit-variance synthetic embeddings
(the graded case runs SigLIP over 3 images plus a 200-token prompt to get 968).
Only K/V and the mask reach the timed step, so the vision tower is skipped.

Correctness is the trusted MegaBench band: each arm's error against the FP32
oracle must stay within ``NOISE_FACTOR`` x the BF16 reference's own error (+ rtol),
on relative L2 and on max-abs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from megabench.cases import Case
from megabench.tasks.common import fill_layer_weights, random_norm

from .reference import pi05 as ref
from .reference.oracle import fp32_reference

NOISE_FACTOR = 2.0


@dataclass(frozen=True)
class Geometry:
    prefix: int = 968
    horizon: int = 50
    action_dim: int = 32
    layers: int = 18
    steps: int = 10
    hidden: int = 1024          # expert width
    intermediate: int = 4096
    q_heads: int = 8
    kv_heads: int = 1
    head_dim: int = 256
    vlm_hidden: int = 2048
    vlm_intermediate: int = 16384

    @property
    def id(self) -> str:
        return f"pi05-b1-p{self.prefix}"


def make_case(geo: Geometry) -> Case:
    params = {"batch": 1, "images": 3, "image_tokens": 256, "image_size": 224,
              "patch_size": 14, "text_tokens": 200, "vision_hidden": 1152,
              "vision_layers": 27, "vision_intermediate": 4304, "vision_heads": 16,
              "layers": geo.layers, "hidden": geo.vlm_hidden,
              "intermediate": geo.vlm_intermediate, "q_heads": geo.q_heads,
              "kv_heads": geo.kv_heads, "head_dim": geo.head_dim,
              "expert_hidden": geo.hidden, "expert_intermediate": geo.intermediate,
              "horizon": geo.horizon, "action_dim": geo.action_dim,
              "denoise_steps": geo.steps, "prefix": geo.prefix}
    return Case(id=geo.id, family="vla_action_step", model="lerobot/pi05_base",
                phase="action_chunk", params=params, category="vision_language_action",
                suite="p1", atol=0.003, rtol=0.003, bf16_rtol=0.008, ready=True,
                note="SOTA comparison fixture; prefix K/V from Gemma-2B prefill math")


def _prefix_kv(case: Case, geo: Geometry, seed: int, device: str):
    p = case.params
    g = torch.Generator(device=device).manual_seed(seed ^ 0x9A11)
    h, layers, inter = p["hidden"], p["layers"], p["intermediate"]
    qd, kvd = p["q_heads"] * p["head_dim"], p["kv_heads"] * p["head_dim"]
    vlm = {"ln1": random_norm(g, device, (layers, h), gemma=True),
           "ln2": random_norm(g, device, (layers, h), gemma=True)}
    for name, rows, cols in (("lq", qd, h), ("lk", kvd, h), ("lv", kvd, h),
                             ("lo", h, qd), ("lg", inter, h), ("lu", inter, h),
                             ("ld", h, inter)):
        vlm[name] = fill_layer_weights(g, device, (layers, rows, cols), cols)
    embeddings = torch.randn(geo.prefix, h, generator=g, device=device).to(torch.bfloat16)
    valid = torch.ones(geo.prefix, dtype=torch.bool, device=device)
    k, v = ref.prefill(case, vlm, embeddings, valid)
    del vlm
    return k.contiguous(), v.contiguous(), valid


def make_inputs(geo: Geometry, seed: int, device: str = "cuda") -> dict[str, torch.Tensor]:
    case = make_case(geo)
    with torch.no_grad():
        prefix_k, prefix_v, mask = _prefix_kv(case, geo, seed, device)
        g = torch.Generator(device=device).manual_seed(seed ^ 0xAC71)
        values = ref.expert_inputs(case, g, device)
        values["noise"] = torch.randn((geo.horizon, geo.action_dim), generator=g, device=device)
    values.update(prefix_k=prefix_k, prefix_v=prefix_v, prefix_mask=mask)
    torch.cuda.synchronize()
    return values


def reference(geo: Geometry, inputs: dict) -> dict:
    with torch.no_grad():
        return ref.reference(make_case(geo), inputs)


def oracle(geo: Geometry, inputs: dict) -> dict:
    return fp32_reference(ref, make_case(geo), inputs)


def band(want: torch.Tensor, got: torch.Tensor, exact: torch.Tensor, case: Case) -> dict:
    """Trusted ``harness.correctness._band`` (noise factor 2)."""
    exact = exact.double()
    error, ref_error = got.double() - exact, want.double() - exact
    scale = exact.norm().clamp_min(torch.finfo(torch.float64).tiny)
    rtol = case.bf16_rtol if want.dtype == torch.bfloat16 else case.rtol
    k = NOISE_FACTOR
    d = {"relative_error": float(error.norm() / scale),
         "reference_relative_error": float(ref_error.norm() / scale),
         "max_abs_error": float(error.abs().max()),
         "reference_max_abs_error": float(ref_error.abs().max())}
    d["pass"] = bool(torch.isfinite(got).all()) and bool(
        d["relative_error"] <= k * d["reference_relative_error"] + rtol and
        d["max_abs_error"] <= (k * d["reference_max_abs_error"] + case.atol
                               + rtol * float(exact.abs().max())))
    return d


def snapshot(inputs: dict) -> dict:
    """Cheap per-tensor fingerprints to detect input mutation."""
    return {k: (v.float().sum().item(), v.float().abs().sum().item())
            for k, v in inputs.items()}
