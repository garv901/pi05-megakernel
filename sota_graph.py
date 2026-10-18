"""SOTA multi-kernel pi0.5 action chunk: FlashInfer + cuBLAS/FlashInfer GEMMs + fused
Triton glue, the whole 10-step chunk in ONE CUDA graph, with PDL on every kernel
that supports it.

Per call (all captured in one graph):
  prologue  time MLP for all 10 timesteps (M=10), the adaRMS modulation of every
            norm and step as two batched FP32 GEMMs + one GEMM (hoisted: the 10 times
            are fixed by the schedule), prefix K/V copy into the static KV buffers,
            valid-prefix count, noise -> x, step-0 head (action_in + layer-0 norm).
  per layer qkv GEMM | rope+q/k/v append | attention | o GEMM | residual+adaRMS |
            gate_up GEMM | gelu_tanh_and_mul | down GEMM | residual+next adaRMS
            (the last layer's residual is the step-boundary kernel: final norm,
            action_out, Euler update, next step's action_in and layer-0 norm).
  = 9 kernels per layer, 1620 per chunk, plus ~12 prologue kernels.

QKV and gate/up weights are concatenated at build time (as vLLM does at load);
``refresh_weights`` redoes that after inputs are overwritten in place.

Variants: ``graph-pdl`` (headline), ``graph-nopdl`` and ``eager`` (no graph, no PDL).
Attention and GEMM backends are autotuned once (under PDL) and shared by all variants.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable

import torch
import torch.nn.functional as F

from . import triton_ops as K
from .fixture import Geometry
from .reference import pi05 as ref

VARIANTS = ("graph-pdl", "graph-nopdl", "eager")
_FACTORS = {"graph-pdl": (True, True), "graph-nopdl": (True, False), "eager": (False, False)}
_FI_GEMM = ("cudnn", "cutlass", "cublaslt", "tgv", "tinygemm", "cutile", "cute-dsl")
_FI_GEMM_PDL = {"tgv", "tinygemm", "cute-dsl"}
_ATTN_FI = ("fa2", "cutlass", "cudnn", "cute-dsl", "auto")
_ATTN_SDPA = ("flash", "cudnn", "efficient")
_TOL = 1e-2
E2E_TOP = 3


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"


def _rel(a, b) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _time_graph(fn: Callable[[], None], reps: int = 30) -> float:
    """Median ms of one replay of a CUDA graph of ``fn``."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    del g
    return statistics.median(times)


# --------------------------------------------------------------------------- GEMM
class Gemm:
    """``out = a @ w.T`` through a chosen backend; ``w`` is the [n, k] nn.Linear weight."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.supports_pdl = backend in _FI_GEMM_PDL
        if backend != "cublas":
            import flashinfer
            self._mm = flashinfer.mm_bf16

    def __call__(self, a, w, out, pdl: bool) -> None:
        if self.backend == "cublas":
            torch.mm(a, w.t(), out=out)
        else:
            self._mm(a, w.t(), pdl=bool(pdl and self.supports_pdl), out=out,
                     backend=self.backend)


def _cold_set(weights: list, device) -> list:
    """Enough copies of the per-layer weights that the set is 3x L2: as in the real
    chunk (622 MB of expert weights per step), every call streams from HBM."""
    l2 = torch.cuda.get_device_properties(device).L2_cache_size
    total = sum(w.numel() * w.element_size() for w in weights)
    reps = max(1, -(-3 * l2 // total))
    return list(weights) + [w.clone() for _ in range(reps - 1) for w in weights]


def autotune_gemm(tag: str, weights: list, m: int, device) -> tuple[list, dict]:
    """All valid backends for ``m`` rows, ranked by L2-cold time per call.

    FlashInfer backends first run under ``flashinfer.autotune`` so their tactic
    (tile / split-K / stages) is profiled for this exact shape and cached."""
    import flashinfer
    n, k = weights[0].shape
    g = torch.Generator(device=device).manual_seed(5)
    a = torch.randn(m, k, generator=g, device=device).to(torch.bfloat16)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    want = torch.mm(a.float(), weights[0].float().t())
    cold = _cold_set(weights, device)
    record: dict[str, Any] = {}
    ranked: list[tuple[float, Gemm]] = []
    for be in ("cublas",) + _FI_GEMM:
        try:
            gemm = Gemm(be)
            if be != "cublas":
                try:
                    with flashinfer.autotune(True):
                        gemm(a, weights[0], out, True)
                except Exception as e:  # noqa: BLE001
                    record[be + ":tactic_autotune"] = _err(e)
            gemm(a, weights[0], out, True)
            torch.cuda.synchronize()
            rel = _rel(out, want)
            if not rel <= _TOL:
                record[be] = f"invalid: rel_l2={rel:.2e}"
                continue

            def many():
                for w in cold:
                    gemm(a, w, out, True)
            ms = _time_graph(many, reps=10) / len(cold)
            record[be] = {"us_per_call": ms * 1e3, "rel_l2": rel, "pdl": gemm.supports_pdl,
                          "GBps": n * k * 2 / (ms * 1e-3) / 1e9}
            ranked.append((ms, gemm))
        except Exception as e:  # noqa: BLE001
            record[be] = _err(e)
    if not ranked:
        raise RuntimeError(f"no GEMM backend for {tag} ({m}x{n}x{k}): {record}")
    ranked.sort(key=lambda t: t[0])
    del cold
    return [gm for _, gm in ranked], record


# --------------------------------------------------------------------------- attention
class Attention:
    """Non-causal single-KV-head GQA of [T, Hq, D] queries over [S, 1, D] keys/values.

    ``triton`` honours the prefix mask; the library kernels see every key (the fixture's
    prefix is fully valid, so all candidates compute the same function)."""

    def __init__(self, cfg: dict, geo: Geometry, device, mask: torch.Tensor) -> None:
        self.cfg, self.geo = cfg, geo
        T, S, H, D = geo.horizon, geo.prefix + geo.horizon, geo.q_heads, geo.head_dim
        self.scale = 1.0 / math.sqrt(D)
        self.mask = mask.view(torch.uint8)
        self.supports_pdl = cfg["kind"] in ("fi", "triton")
        if cfg["kind"] == "fi":
            import flashinfer
            i32 = dict(dtype=torch.int32, device=device)
            if cfg["backend"] == "cudnn":   # cuDNN ragged indptrs are element offsets
                self.qo = torch.tensor([0, T * H * D], **i32)
                self.kv = torch.tensor([0, S * D], **i32)
            else:
                self.qo = torch.tensor([0, T], **i32)
                self.kv = torch.tensor([0, S], **i32)
            self.ws = torch.zeros(512 << 20, dtype=torch.uint8, device=device)
            self.w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                self.ws, "NHD", use_cuda_graph=True, qo_indptr_buf=self.qo,
                kv_indptr_buf=self.kv, backend=cfg["backend"])
            self.w.plan(self.qo, self.kv, H, geo.kv_heads, D, causal=False,
                        sm_scale=self.scale, q_data_type=torch.bfloat16,
                        kv_data_type=torch.bfloat16)

    def __call__(self, q, k, v, out, pdl: bool) -> None:
        kind = self.cfg["kind"]
        if kind == "fi":
            self.w.run(q, k, v, out=out, enable_pdl=bool(pdl))
        elif kind == "triton":
            c = self.cfg
            K.attention(q, k, v, self.mask, out, self.scale, self.geo.prefix, bm=c["bm"],
                        bn=c["bn"], warps=c["warps"], stages=c["stages"], pdl=pdl)
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION,
                       "efficient": SDPBackend.EFFICIENT_ATTENTION}[self.cfg["backend"]]
            with sdpa_kernel(backend):
                o = F.scaled_dot_product_attention(
                    q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                    scale=self.scale, enable_gqa=True)
            out.copy_(o[0].transpose(0, 1))

    @property
    def label(self) -> str:
        c = self.cfg
        if c["kind"] == "triton":
            return f"triton:bm{c['bm']}_bn{c['bn']}_w{c['warps']}_s{c['stages']}"
        return f"{c['kind']}:{c['backend']}"


def _attn_candidates() -> list[dict]:
    cands = [{"kind": "fi", "backend": b} for b in _ATTN_FI]
    cands += [{"kind": "sdpa", "backend": b} for b in _ATTN_SDPA]
    for bm in (16, 32, 64):
        for bn in (32, 64):
            for warps in (4, 8):
                for stages in (1, 2):
                    cands.append({"kind": "triton", "bm": bm, "bn": bn, "warps": warps,
                                  "stages": stages})
    return cands


def autotune_attention(geo: Geometry, device) -> tuple[list, dict]:
    """All valid attention candidates, ranked over 18 distinct layers' K/V."""
    T, S, H, D, L = geo.horizon, geo.prefix + geo.horizon, geo.q_heads, geo.head_dim, geo.layers
    g = torch.Generator(device=device).manual_seed(11)
    q = torch.randn(T, H, D, generator=g, device=device).to(torch.bfloat16)
    k = torch.randn(L, S, 1, D, generator=g, device=device).to(torch.bfloat16)
    v = torch.randn(L, S, 1, D, generator=g, device=device).to(torch.bfloat16)
    mask = torch.ones(geo.prefix, dtype=torch.bool, device=device)
    s = torch.einsum("thd,sd->hts", q.float(), k[0, :, 0].float()) / math.sqrt(D)
    want = torch.einsum("hts,sd->thd", s.softmax(-1), v[0, :, 0].float())
    out = torch.empty(T, H, D, dtype=torch.bfloat16, device=device)
    record: dict[str, Any] = {}
    ranked: list[tuple[float, Attention]] = []
    for cfg in _attn_candidates():
        attn = None
        try:
            attn = Attention(cfg, geo, device, mask)
            attn(q, k[0], v[0], out, True)
            torch.cuda.synchronize()
            rel = _rel(out, want)
            if not rel <= _TOL:
                record[attn.label] = f"invalid: rel_l2={rel:.2e}"
                continue

            def many():
                for l in range(L):
                    attn(q, k[l], v[l], out, True)
            ms = _time_graph(many) / L
            record[attn.label] = {"us_per_call": ms * 1e3, "rel_l2": rel,
                                  "pdl": attn.supports_pdl}
            ranked.append((ms, attn))
        except Exception as e:  # noqa: BLE001
            record[attn.label if attn else f"{cfg}"] = _err(e)
    if not ranked:
        raise RuntimeError(f"no attention backend: {record}")
    ranked.sort(key=lambda t: t[0])
    return [a for _, a in ranked], record


# --------------------------------------------------------------------------- arm
class SotaGraphArm:
    name = "sota-graph"

    def __init__(self, geo: Geometry, inputs: dict, device, tuned: dict | None = None) -> None:
        self.geo, self.inputs, self.device = geo, inputs, torch.device(device)
        L, W, T, A = geo.layers, geo.hidden, geo.horizon, geo.action_dim
        H, D, I, S = geo.q_heads, geo.head_dim, geo.intermediate, geo.steps
        f32, bf16 = dict(dtype=torch.float32, device=device), dict(dtype=torch.bfloat16, device=device)
        self.w_qkv = torch.empty(L, (H + 2 * geo.kv_heads) * D, W, **bf16)
        self.w_gu = torch.empty(L, 2 * I, W, **bf16)
        self.refresh_weights()
        # Schedule constants (input-independent): times and their sin-cos embeddings.
        dt = -1.0 / S
        times = torch.tensor([1.0 + s * dt for s in range(S)], dtype=torch.float32, device=device)
        self.temb = ref.time_embedding(times, W).contiguous()      # [S, W] fp32
        self.dt = dt
        self.cos, self.sin = K.rope_tables(geo.prefix + T + 1, D, ref.ROPE_THETA, device)
        # Static activations.
        self.cond1, self.cond = torch.empty(S, W, **f32), torch.empty(S, W, **f32)
        self.mod_attn = torch.empty(L, S, 3 * W, **f32)
        self.mod_ffn = torch.empty(L, S, 3 * W, **f32)
        self.mod_fin = torch.empty(S, 3 * W, **f32)
        self.x = torch.empty(T, A, **f32)
        self.h, self.nrm, self.hs = (torch.empty(T, W, **bf16) for _ in range(3))
        self.qkv = torch.empty(T, (H + 2 * geo.kv_heads) * D, **bf16)
        self.q = torch.empty(T, H, D, **bf16)
        self.kbuf = torch.empty(L, geo.prefix + T, D, **bf16)
        self.vbuf = torch.empty(L, geo.prefix + T, D, **bf16)
        self.attn = torch.empty(T, H, D, **bf16)
        self.o = torch.empty(T, W, **bf16)
        self.gu = torch.empty(T, 2 * I, **bf16)
        self.act = torch.empty(T, I, **bf16)
        self.pos = torch.zeros((), dtype=torch.int32, device=device)
        # Backends (autotuned once per geometry, shared by variants).
        if tuned is None:
            tuned = self.autotune()
        self.tuned = tuned
        self._graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self.variant = "graph-pdl"

    def refresh_weights(self) -> None:
        i = self.inputs
        torch.cat((i["wq"], i["wk"], i["wv"]), 1, out=self.w_qkv)
        torch.cat((i["wg"], i["wu"]), 1, out=self.w_gu)

    def autotune(self) -> dict:
        """Per-op microbenchmarks rank the candidates; then, one op at a time, the top
        ``E2E_TOP`` candidates are swapped into the full graph-pdl chunk and the fastest
        whole chunk wins (PDL overlap and L2 interplay only show up end to end)."""
        i, T = self.inputs, self.geo.horizon
        ranked: dict[str, list] = {}
        record: dict[str, Any] = {}
        for tag, ws in (("qkv", list(self.w_qkv)), ("o", list(i["wo"])),
                        ("gu", list(self.w_gu)), ("down", list(i["wd"]))):
            ranked[tag], record[f"gemm_{tag}"] = autotune_gemm(tag, ws, T, self.device)
        ranked["attn"], record["attention"] = autotune_attention(self.geo, self.device)
        for c in ranked["attn"]:
            c.mask = i["prefix_mask"].view(torch.uint8)
        self.tuned = {"gemm": {t: ranked[t][0] for t in ("qkv", "o", "gu", "down")},
                      "attn": ranked["attn"][0], "record": record}
        e2e: dict[str, Any] = {}
        for tag in ("attn", "qkv", "o", "gu", "down"):
            trials = {}
            best = None
            for cand in ranked[tag][:E2E_TOP]:
                self._set_op(tag, cand)
                try:
                    ms = _time_graph(lambda: self._chunk(True), reps=10)
                except Exception as e:  # noqa: BLE001
                    trials[self._op_label(cand)] = _err(e)
                    continue
                trials[self._op_label(cand)] = ms
                if best is None or ms < best[0]:
                    best = (ms, cand)
            if best is None:
                best = (float("nan"), ranked[tag][0])
            self._set_op(tag, best[1])
            e2e[tag] = {"trials_ms": trials, "chosen": self._op_label(best[1])}
        record["e2e_refine"] = e2e
        return self.tuned

    def _set_op(self, tag: str, cand) -> None:
        if tag == "attn":
            self.tuned["attn"] = cand
        else:
            self.tuned["gemm"][tag] = cand

    @staticmethod
    def _op_label(cand) -> str:
        return cand.label if isinstance(cand, Attention) else cand.backend

    # -- one chunk -----------------------------------------------------------
    def _chunk(self, pdl: bool) -> None:
        geo, i, gm = self.geo, self.inputs, self.tuned["gemm"]
        L, S, P = geo.layers, geo.steps, geo.prefix
        attn = self.tuned["attn"]
        # Prologue: time MLP for all steps, then every adaRMS modulation (FP32).
        torch.addmm(i["time_in_bias"], self.temb, i["time_in_weight"].t(), out=self.cond1)
        F.silu(self.cond1, inplace=True)
        torch.addmm(i["time_out_bias"], self.cond1, i["time_out_weight"].t(), out=self.cond)
        F.silu(self.cond, inplace=True)
        cond = self.cond.expand(L, S, geo.hidden)
        torch.baddbmm(i["attn_norm_bias"][:, None], cond, i["attn_norm_weight"].transpose(1, 2),
                      out=self.mod_attn)
        torch.baddbmm(i["ffn_norm_bias"][:, None], cond, i["ffn_norm_weight"].transpose(1, 2),
                      out=self.mod_ffn)
        torch.addmm(i["final_norm_bias"], self.cond, i["final_norm_weight"].t(), out=self.mod_fin)
        self.kbuf[:, :P].copy_(i["prefix_k"].view(L, P, -1))
        self.vbuf[:, :P].copy_(i["prefix_v"].view(L, P, -1))
        torch.sum(i["prefix_mask"], 0, dtype=torch.int32, out=self.pos)
        self.x.copy_(i["noise"])
        K.boundary(self.h, None, None, None, self.hs, i["action_out_weight"], i["action_out_bias"],
                   self.x, self.dt, i["action_in_weight"], i["action_in_bias"],
                   self.mod_attn[0, 0], self.nrm, tail=False, head=True, pdl=pdl)
        from flashinfer.activation import gelu_tanh_and_mul as _fi_act
        q2, attn2 = self.q.view(geo.horizon, -1), self.attn.view(geo.horizon, -1)
        k3, v3 = self.kbuf.unsqueeze(2), self.vbuf.unsqueeze(2)
        for s in range(S):
            for l in range(L):
                gm["qkv"](self.nrm, self.w_qkv[l], self.qkv, pdl)
                K.rope_kv(self.qkv, self.q, self.kbuf[l], self.vbuf[l], self.cos, self.sin,
                          self.pos, P, geo.q_heads, geo.head_dim, pdl=pdl)
                attn(self.q, k3[l], v3[l], self.attn, pdl)
                gm["o"](attn2, i["wo"][l], self.o, pdl)
                K.resid_adarms(self.h, self.o, self.mod_attn[l, s], self.mod_ffn[l, s], self.nrm,
                               pdl=pdl)
                gm["gu"](self.nrm, self.w_gu[l], self.gu, pdl)
                _fi_act(self.gu, out=self.act, enable_pdl=bool(pdl))
                gm["down"](self.act, i["wd"][l], self.o, pdl)
                if l + 1 < L:
                    K.resid_adarms(self.h, self.o, self.mod_ffn[l, s], self.mod_attn[l + 1, s],
                                   self.nrm, pdl=pdl)
                else:
                    last = s + 1 == S
                    K.boundary(self.h, self.o, self.mod_ffn[l, s], self.mod_fin[s], self.hs,
                               i["action_out_weight"], i["action_out_bias"], self.x, self.dt,
                               i["action_in_weight"], i["action_in_bias"],
                               None if last else self.mod_attn[0, s + 1], self.nrm,
                               tail=True, head=not last, pdl=pdl)

    # -- protocol ------------------------------------------------------------
    def set_variant(self, variant: str) -> None:
        graph, pdl = _FACTORS[variant]
        self.variant = variant
        if graph and variant not in self._graphs:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._chunk(pdl)
                self._chunk(pdl)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                self._chunk(pdl)
            torch.cuda.synchronize()
            self._graphs[variant] = g

    def launch(self) -> None:
        graph, pdl = _FACTORS[self.variant]
        if graph:
            self._graphs[self.variant].replay()
        else:
            self._chunk(pdl)

    def outputs(self) -> dict:
        return {"actions": self.x}

    def config(self) -> dict:
        rec = self.tuned["record"]
        return {"gemm_backends": {t: g.backend for t, g in self.tuned["gemm"].items()},
                "attention_backend": self.tuned["attn"].label,
                "autotune": rec, "kernels_per_layer": 9}

    def close(self) -> None:
        self._graphs.clear()
