"""torch.compile baseline: Inductor max-autotune of one Euler step, the 10-step chunk
captured by hand in ONE CUDA graph (Inductor never sets PDL).

The step is the reference math with the same build-time weight concatenation and
hoisted modulation as the SOTA graph arm, and SDPA attention (Inductor lowers it to
the ATen flash/cuDNN kernel). ``mode="max-autotune-no-cudagraphs"`` autotunes GEMM
templates (Triton vs ATen/cuBLAS) and fuses elementwise ops; our own capture removes
the per-step host overhead that cudagraph trees would add.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .fixture import Geometry
from .reference import pi05 as ref


def _step(x, mod_a, mod_f, mod_fin, kpre, vpre, pos, w_qkv, wo, w_gu, wd,
          win, bin_, wout, bout, dt: float, H: int, D: int):
    T = x.shape[0]
    h = F.linear(x, win, bin_).to(torch.bfloat16)
    for l in range(w_qkv.shape[0]):
        nrm, gate = ref.ada_rms(h, mod_a[l])
        qkv = nrm @ w_qkv[l].T
        q, k, v = qkv.split((H * D, D, D), -1)
        q = ref.rope(q.reshape(T, H, D), pos)
        k = ref.rope(k.reshape(T, 1, D), pos)
        key = torch.cat((kpre[l], k))
        val = torch.cat((vpre[l], v.reshape(T, 1, D)))
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], key.transpose(0, 1)[None],
                                           val.transpose(0, 1)[None], scale=1.0 / math.sqrt(D),
                                           enable_gqa=True)
        h = h + (o[0].transpose(0, 1).reshape(T, H * D) @ wo[l].T) * gate
        nrm, gate = ref.ada_rms(h, mod_f[l])
        g, u = (nrm @ w_gu[l].T).chunk(2, -1)
        h = h + ((F.gelu(g, approximate="tanh") * u) @ wd[l].T) * gate
    y = ref.ada_rms(h, mod_fin)[0]
    return x + dt * F.linear(y.float(), wout, bout)


class CompileArm:
    name = "torch-compile"

    def __init__(self, geo: Geometry, inputs: dict, device) -> None:
        self.geo, self.inputs = geo, inputs
        L, W, S, T = geo.layers, geo.hidden, geo.steps, geo.horizon
        f32 = dict(dtype=torch.float32, device=device)
        bf16 = dict(dtype=torch.bfloat16, device=device)
        self.w_qkv = torch.empty(L, (geo.q_heads + 2 * geo.kv_heads) * geo.head_dim, W, **bf16)
        self.w_gu = torch.empty(L, 2 * geo.intermediate, W, **bf16)
        self.refresh_weights()
        self.dt = -1.0 / S
        times = torch.tensor([1.0 + s * self.dt for s in range(S)], **f32)
        self.temb = ref.time_embedding(times, W).contiguous()
        self.cond1, self.cond = torch.empty(S, W, **f32), torch.empty(S, W, **f32)
        self.mod_attn = torch.empty(L, S, 3 * W, **f32)
        self.mod_ffn = torch.empty(L, S, 3 * W, **f32)
        self.mod_fin = torch.empty(S, 3 * W, **f32)
        self.pos = torch.empty(T, dtype=torch.int64, device=device)
        self.ar = torch.arange(T, device=device)
        self.x = torch.empty(T, geo.action_dim, **f32)
        self.step = torch.compile(_step, fullgraph=True, dynamic=False,
                                  mode="max-autotune-no-cudagraphs")
        self.graph: torch.cuda.CUDAGraph | None = None
        self.variant = "max-autotune"

    def refresh_weights(self) -> None:
        i = self.inputs
        torch.cat((i["wq"], i["wk"], i["wv"]), 1, out=self.w_qkv)
        torch.cat((i["wg"], i["wu"]), 1, out=self.w_gu)

    def _chunk(self) -> None:
        i, geo = self.inputs, self.geo
        torch.addmm(i["time_in_bias"], self.temb, i["time_in_weight"].t(), out=self.cond1)
        F.silu(self.cond1, inplace=True)
        torch.addmm(i["time_out_bias"], self.cond1, i["time_out_weight"].t(), out=self.cond)
        F.silu(self.cond, inplace=True)
        cond = self.cond.expand(geo.layers, geo.steps, geo.hidden)
        torch.baddbmm(i["attn_norm_bias"][:, None], cond, i["attn_norm_weight"].transpose(1, 2),
                      out=self.mod_attn)
        torch.baddbmm(i["ffn_norm_bias"][:, None], cond, i["ffn_norm_weight"].transpose(1, 2),
                      out=self.mod_ffn)
        torch.addmm(i["final_norm_bias"], self.cond, i["final_norm_weight"].t(), out=self.mod_fin)
        torch.add(self.ar, i["prefix_mask"].sum(), out=self.pos)
        x = i["noise"]
        for s in range(geo.steps):
            x = self.step(x, self.mod_attn[:, s], self.mod_ffn[:, s], self.mod_fin[s],
                          i["prefix_k"], i["prefix_v"], self.pos, self.w_qkv, i["wo"], self.w_gu,
                          i["wd"], i["action_in_weight"], i["action_in_bias"],
                          i["action_out_weight"], i["action_out_bias"], self.dt,
                          geo.q_heads, geo.head_dim)
        self.x.copy_(x)

    def set_variant(self, variant: str) -> None:
        if self.graph is not None:
            return
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._chunk()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            self._chunk()
        torch.cuda.synchronize()
        self.graph = g

    def launch(self) -> None:
        self.graph.replay()

    def outputs(self) -> dict:
        return {"actions": self.x}

    def config(self) -> dict:
        return {"mode": "max-autotune-no-cudagraphs + manual whole-chunk CUDA graph",
                "pdl": False}

    def close(self) -> None:
        self.graph = None
