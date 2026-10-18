"""Fused Triton glue kernels for the pi0.5 SOTA graph, all PDL-capable.

There is no library kernel for adaRMS or the pi0.5 step boundary, so these play the
role vLLM's fused norm / rope / cache kernels play for an LLM. Each kernel calls
``gdc_wait`` before touching memory (safe when PDL is off) and then
``gdc_launch_dependents``, so with ``launch_pdl=True`` the next kernel can become
resident while this one runs but never reads before it finishes.

BF16 rounding points follow ``reference/pi05.py`` op by op:
  * gated residual: ``h + (y * gate)`` with both products rounded to BF16;
  * adaRMS: FP32 normalize / scale / shift of the BF16 residual, BF16 out;
  * RoPE: BF16 cos/sin tables, each product and the sum rounded to BF16;
  * action in/out projections and the Euler update in FP32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import cuda as tlc

EPS = 1e-6


@triton.jit
def _pdl_enter(PDL: tl.constexpr):
    if PDL:
        tlc.gdc_wait()
        tlc.gdc_launch_dependents()


@triton.jit
def _resid_adarms_kernel(h_ptr, d_ptr, gmod_ptr, mod_ptr, nrm_ptr, eps,
                         W: tl.constexpr, HAS_RES: tl.constexpr, PDL: tl.constexpr):
    """[h += bf16(d * gate)]; nrm = bf16(rms(h) * (1 + scale) + shift)."""
    _pdl_enter(PDL)
    row = tl.program_id(0)
    cols = tl.arange(0, W)
    h = tl.load(h_ptr + row * W + cols)
    if HAS_RES:
        d = tl.load(d_ptr + row * W + cols).to(tl.float32)
        g = tl.load(gmod_ptr + 2 * W + cols).to(tl.bfloat16).to(tl.float32)
        t = (d * g).to(tl.bfloat16).to(tl.float32)
        h = (h.to(tl.float32) + t).to(tl.bfloat16)
        tl.store(h_ptr + row * W + cols, h)
    hf = h.to(tl.float32)
    r = 1.0 / tl.sqrt_rn(tl.sum(hf * hf, 0) / W + eps)
    scale = tl.load(mod_ptr + cols)
    shift = tl.load(mod_ptr + W + cols)
    y = (hf * r) * (1.0 + scale) + shift
    tl.store(nrm_ptr + row * W + cols, y.to(tl.bfloat16))


@triton.jit
def _rope_kv_kernel(qkv_ptr, q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, pos_ptr,
                    P: tl.constexpr, D: tl.constexpr, NQ: tl.constexpr, PDL: tl.constexpr):
    """Rotate q/k (half-split Gemma RoPE), write q, and append k/v at rows P + t."""
    _pdl_enter(PDL)
    t = tl.program_id(0)
    hd = tl.program_id(1)            # 0..NQ-1 query heads, NQ key, NQ+1 value
    half: tl.constexpr = D // 2
    i = tl.arange(0, half)
    base = qkv_ptr + t * (NQ + 2) * D + hd * D
    x1 = tl.load(base + i)
    x2 = tl.load(base + half + i)
    if hd <= NQ:
        pos = tl.load(pos_ptr).to(tl.int32) + t
        c = tl.load(cos_ptr + pos * half + i).to(tl.float32)
        s = tl.load(sin_ptr + pos * half + i).to(tl.float32)
        f1, f2 = x1.to(tl.float32), x2.to(tl.float32)
        o1 = ((f1 * c).to(tl.bfloat16).to(tl.float32)
              + (-f2 * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        o2 = ((f2 * c).to(tl.bfloat16).to(tl.float32)
              + (f1 * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        if hd < NQ:
            dst = q_ptr + (t * NQ + hd) * D
        else:
            dst = k_ptr + (P + t) * D
        tl.store(dst + i, o1)
        tl.store(dst + half + i, o2)
    else:
        dst = v_ptr + (P + t) * D
        tl.store(dst + i, x1)
        tl.store(dst + half + i, x2)


@triton.jit
def _boundary_kernel(h_ptr, d_ptr, gmod_ptr, fmod_ptr, hs_ptr, wout_ptr, bout_ptr, x_ptr, dt,
                     win_ptr, bin_ptr, nmod_ptr, nrm_ptr, eps,
                     W: tl.constexpr, A: tl.constexpr, KB: tl.constexpr, CB: tl.constexpr,
                     TAIL: tl.constexpr, HEAD: tl.constexpr, PDL: tl.constexpr):
    """Step boundary, one program per action token (everything here is row-local).

    TAIL: h += bf16(d * gate_ffn); y = final adaRMS(h); x += dt * (y @ Wout^T + b).
    HEAD: h = bf16(x @ Win^T + b); nrm = layer-0 attention adaRMS(h).
    """
    _pdl_enter(PDL)
    row = tl.program_id(0)
    cols = tl.arange(0, W)
    a = tl.arange(0, A)
    if TAIL:
        h = tl.load(h_ptr + row * W + cols).to(tl.float32)
        d = tl.load(d_ptr + row * W + cols).to(tl.float32)
        g = tl.load(gmod_ptr + 2 * W + cols).to(tl.bfloat16).to(tl.float32)
        hb = (h + (d * g).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        hf = hb.to(tl.float32)
        r = 1.0 / tl.sqrt_rn(tl.sum(hf * hf, 0) / W + eps)
        y = ((hf * r) * (1.0 + tl.load(fmod_ptr + cols)) + tl.load(fmod_ptr + W + cols))
        tl.store(hs_ptr + row * W + cols, y.to(tl.bfloat16))
        tl.debug_barrier()
        acc = tl.zeros([A], dtype=tl.float32)
        for k0 in tl.static_range(0, W, KB):
            kk = k0 + tl.arange(0, KB)
            yk = tl.load(hs_ptr + row * W + kk).to(tl.float32)
            wk = tl.load(wout_ptr + a[:, None] * W + kk[None, :])
            acc += tl.sum(wk * yk[None, :], 1)
        v = acc + tl.load(bout_ptr + a)
        x = tl.load(x_ptr + row * A + a)
        tl.store(x_ptr + row * A + a, x + dt * v)
        tl.debug_barrier()
    if HEAD:
        x = tl.load(x_ptr + row * A + a)
        ss = 0.0
        for c0 in tl.static_range(0, W, CB):
            cc = c0 + tl.arange(0, CB)
            wi = tl.load(win_ptr + cc[:, None] * A + a[None, :])
            hc = (tl.sum(wi * x[None, :], 1) + tl.load(bin_ptr + cc)).to(tl.bfloat16)
            tl.store(h_ptr + row * W + cc, hc)
            hcf = hc.to(tl.float32)
            ss += tl.sum(hcf * hcf, 0)
        tl.debug_barrier()
        r = 1.0 / tl.sqrt_rn(ss / W + eps)
        hf = tl.load(h_ptr + row * W + cols).to(tl.float32)
        y = (hf * r) * (1.0 + tl.load(nmod_ptr + cols)) + tl.load(nmod_ptr + W + cols)
        tl.store(nrm_ptr + row * W + cols, y.to(tl.bfloat16))


@triton.jit
def _attn_kernel(q_ptr, k_ptr, v_ptr, mask_ptr, o_ptr, scale,
                 ROWS: tl.constexpr, NK: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, PDL: tl.constexpr):
    """Single-KV-head attention: every (token, head) query row sees the same K/V, so the
    T x H queries are one [ROWS, D] block over [NK, D] keys (prefix rows masked by
    ``mask``, suffix rows always visible). Online softmax in FP32, BF16 P @ V."""
    _pdl_enter(PDL)
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, D)
    rmask = rows < ROWS
    q = tl.load(q_ptr + rows[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
    m = tl.full([BM], float("-inf"), tl.float32)
    l = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for j0 in tl.static_range(0, NK, BN):
        j = j0 + tl.arange(0, BN)
        inb = j < NK
        valid = inb
        if j0 < P:
            pm = tl.load(mask_ptr + j, mask=j < P, other=1)
            valid = valid & ((j >= P) | (pm != 0))
        kt = tl.load(k_ptr + j[:, None] * D + d[None, :], mask=inb[:, None], other=0.0)
        s = tl.dot(q, tl.trans(kt)) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        mn = tl.maximum(m, tl.max(s, 1))
        ms = tl.where(mn == float("-inf"), 0.0, mn)
        alpha = tl.exp(m - ms)
        p = tl.exp(s - ms[:, None])
        l = l * alpha + tl.sum(p, 1)
        vt = tl.load(v_ptr + j[:, None] * D + d[None, :], mask=inb[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vt)
        m = mn
    out = acc / l[:, None]
    tl.store(o_ptr + rows[:, None] * D + d[None, :], out.to(tl.bfloat16), mask=rmask[:, None])


def attention(q, k, v, mask_u8, out, scale: float, prefix: int, *, bm: int, bn: int,
              warps: int, stages: int, pdl: bool) -> None:
    """q/out [T, H, D] (single KV head), k/v [NK, D] or [NK, 1, D], mask_u8 [prefix]."""
    rows, dim = q.shape[0] * q.shape[1], q.shape[-1]
    _attn_kernel[(triton.cdiv(rows, bm),)](q, k, v, mask_u8, out, scale, ROWS=rows,
                                           NK=k.shape[0], P=prefix, D=dim, BM=bm, BN=bn,
                                           PDL=pdl, num_warps=warps, num_stages=stages,
                                           launch_pdl=pdl)


def resid_adarms(h, d, gmod, mod, nrm, *, pdl: bool) -> None:
    rows, width = h.shape
    _resid_adarms_kernel[(rows,)](h, d if d is not None else h, gmod if gmod is not None else mod,
                                  mod, nrm, EPS, W=width, HAS_RES=d is not None, PDL=pdl,
                                  num_warps=4, launch_pdl=pdl)


def rope_kv(qkv, q, kbuf, vbuf, cos, sin, pos, prefix: int, n_q: int, head_dim: int,
            *, pdl: bool) -> None:
    rows = qkv.shape[0]
    _rope_kv_kernel[(rows, n_q + 2)](qkv, q, kbuf, vbuf, cos, sin, pos, P=prefix, D=head_dim,
                                     NQ=n_q, PDL=pdl, num_warps=1, launch_pdl=pdl)


def boundary(h, d, gmod, fmod, hs, wout, bout, x, dt, win, bin_, nmod, nrm, *,
             tail: bool, head: bool, pdl: bool) -> None:
    rows, width = h.shape
    dummy = fmod if fmod is not None else nmod
    _boundary_kernel[(rows,)](h, d if d is not None else h, gmod if gmod is not None else dummy,
                              dummy, hs, wout, bout, x, dt, win, bin_,
                              nmod if nmod is not None else dummy, nrm, EPS,
                              W=width, A=x.shape[1], KB=128, CB=256, TAIL=tail, HEAD=head,
                              PDL=pdl, num_warps=8, launch_pdl=pdl)


def rope_tables(max_pos: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [max_pos, head_dim/2] cos/sin with the reference's FP32 phases."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    phase = torch.arange(max_pos, device=device).float()[:, None] * inv.to(device)[None, :]
    return (phase.cos().to(torch.bfloat16).contiguous(),
            phase.sin().to(torch.bfloat16).contiguous())
