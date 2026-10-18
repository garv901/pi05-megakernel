"""CUDA-parity scalar math for the CuTe port. Each primitive has candidate lowerings (probes/math_parity.py ranks them
against nvcc -O3); `CHOICE` holds the pick per primitive, defaults = best guess until the GPU run.

Rules mirrored from pi05_mk.cu: nvcc contracts `a*b + c` into FMA unless __fadd_rn/__fmul_rn are used; expf/tanhf/sinf/
cosf/powf are the precise libm versions; `/` is IEEE; rsqrtf is the approximate instruction. Call these inside traced
device code; `use()` must run before tracing."""

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as M
from cutlass import Float32, Float64, Int32, Uint32, BFloat16

from .config import HD, NSTEP, W

RN = M.RoundingMode.NEAREST_EVEN

CANDIDATES = {
    "rbf": ("cvt", "int"),
    "pack": ("cvt", "pair"),
    "fadd": ("rn", "plain"),
    "fmul": ("rn", "plain"),
    "fma": ("fma", "rn", "plain"),
    "contract": ("plain", "fma", "split"),
    "expf": ("libm", "ptx", "ex2", "fast"),
    "tanhf": ("libm", "ptx", "approx"),
    "sinf": ("libm", "approx", "fast"),
    "cosf": ("libm", "approx", "fast"),
    "powf": ("libm", "exp2log2"),
    "rsqrtf": ("dsl", "approx_ftz", "sqrt_div"),
    "div": ("ieee", "full", "approx"),
    "sin64": ("libm",),
    "cos64": ("libm",),
    "pow64": ("libm",),
}
CHOICE = {k: v[0] for k, v in CANDIDATES.items()}


def use(choice: dict) -> None:
    """Select lowerings for code traced afterwards."""
    for k, v in choice.items():
        if v not in CANDIDATES[k]:
            raise KeyError(f"{k}: {v}")
        CHOICE[k] = v


def reset() -> None:
    CHOICE.update({k: v[0] for k, v in CANDIDATES.items()})


def _ptx(code, *args, out=Float32):
    return cute.arch.inline_ptx(code, write_only_types=[out], read_only_args=list(args))


def f32(x):
    return Float32(x)


def bits(x):
    return _ptx("mov.b32 {$w0}, {$r0};", x, out=Uint32)


def from_bits(u):
    return _ptx("mov.b32 {$w0}, {$r0};", u)


# ------------------------------------------------------------------ rounding / packing / basic ops
def rbf(x):
    """(float)__float2bfloat16_rn(x)"""
    if CHOICE["rbf"] == "int":
        u = bits(x)
        return from_bits((u + Uint32(0x7FFF) + ((u >> Uint32(16)) & Uint32(1))) & Uint32(0xFFFF0000))
    return x.to(BFloat16).to(Float32)


def pack_bf16x2(a, b):
    """__floats2bfloat162_rn(a, b) as u32 (a in the low half)."""
    if CHOICE["pack"] == "pair":
        return _ptx("{ .reg .b16 l, h; cvt.rn.bf16.f32 l, {$r0}; cvt.rn.bf16.f32 h, {$r1}; mov.b32 {$w0}, {l, h}; }",
                    a, b, out=Uint32)
    return _ptx("cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};", a, b, out=Uint32)


def fadd_rn(a, b):
    return M.add(a, b, rounding=RN) if CHOICE["fadd"] == "rn" else a + b


def fmul_rn(a, b):
    return M.mul(a, b, rounding=RN) if CHOICE["fmul"] == "rn" else a * b


def fma(a, b, c):
    """fmaf(a, b, c)"""
    sel = CHOICE["fma"]
    if sel == "plain":
        return a * b + c
    return M.fma(a, b, c, rounding=RN) if sel == "rn" else M.fma(a, b, c)


def mac(a, b, c):
    """a * b + c as nvcc compiles it when the source has no __fadd_rn/__fmul_rn: one FMA."""
    sel = CHOICE["contract"]
    if sel == "fma":
        return M.fma(a, b, c)
    if sel == "split":
        return M.add(M.mul(a, b, rounding=RN), c, rounding=RN)
    return a * b + c


def div(a, b):
    sel = CHOICE["div"]
    if sel == "full":
        return M.div(a, b, full=True)
    return M.div(a, b, approx=True) if sel == "approx" else a / b


# ------------------------------------------------------------------ transcendentals
def expf(x):
    sel = CHOICE["expf"]
    if sel == "fast":
        return M.exp(x, fastmath=True)
    if sel == "ex2":
        return M.exp2(x * f32(1.4426950408889634), approx=True, ftz=True)
    if sel == "ptx":  # nvcc's expansion of expf (non-ftz): 2^(x*log2e) split into an integer and a fraction part
        return _ptx(
            "{ .reg .f32 a, b, c, d, e, f, g; .reg .b32 s;"
            " fma.rn.f32 a, {$r0}, 0f3BBB989D, 0f3F000000; cvt.sat.f32.f32 b, a;"
            " mov.b32 c, 0f4B400001; mov.b32 d, 0f437C0000; fma.rm.f32 e, b, d, c;"
            " add.f32 f, e, 0fCB40007F; neg.f32 f, f; fma.rn.f32 g, {$r0}, 0f3FB8AA3B, f;"
            " fma.rn.f32 g, {$r0}, 0f32A57060, g; shl.b32 s, e, 23; ex2.approx.ftz.f32 f, g; mul.f32 {$w0}, f, s; }", x)
    return M.exp(x)


def tanhf(x):
    sel = CHOICE["tanhf"]
    if sel == "approx":
        return _ptx("tanh.approx.f32 {$w0}, {$r0};", x)
    if sel == "ptx":  # nvcc's expansion of tanhf
        return _ptx(
            "{ .reg .f32 a, b, c, d, e, f, g, h, s, q; .reg .pred p, p2;"
            " abs.f32 a, {$r0}; mul.f32 b, a, 0f4038AA3B; ex2.approx.ftz.f32 c, b; add.f32 d, c, 0f3F800000;"
            " rcp.approx.ftz.f32 e, d; fma.rn.f32 f, e, 0fC0000000, 0f3F800000; setp.ge.f32 p, a, 0f41102CB4;"
            " selp.f32 g, 0f3F800000, f, p; copysign.f32 h, {$r0}, g; mul.f32 s, {$r0}, {$r0};"
            " fma.rn.f32 q, s, 0f3C80F082, 0fBD563CAE; fma.rn.f32 q, q, s, 0f3E085941;"
            " fma.rn.f32 q, q, s, 0fBEAAA9ED; fma.rn.f32 q, q, s, 0f00000000; fma.rn.f32 q, q, {$r0}, {$r0};"
            " setp.ge.f32 p2, a, 0f3F19999A; selp.f32 {$w0}, h, q, p2; }", x)
    return M.tanh(x)


def sinf(x):
    sel = CHOICE["sinf"]
    if sel == "approx":
        return _ptx("sin.approx.ftz.f32 {$w0}, {$r0};", x)
    return M.sin(x, fastmath=sel == "fast")


def cosf(x):
    sel = CHOICE["cosf"]
    if sel == "approx":
        return _ptx("cos.approx.ftz.f32 {$w0}, {$r0};", x)
    return M.cos(x, fastmath=sel == "fast")


def powf(x, y):
    if CHOICE["powf"] == "exp2log2":
        return M.exp2(y * M.log2(x, approx=True, ftz=True), approx=True, ftz=True)
    return M.pow(x, y)


def rsqrtf(x):
    """rsqrtf: rsqrt.approx.f32 (the DSL's default lowering emits exactly this)."""
    sel = CHOICE["rsqrtf"]
    if sel == "approx_ftz":
        return M.rsqrt(x, approx=True, ftz=True)
    if sel == "sqrt_div":
        return div(f32(1.0), M.sqrt(x))
    return M.rsqrt(x)


# ------------------------------------------------------------------ expressions of the kernel
def silu(x):
    return div(x, f32(1.0) + expf(-x))


def gelu_tanh(x):
    x3 = x * x * x
    inner = f32(0.7978845608028654) * mac(f32(0.044715), x3, x)
    return f32(0.5) * x * (f32(1.0) + tanhf(inner))


def gelu_mul(g, u):
    """G epilogue: rbf(gelu_tanh(rbf(g))) * rbf(u), before the bf16 pack."""
    return rbf(gelu_tanh(rbf(g))) * rbf(u)


def ada_rs(ss):
    """rsqrtf(__fadd_rn(ss * (1.0f / W), 1e-6f))"""
    return rsqrtf(fadd_rn(ss * f32(1.0 / W), f32(1e-6)))


def ada_apply(h, rs, sc, sh):
    """__fadd_rn(__fmul_rn(__fmul_rn(h, rs), __fadd_rn(1, sc)), sh)"""
    return fadd_rn(fmul_rn(fmul_rn(h, rs), fadd_rn(f32(1.0), sc)), sh)


def residual(h, o, g):
    """rbf(__fadd_rn(h, rbf(__fmul_rn(rbf(o), rbf(g)))))"""
    return rbf(fadd_rn(h, rbf(fmul_rn(rbf(o), rbf(g)))))


def attn_score(s, keep):
    """keep ? rbf(s) * 0.0625f : -inf"""
    return Float32(cutlass.select_(keep, rbf(s) * f32(0.0625), f32(-float("inf"))))


def attn_p(s, m):
    """rbf(expf(s - m)) with a fully masked row (m = -inf) subtracting 0"""
    ms = Float32(cutlass.select_(m == f32(-float("inf")), f32(0.0), m))
    return rbf(expf(s - ms))


def comb_w(w, mx):
    """w == -inf ? 0 : expf(w - M)"""
    return Float32(cutlass.select_(w == f32(-float("inf")), f32(0.0), expf(w - mx)))


def rope_inv(j):
    """1.0f / powf(10000.0f, (float)(2 * j) / (float)HD), j: Int32"""
    return div(f32(1.0), powf(f32(10000.0), div(Float32(j * 2), f32(float(HD)))))


def rope_cs(pos, inv):
    """(rbf(cosf(ph)), rbf(sinf(ph))) with ph = __fmul_rn((float)pos, inv)"""
    ph = fmul_rn(Float32(pos), inv)
    return rbf(cosf(ph)), rbf(sinf(ph))


def rope_apply(x1, x2, c, s):
    return rbf(rbf(x1 * c) + rbf(-x2 * s)), rbf(rbf(x2 * c) + rbf(x1 * s))


def _mac64(a, b, c):
    sel = CHOICE["contract"]
    if sel == "split":
        return M.add(M.mul(a, b, rounding=RN), c, rounding=RN)
    return M.fma(a, b, c) if sel == "fma" else a * b + c


def temb(s, j):
    """Time embedding of step s, column j (FP64 -> FP32): ((float)sin(ph), (float)cos(ph)), s / j: Int32."""
    d = Float64
    tf = _mac64(d(s), d(-1.0 / NSTEP), d(1.0)).to(Float32)
    steps = W // 2
    step = d(1.0 / float(steps - 1))
    low = step * d(j)
    high = _mac64(-step, d(steps - j - 1), d(1.0))
    frac = Float64(cutlass.select_(j < Int32(steps // 2), low, high))
    period = d(4e-3) * (M.pow(d(4.0 / 4e-3), frac))
    freq = (d(1.0) / period) * d(2.0) * d(3.141592653589793)
    ph = tf.to(Float64) * freq
    return M.sin(ph).to(Float32), M.cos(ph).to(Float32)
