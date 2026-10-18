"""Phase-0 probe: which DSL lowering of each scalar expression of pi05_mk.cu is bit-identical to nvcc -O3?

CUDA side: math_ref.cu (expressions copied from the kernel). DSL side: mathx.py composites traced under several
lowering profiles. Output math.json: per expression and profile, mismatch count and max ulp.
CPU check: CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a python -m megabench.sota.pi05.cute.probes.math_parity --compile-only
GPU run: python -m megabench.sota.pi05.cute.probes.math_parity --out DIR"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from .. import mathx as X

HERE = os.path.dirname(os.path.abspath(__file__))
INF = float("inf")

PROFILES = {
    "default": {},
    "ptx": {"expf": "ptx", "tanhf": "ptx", "contract": "fma", "fma": "fma"},
    "split": {"contract": "split", "fadd": "plain", "fmul": "plain", "fma": "plain"},
    "ex2": {"expf": "ex2", "tanhf": "approx", "rbf": "int", "pack": "pair"},
    "fast": {"expf": "fast", "tanhf": "approx", "sinf": "fast", "cosf": "fast", "powf": "exp2log2",
             "rsqrtf": "approx_ftz", "div": "approx"},
}


@dataclass(frozen=True)
class Expr:
    name: str
    n_in: int
    n_out: int
    uses: tuple
    gen: Callable  # (rng, n) -> list of float32 arrays
    dsl: Callable  # list[Float32] -> list[Float32] (or u32 bit patterns for kind == "bits")
    kind: str = "float"


def _bf(a):
    return (a.astype(np.float32).view(np.uint32) + 0x7FFF + ((a.astype(np.float32).view(np.uint32) >> 16) & 1)
            & 0xFFFF0000).astype(np.uint32).view(np.float32)


def _mix(rng, n, lo=-4, hi=4):
    """Signed values with log-uniform magnitude."""
    return (rng.choice([-1.0, 1.0], n) * 10.0 ** rng.uniform(lo, hi, n)).astype(np.float32)


EDGE = np.array([0.0, -0.0, INF, -INF, 1e-40, -1e-40, 1.0, -1.0, 0.5, 65504.0, 3.38e38, -3.38e38], np.float32)


def _with_edge(arrs, edge_rows):
    """Overwrite the first rows with edge cases: edge_rows is a list of tuples (one value per input)."""
    for r, row in enumerate(edge_rows):
        for k, v in enumerate(row):
            arrs[k][r] = v
    return arrs


def g_unary(lo=-4, hi=4, edge=True):
    def gen(rng, n):
        a = _mix(rng, n, lo, hi)
        if edge:
            a[:len(EDGE)] = EDGE
        return [a]
    return gen


def g_range(lo, hi, k=1):
    def gen(rng, n):
        return [rng.uniform(lo, hi, n).astype(np.float32) for _ in range(k)]
    return gen


def g_silu(rng, n):
    a = np.where(rng.random(n) < 0.5, rng.uniform(-20, 20, n), rng.standard_normal(n) * 5).astype(np.float32)
    a[:6] = [0.0, -0.0, 88.0, -88.0, -100.0, 100.0]
    return [a]


def g_gelu_mul(rng, n):
    return [rng.uniform(-10, 10, n).astype(np.float32), rng.standard_normal(n).astype(np.float32) * 2]


def g_ada_rs(rng, n):
    a = (10.0 ** rng.uniform(-2, 5.5, n)).astype(np.float32)
    a[:3] = [0.0, 1e-30, 1e-40]
    return [a]


def g_ada_apply(rng, n):
    h = _bf(rng.standard_normal(n).astype(np.float32) * 3)
    return [h, rng.uniform(0.2, 3, n).astype(np.float32), rng.standard_normal(n).astype(np.float32) * 0.5,
            rng.standard_normal(n).astype(np.float32) * 0.5]


def g_residual(rng, n):
    return [_bf(rng.standard_normal(n).astype(np.float32) * 3), rng.standard_normal(n).astype(np.float32) * 3,
            rng.standard_normal(n).astype(np.float32)]


def g_attn_score(rng, n):
    s = (rng.standard_normal(n) * 50).astype(np.float32)
    return [s, (rng.random(n) < 0.9).astype(np.float32)]


def g_attn_p(rng, n):
    m = (rng.standard_normal(n) * 3).astype(np.float32)
    s = (m - np.abs(rng.standard_normal(n)) * rng.choice([1.0, 5.0, 20.0, 80.0], n)).astype(np.float32)
    _with_edge([s, m], [(-INF, -INF), (-INF, 0.0), (0.0, 0.0), (-1e-40, 0.0), (-103.0, 0.0), (-88.0, 0.0), (-87.5, 0.0)])
    return [s, m]


def g_comb_w(rng, n):
    w = (rng.standard_normal(n) * 10).astype(np.float32)
    mx = (w + np.abs(rng.standard_normal(n)) * rng.choice([0.1, 1.0, 5.0, 40.0], n)).astype(np.float32)
    _with_edge([w, mx], [(-INF, 1.0), (-INF, -INF), (0.0, 0.0), (-100.0, 0.0)])
    return [w, mx]


def g_comb_acc(rng, n):
    return [rng.uniform(0, 1, n).astype(np.float32), rng.standard_normal(n).astype(np.float32),
            rng.standard_normal(n).astype(np.float32), rng.uniform(0, 64, n).astype(np.float32),
            rng.uniform(0, 64, n).astype(np.float32)]


def g_div(rng, n):
    return [_mix(rng, n, -3, 3), _mix(rng, n, -3, 3)]


def g_rope_inv(rng, n):
    return [rng.integers(0, 128, n).astype(np.float32)]


def g_rope_cs(rng, n):
    return [rng.integers(0, 1200, n).astype(np.float32), rng.integers(0, 128, n).astype(np.float32)]


def g_rope_apply(rng, n):
    return [_bf(rng.standard_normal(n).astype(np.float32) * 3), _bf(rng.standard_normal(n).astype(np.float32) * 3),
            _bf(rng.uniform(-1, 1, n).astype(np.float32)), _bf(rng.uniform(-1, 1, n).astype(np.float32))]


def g_temb(rng, n):
    s, j = np.meshgrid(np.arange(10), np.arange(512), indexing="ij")
    return [s.reshape(-1).astype(np.float32), j.reshape(-1).astype(np.float32)]


def g_randn(k, scale=1.0, bf=False):
    def gen(rng, n):
        return [(_bf if bf else (lambda a: a))(rng.standard_normal(n).astype(np.float32) * scale) for _ in range(k)]
    return gen


# ---- DSL expressions (mathx composites in the C++ order of operations)
def d_rbf(i): return [X.rbf(i[0])]
def d_pack(i): return [X.pack_bf16x2(i[0], i[1])]
def d_f2d2f(i): return [i[0].to(cutlass.Float64).to(Float32)]
def d_silu(i): return [X.silu(i[0])]
def d_gelu(i): return [X.gelu_tanh(i[0])]
def d_gelu_mul(i):
    o = X.gelu_mul(i[0], i[1])
    return [o, X.rbf(o)]
def d_ada_rs(i): return [X.ada_rs(i[0])]
def d_ada_apply(i):
    o = X.ada_apply(i[0], i[1], i[2], i[3])
    return [o, X.rbf(o)]
def d_residual(i): return [X.residual(i[0], i[1], i[2])]
def d_attn_score(i): return [X.attn_score(i[0], i[1] > Float32(0.5))]
def d_attn_p(i): return [X.attn_p(i[0], i[1])]
def d_comb_w(i): return [X.comb_w(i[0], i[1])]
def d_comb_acc(i): return [X.fma(i[0], i[1], i[2]), X.fma(i[0], i[3], i[4])]
def d_div(i): return [X.div(i[0], i[1])]
def d_rope_inv(i): return [X.rope_inv(i[0].to(Int32))]
def d_rope_cs(i):
    inv = X.rope_inv(i[1].to(Int32))
    c, s = X.rope_cs(i[0].to(Int32), inv)
    return [c, s]
def d_rope_apply(i):
    a, b = X.rope_apply(i[0], i[1], i[2], i[3])
    return [a, b]
def d_temb(i):
    a, b = X.temb(i[0].to(Int32), i[1].to(Int32))
    return [a, b]
def d_cond_dot(i):
    acc = i[8]
    for k in range(4):
        acc = X.fma(i[4 + k], i[k], acc)
    return [acc]
def d_action_in(i):
    v = Float32(0.0)
    for k in range(4):
        v = X.fma(i[k], i[4 + k], v)
    return [X.rbf(v + i[8]), v]
def d_ss2(i): return [X.fma(i[1], i[1], i[0] * i[0])]
def d_action_out(i):
    v = Float32(0.0)
    for k in range(4):
        v = X.fma(i[k], i[4 + k], v)
    vel = v + i[8]
    return [X.fadd_rn(i[9], X.fmul_rn(Float32(-0.1), vel)), vel]
def d_ada_ss(i):
    ss = Float32(0.0)
    for k in range(4):
        ss = X.fma(i[k], i[k], ss)
    return [ss]


# order == case ids of math_ref.cu
EXPRS = [
    Expr("rbf", 1, 1, ("rbf",), g_unary(-6, 6), d_rbf),
    Expr("pack2", 2, 1, ("pack",), lambda r, n: [_mix(r, n, -3, 3), _mix(r, n, -3, 3)], d_pack, "bits"),
    Expr("f2d2f", 1, 1, (), g_unary(), d_f2d2f),
    Expr("silu", 1, 1, ("expf", "div"), g_silu, d_silu),
    Expr("gelu_tanh", 1, 1, ("tanhf", "contract"), g_range(-10, 10), d_gelu),
    Expr("gelu_mul", 2, 2, ("tanhf", "contract", "rbf"), g_gelu_mul, d_gelu_mul),
    Expr("ada_rs", 1, 1, ("rsqrtf", "fadd"), g_ada_rs, d_ada_rs),
    Expr("ada_apply", 4, 2, ("fadd", "fmul", "rbf"), g_ada_apply, d_ada_apply),
    Expr("residual", 3, 1, ("rbf", "fadd", "fmul"), g_residual, d_residual),
    Expr("attn_score", 2, 1, ("rbf",), g_attn_score, d_attn_score),
    Expr("attn_p", 2, 1, ("expf", "rbf"), g_attn_p, d_attn_p),
    Expr("comb_w", 2, 1, ("expf",), g_comb_w, d_comb_w),
    Expr("comb_acc", 5, 2, ("fma",), g_comb_acc, d_comb_acc),
    Expr("div", 2, 1, ("div",), g_div, d_div),
    Expr("rope_inv", 1, 1, ("powf", "div"), g_rope_inv, d_rope_inv),
    Expr("rope_cs", 2, 2, ("powf", "div", "fmul", "sinf", "cosf", "rbf"), g_rope_cs, d_rope_cs),
    Expr("rope_apply", 4, 2, ("rbf",), g_rope_apply, d_rope_apply),
    Expr("temb", 2, 2, ("contract", "sin64", "cos64", "pow64"), g_temb, d_temb),
    Expr("cond_dot", 9, 1, ("fma",), g_randn(9), d_cond_dot),
    Expr("action_in", 9, 2, ("fma", "rbf"), g_randn(9), d_action_in),
    Expr("ss2", 2, 1, ("fma",), g_randn(2, 3.0, True), d_ss2),
    Expr("action_out", 10, 2, ("fma", "fadd", "fmul"), g_randn(10), d_action_out),
    Expr("ada_ss", 4, 1, ("fma",), g_randn(4, 3.0, True), d_ada_ss),
]
BY_NAME = {e.name: e for e in EXPRS}


def profiles_for(e: Expr) -> dict[str, dict]:
    """Profiles that differ in the primitives `e` uses (first name wins); the rest would compile to the same code."""
    seen, out = {}, {}
    for name, choice in PROFILES.items():
        key = tuple(choice.get(u, X.CANDIDATES[u][0]) for u in e.uses)
        if key not in seen:
            seen[key] = name
            out[name] = choice
    return out


@cute.kernel
def eval_kernel(inp: cute.Tensor, out: cute.Tensor, n: Int32, expr: cutlass.Constexpr, profs: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * 256 + tid
    e = EXPRS[expr]
    if i < n:
        ins = []
        for k in cutlass.range_constexpr(e.n_in):
            ins.append(inp[k * n + i])
        for p in cutlass.range_constexpr(len(profs)):
            X.use(PROFILES[profs[p]])
            res = e.dsl(ins)
            for o in cutlass.range_constexpr(e.n_out):
                v = res[o]
                if cutlass.const_expr(e.kind == "bits"):
                    v = X.from_bits(v)
                out[(p * 2 + o) * n + i] = v
            X.reset()


@cute.jit
def host(inp: cute.Tensor, out: cute.Tensor, n: Int32, expr: cutlass.Constexpr, profs: cutlass.Constexpr):
    eval_kernel(inp, out, n, expr, profs).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1))


def load_ref():
    from torch.utils.cpp_extension import load
    import torch
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "{}.{}".format(*torch.cuda.get_device_capability()) \
            if torch.cuda.is_available() else "10.0"
    build = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp"), "pi05_math_ref")
    os.makedirs(build, exist_ok=True)
    return load("pi05_math_ref", [os.path.join(HERE, "math_ref.cu")],
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"], build_directory=build, verbose=False)


def ordered(u):
    """Float bit patterns as ordered integers (ulp distance = difference)."""
    u = u.astype(np.int64)
    return np.where(u >= 2**31, 2**31 - u, u)


def compare(ref, got, kind):
    """-> (mismatches, max ulp, first bad index) of two uint32 bit arrays; NaN == NaN."""
    if kind == "bits":
        bad = ref != got
        return int(bad.sum()), 0, (int(np.argmax(bad)) if bad.any() else -1)
    rn = (ref & 0x7FFFFFFF) > 0x7F800000
    gn = (got & 0x7FFFFFFF) > 0x7F800000
    bad = (ref != got) & ~(rn & gn)
    ulp = np.abs(ordered(ref) - ordered(got))
    ulp = np.where(rn != gn, 2**31, ulp)
    return int(bad.sum()), int(ulp[bad].max()) if bad.any() else 0, (int(np.argmax(bad)) if bad.any() else -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--only", default=None, help="comma-separated expression names")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [e.name for e in EXPRS]
    ref = load_ref()
    print("built math_ref", flush=True)
    result = {"n": args.n, "exprs": {}}
    if not args.compile_only:
        import torch
        from cutlass.cute.runtime import from_dlpack
        rng = np.random.default_rng(args.seed)
    for name in names:
        e = BY_NAME[name]
        profs = profiles_for(e)
        pn = tuple(profs)
        t0 = time.time()
        if args.compile_only:
            fi = cute.runtime.make_fake_compact_tensor(Float32, (e.n_in * args.n,), assumed_align=16)
            fo = cute.runtime.make_fake_compact_tensor(Float32, (len(pn) * 2 * args.n,), assumed_align=16)
            cute.compile(host, fi, fo, args.n, EXPRS.index(e), pn)
            print(f"compiled {name} x {pn} in {time.time() - t0:.1f}s", flush=True)
            continue
        arrs = e.gen(rng, args.n)
        n = len(arrs[0])
        host_in = np.concatenate([a.reshape(-1) for a in arrs]).astype(np.float32)
        d_in = torch.from_numpy(host_in).cuda()
        d_ref = torch.zeros(2 * n, dtype=torch.int32, device="cuda")
        ref.run(EXPRS.index(e), d_in, d_ref, n)
        d_out = torch.zeros(len(pn) * 2 * n, dtype=torch.float32, device="cuda")
        compiled = cute.compile(host, from_dlpack(d_in, assumed_align=16), from_dlpack(d_out, assumed_align=16), n,
                                EXPRS.index(e), pn)
        compiled(from_dlpack(d_in, assumed_align=16), from_dlpack(d_out, assumed_align=16), n)
        torch.cuda.synchronize()
        r = d_ref.cpu().numpy().view(np.uint32).reshape(2, n)
        g = d_out.cpu().numpy().view(np.uint32).reshape(len(pn), 2, n)
        ent = {}
        for pi, p in enumerate(pn):
            cmp = [compare(r[o], g[pi, o], e.kind) for o in range(e.n_out)]
            ent[p] = {"mismatch": [c[0] for c in cmp], "max_ulp": [c[1] for c in cmp],
                      "first_bad": [c[2] for c in cmp], "choice": {u: profs[p].get(u, X.CANDIDATES[u][0])
                                                                  for u in e.uses}}
        exact = [p for p in pn if sum(ent[p]["mismatch"]) == 0]
        best = exact[0] if exact else min(pn, key=lambda p: sum(ent[p]["mismatch"]))
        result["exprs"][name] = {"n": n, "bit_exact": bool(exact), "winner": best, "candidates": ent}
        print(f"{name:12s} n={n} winner={best} " + " ".join(f"{p}:{sum(ent[p]['mismatch'])}" for p in pn), flush=True)
    if args.out and not args.compile_only:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "math.json"), "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
