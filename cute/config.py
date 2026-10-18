"""Geometry, tuning knobs and workspace layout of the pi0.5 megakernel (mirrors mk/pi05_mk.cu, production config)."""

from __future__ import annotations

from dataclasses import dataclass

from . import iket_marks as IM

T = 50
ADIM = 32
W = 1024
L = 18
NSTEP = 10
QD = 2048
HD = 256
NQH = 8
KVD = 256
FF = 4096
NORMS = 2 * L + 1
MODROW = NORMS * 3 * W
QROWS = T * NQH
NQKV = QD + 2 * KVD

NTHREADS = 256
NWARPS = NTHREADS // 32
KC = 128
NSSP = W // 64

CNT_STRIDE = 32
CNT_QKV = 0
CNT_ATT = 16
CNT_GU = 32
NCNT = CNT_GU + FF // 32
ARR_QKV = (HD // 32) * (W // KC)
ARR_GU = 2 * (W // KC)
FUSE_AB, FUSE_CC3, FUSE_FG, FUSE_BACKOFF, FUSE_DEFNORM = 1, 2, 4, 8, 16
FUSE_ABL = 32  # A + B fused into one barrier-free phase (CuTe only; the CUDA kernel has no bit 32, see abl.py)

BAR_STRIDE = 32
BAR_WORDS = 160 * BAR_STRIDE + 2 * BAR_STRIDE
BAR_CNT_WORD = 160 * BAR_STRIDE  # grid-barrier counter (flag words 0..159 are unused with PI05_BARRIER=3)
BAR_EXIT_WORD = BAR_CNT_WORD + BAR_STRIDE  # exit counter of the in-kernel reset (second line of the bar area; CUDA leaves it unused)

ATT_ROWS = 64
ATT_NRG = (QROWS + ATT_ROWS - 1) // ATT_ROWS
KS_STRIDE = HD + 8
VS_STRIDE = HD + 8
AS_STRIDE = KC + 32
WS_STRIDE = 40
NUM_SMS = 148
GEMM_SCRATCH = (NWARPS * 4 + 15) // 16 * 16 if IM.GEMM else 0  # PI05_IKET_GEMM=1: one u32 per warp (gmark_dep) after all other smem

LAYOUT_ORDER = ("temb", "tmid", "cond", "mod", "xbuf", "rope", "h", "xn", "part", "q", "ksuf", "vsuf",
                "scores", "stats", "opart", "attn", "act", "cnt", "ssp", "bar")
# dtype of each workspace buffer (parity compares by element type)
LAYOUT_DTYPE = {"temb": "f32", "tmid": "f32", "cond": "f32", "mod": "f32", "xbuf": "f32", "rope": "f32",
                "h": "bf16", "xn": "bf16", "part": "f32", "q": "bf16", "ksuf": "bf16", "vsuf": "bf16",
                "scores": "f32", "stats": "f32", "opart": "f32", "attn": "bf16", "act": "bf16",
                "cnt": "u32", "ssp": "f32", "bar": "u32"}


@dataclass(frozen=True)
class Knobs:
    prefix: int = 968
    att_keys: int = 64
    fuse: int = 18
    wpre: int = 15
    d_wpt: int = 2
    g_batch: int = 0  # stage G load batching, bit-exact: 1 = loads of both of a CTA's tasks before any math, 2 = same with task 2's up loads after task 1's sums (default off until the GPU A/B)
    host_zero: bool = False  # True: the host zeroes cnt..bar before each launch (phase 5); False: last-CTA-out reset in the kernel

    def __post_init__(self):
        if self.prefix <= 0:
            raise ValueError("prefix must be positive")
        if self.att_keys % 16 or not 0 < self.att_keys <= 64:
            raise ValueError("att_keys must be a multiple of 16, at most 64")
        if QROWS % 16 or NWARPS != 2 * (ATT_ROWS // 16) or HD != 256:
            raise ValueError("FA unit geometry")
        if self.att_nchunk * ATT_NRG > NUM_SMS:
            raise ValueError("attention units must fit one wave")
        if NWARPS % self.d_wpt or 4 % self.d_wpt:
            raise ValueError("d_wpt must divide 4 and the warp count")
        if self.g_batch not in (0, 1, 2):
            raise ValueError("g_batch must be 0, 1 or 2")
        if not 0 <= self.wpre <= 15 or not 0 <= self.fuse < 64:
            raise ValueError("wpre / fuse bits out of range")
        if self.fuse & FUSE_ABL and self.fuse & FUSE_AB:
            raise ValueError("FUSE_ABL (32) excludes FUSE_AB (1)")
        if not self.fuse & FUSE_DEFNORM:
            raise ValueError("fuse must include FUSE_DEFNORM (16): the non-defnorm row path is not ported")
        nsl = 4 // self.d_wpt
        if self.d_wpt > 1 and 64 * AS_STRIDE * 2 + NWARPS * T * WS_STRIDE * 4 + NWARPS * nsl * 4 * 32 * 16 > self.smem_bytes:
            raise ValueError("wsplit smem")

    @property
    def nk(self) -> int:
        return self.prefix + T

    @property
    def att_nchunk(self) -> int:
        return (self.nk + self.att_keys - 1) // self.att_keys

    @property
    def smem_bytes(self) -> int:
        c1 = (self.att_keys * KS_STRIDE * 2 + ATT_ROWS * KS_STRIDE * 2 + self.att_keys * VS_STRIDE * 2
              + self.att_keys)
        c2 = 64 * (KC + 32) * 2 + NWARPS * T * 40 * 4 + NWARPS * 2 * 4 * 32 * 16
        base = max(c1, c2, NSTEP * W * 4)
        if self.fuse & FUSE_ABL:
            base = max(base, 8 * 64 * KC * 2)  # FUSE_ABL: 8 private A tiles (the fp32 partial buffer reuses them)
        return base + GEMM_SCRATCH


def _align(x: int) -> int:
    return (x + 255) & ~255


def workspace_layout(k: Knobs) -> dict[str, tuple[int, int]]:
    """name -> (byte offset, byte size) in the order of make_layout(); 'total' -> (0, total bytes)."""
    part = max((W // KC) * T * NQKV, (QD // KC) * T * W, (W // KC) * T * 2 * FF, (FF // KC) * T * W)
    nc = k.att_nchunk
    sizes = {
        "temb": NSTEP * W * 4, "tmid": NSTEP * W * 4, "cond": NSTEP * W * 4, "mod": NSTEP * MODROW * 4,
        "xbuf": T * ADIM * 4, "rope": T * (HD // 2) * 2 * 4, "h": T * W * 2, "xn": T * W * 2,
        "part": part * 4, "q": T * QD * 2, "ksuf": T * KVD * 2, "vsuf": T * KVD * 2,
        "scores": QROWS * nc * k.att_keys * 4, "stats": QROWS * nc * 2 * 4, "opart": nc * QROWS * HD * 4,
        "attn": T * QD * 2, "act": T * FF * 2, "cnt": NCNT * CNT_STRIDE * 4, "ssp": T * NSSP * 4,
        "bar": BAR_WORDS * 4,
    }
    out, off = {}, 0
    for name in LAYOUT_ORDER:
        out[name] = (off, sizes[name])
        off = _align(off + sizes[name])
    out["total"] = (0, off)
    return out
