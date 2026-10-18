"""One phase taxonomy for both arms + byte model.

Per layer: A qkv | B rope/qkv-reduce | C attention (+combine) | D o-proj | E resid1+adaRMS |
           F gate/up | G act | H down | I resid2+adaRMS(+step tail on the last layer).
Megakernel: phase = interval between consecutive block-0 barrier-exit stamps.
Graph:      phase = completion-to-completion of the corresponding kernel (valid under PDL).
"""
from __future__ import annotations

PHASES = ["A_qkv", "B_rope", "C_attn", "D_o", "E_res1", "F_gu", "G_act", "H_down", "I_res2"]
PHASE_LONG = {"A_qkv": "qkv GEMM", "B_rope": "qkv reduce+RoPE+KV append", "C_attn": "attention (+combine)",
              "D_o": "o-proj GEMM", "E_res1": "residual+adaRMS (ffn)", "F_gu": "gate/up GEMM",
              "G_act": "gelu*up", "H_down": "down GEMM", "I_res2": "residual+adaRMS (next)"}

L, NSTEP = 18, 10
W, FF, QD, KVD, HD, T = 1024, 4096, 2048, 256, 256, 50

# megakernel block-0 stamp slots: per step s, layer l: base = 5 + 163*s + 9*l;
# phase i spans stamp[base+i] -> stamp[base+i+1]  (A-in .. after-A, ...); phase I ends at the next layer's A-in
MK_SLOT_BASE = 5
MK_STEP_STRIDE = 163
MK_LAYER_STRIDE = 9


def mk_slot(step: int, layer: int, k: int) -> int:
    return MK_SLOT_BASE + MK_STEP_STRIDE * step + MK_LAYER_STRIDE * layer + k


def ideal_bytes(prefix: int = 128) -> dict:
    """Unique HBM bytes per layer per phase (attributed to the consuming phase)."""
    bf = 2
    kv = prefix * KVD * bf * 2          # prefix K and V of this layer
    mod = 3 * W * 4                     # one adaRMS modulation row (fp32)
    return {"A_qkv": (QD + 2 * KVD) * W * bf, "B_rope": 0, "C_attn": kv,
            "D_o": W * QD * bf, "E_res1": mod, "F_gu": 2 * FF * W * bf + mod, "G_act": 0,
            "H_down": W * FF * bf, "I_res2": mod}


def active_sms_mk() -> dict:
    """Active SMs per phase from the kernel's static task maps (148 CTAs; README/plan numbers)."""
    return {"A_qkv": 80, "B_rope": 70, "C_attn": "21-84", "D_o": 128, "E_res1": "50?", "F_gu": 148,
            "G_act": 148, "H_down": 128, "I_res2": "50?"}
