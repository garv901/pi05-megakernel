"""Per-phase critical-path attribution from the megakernel block-0 stamps."""
import json, sys
import numpy as np
from .phases import PHASES, mk_slot, L, NSTEP


def analyze(ts: np.ndarray, total_us: float | None = None) -> dict:
    ts = ts.astype(np.int64)
    t0 = ts[0]
    us = (ts - t0) / 1e3
    per = {p: [] for p in PHASES}
    tail = []
    for s in range(NSTEP):
        for l in range(L):
            for i, p in enumerate(PHASES):
                a = mk_slot(s, l, i)
                if i < 8:
                    b = a + 1
                elif l < L - 1:
                    b = mk_slot(s, l + 1, 0)
                else:   # I on the last layer: residual, +9 stamp; tail until next step A-in
                    b = a + 1
                per[p].append(us[b] - us[a])
        if s < NSTEP - 1:
            tail.append(us[mk_slot(s + 1, 0, 0)] - us[mk_slot(s, L - 1, 9)])
    out = {"phases_us_per_layer": {}, "tail_us_per_step": float(np.mean(tail)) if tail else None}
    for p in PHASES:
        v = np.array(per[p])
        out["phases_us_per_layer"][p] = {"mean": float(v.mean()), "median": float(np.median(v)),
                                         "p90": float(np.percentile(v, 90)), "n": len(v)}
    out["prologue_us"] = float(us[mk_slot(0, 0, 0)])  # kernel start (block 0) -> first A barrier exit
    out["last_stamp_us"] = float(us[mk_slot(NSTEP - 1, L - 1, 9)])
    out["sum_phase_mean_us_x180"] = float(sum(out["phases_us_per_layer"][p]["mean"] for p in PHASES) * 180)
    return out


if __name__ == "__main__":
    ts = np.load(sys.argv[1])
    print(json.dumps(analyze(ts), indent=1))
