"""Per-phase work / skew / barrier-exit decomposition and active-CTA counts from ptrace.npz."""
import json, sys
import numpy as np
from .phases import PHASES, mk_slot, L, NSTEP

ACTIVE_FRAC = 0.5   # a CTA is "loaded" in a phase when its work exceeds this fraction of the slowest CTA's work


def phase_bounds(s, l, i):
    a = mk_slot(s, l, i)
    if i < 8:
        b = mk_slot(s, l, i + 1)
    elif l < L - 1:
        b = mk_slot(s, l + 1, 0)
    else:
        b = mk_slot(s, l, 9)
    return a, b


def analyze(npz_path):
    z = np.load(npz_path)
    pt = z["ptrace"].astype(np.float64) / 1e3   # [launch, bar, cta, 2] us; -1e-3 = unwritten
    nl = pt.shape[0]
    res = {}
    for i, p in enumerate(PHASES):
        rows = []
        for li in range(nl):
            for s in range(NSTEP):
                for l in range(L):
                    a, b = phase_bounds(s, l, i)
                    st_exit = pt[li, a, :, 1]
                    en_arr = pt[li, b, :, 0]
                    en_exit = pt[li, b, :, 1]
                    if (st_exit < -1e6).any() or (en_arr < -1e6).any() or (en_exit < -1e6).any():
                        continue
                    work = en_arr - st_exit
                    t_s = np.median(st_exit)
                    interval = np.median(en_exit) - t_s
                    rows.append([interval, en_arr.max() - t_s, np.median(en_exit) - en_arr.max(), work.mean(),
                                 np.median(work), work.max(), (work > ACTIVE_FRAC * work.max()).sum(),
                                 (work[work > ACTIVE_FRAC * work.max()]).mean(),
                                 en_arr.max() - np.median(en_arr), st_exit.max() - st_exit.min()])
        r = np.array(rows)
        m = r.mean(0)
        res[p] = {"n": len(r), "interval_us": m[0], "slowest_cta_done_us": m[1], "barrier_exit_latency_us": m[2],
                  "mean_work_us_all_ctas": m[3], "median_work_us": m[4], "max_work_us": m[5],
                  "active_ctas": m[6], "mean_work_us_active": m[7], "arrive_skew_us": m[8],
                  "entry_spread_us": m[9],
                  "busy_frac_of_interval": m[3] / m[0]}
    return res


if __name__ == "__main__":
    out = analyze(sys.argv[1])
    print(json.dumps(out, indent=1))
