"""Completion-to-completion per-phase attribution of the graph arm from an nsys cuda_gpu_trace CSV
(--cuda-graph-trace=node).  Valid under PDL, where kernel start times overlap the predecessor."""
import csv, json, statistics, sys
import numpy as np
from .phases import PHASES, L, NSTEP

NPRO = 18  # kernels before the first layer's qkv GEMM (prologue incl. step-0 boundary head)


def load_reps(path):
    rows = list(csv.DictReader(open(path)))
    rows = [r for r in rows if not r["Name"].startswith("[CUDA")]   # drop memcpy rows
    rows = [r for r in rows if "FillFunctor<unsigned char>" not in r["Name"]]   # the L2-flush fill
    rows.sort(key=lambda r: int(r["Start (ns)"]))
    reps, cur, last_end = [], [], None
    for r in rows:
        st = int(r["Start (ns)"])
        if last_end is not None and st - last_end > 100_000 and cur:   # > 100 us idle: a new launch
            reps.append(cur); cur = []
        cur.append(r)
        last_end = max(last_end or 0, st + int(r["Duration (ns)"]))
    if cur:
        reps.append(cur)
    return [r for r in reps if len(r) >= 1600]


def find_layer_start(rep):
    for i, r in enumerate(rep):
        if "_rope_kv_kernel" in r["Name"]:
            return i - 1  # qkv gemm precedes first rope
    raise RuntimeError("no rope kernel")


def analyze_rep(rep):
    s0 = find_layer_start(rep)
    n_layer_k = NSTEP * L * 9
    ks = rep[s0:s0 + n_layer_k]
    assert len(ks) == n_layer_k and "_rope_kv" in ks[1]["Name"], (len(ks), s0)
    start = np.array([int(r["Start (ns)"]) for r in ks], dtype=np.int64)
    dur = np.array([int(r["Duration (ns)"]) for r in ks], dtype=np.int64)
    end = start + dur
    prev_end = np.concatenate([[int(rep[s0 - 1]["Start (ns)"]) + int(rep[s0 - 1]["Duration (ns)"])], end[:-1]])
    c2c = (end - prev_end) / 1e3
    own = dur / 1e3
    # overlap with predecessor (PDL): how early the kernel started relative to the predecessor's end
    lead = (prev_end - start) / 1e3
    res = {"phases": {}}
    for i, p in enumerate(PHASES):
        sel = slice(i, None, 9)
        res["phases"][p] = {"c2c_us_mean": float(c2c[sel].mean()), "c2c_us_median": float(np.median(c2c[sel])),
                            "own_us_mean": float(own[sel].mean()), "own_us_median": float(np.median(own[sel])),
                            "start_before_prev_end_us_mean": float(lead[sel].mean()),
                            "name": ks[i]["Name"][:60], "grid": [int(ks[i][g]) for g in ("GrdX", "GrdY", "GrdZ")],
                            "block": int(ks[i]["BlkX"]), "regs": int(ks[i]["Reg/Trd"])}
    t_first = int(rep[0]["Start (ns)"])
    t_layers0 = int(start[0])
    t_end = int(end[-1])
    res["prologue_us"] = (int(rep[s0 - 1]["Start (ns)"]) + int(rep[s0 - 1]["Duration (ns)"]) - t_first) / 1e3
    res["layers_window_us"] = (t_end - int(prev_end[0])) / 1e3
    res["total_window_us"] = (t_end - t_first) / 1e3
    res["layer_starts_ns"] = prev_end[0::9].tolist()          # 180 layer starts (end of previous layer's last kernel)
    res["phase_end_rel_us_mean"] = [float(np.mean((end[i::9] - prev_end[0::9]) / 1e3)) for i in range(9)]
    res["sum_c2c_us"] = float(c2c.sum())
    res["sum_phase_c2c_per_layer_us"] = float(sum(res["phases"][p]["c2c_us_mean"] for p in PHASES))
    return res


if __name__ == "__main__":
    reps = load_reps(sys.argv[1])
    out = [analyze_rep(r) for r in reps]
    print(json.dumps({k: v for k, v in out[0].items() if k != "layer_starts_ns"}, indent=1))
    print("n reps", len(out), "total windows", [round(o["total_window_us"], 1) for o in out])
