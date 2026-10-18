"""Per-kernel ncu tables (graph layer kernels, mk kernel) from stage2 raw CSVs."""
import csv, json, sys
from .phases import PHASES
STALLS = ["barrier","branch_resolving","dispatch_stall","drain","lg_throttle","long_scoreboard","math_pipe_throttle",
          "membar","mio_throttle","no_instruction","not_selected","selected","short_scoreboard","sleeping","tex_throttle","wait"]

def load(path):
    rows = list(csv.reader(open(path)))
    h = rows[0]; units = rows[1]
    return h, units, rows[2:]

def num(x):
    try: return float(x.replace(",", ""))
    except Exception: return float("nan")

def table(path):
    h, u, rows = load(path)
    ix = {c: i for i, c in enumerate(h)}
    out = []
    for r in rows:
        g = lambda c: num(r[ix[c]]) if c in ix else float("nan")
        st = {s: g(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") for s in STALLS}
        top = sorted(st.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1))[:3]
        tu = u[ix["gpu__time_duration.sum"]]
        t = g("gpu__time_duration.sum") * {"ns": 1e-3, "us": 1, "usecond": 1, "ms": 1e3, "msecond": 1e3, "nsecond": 1e-3}.get(tu, 1)
        bu = u[ix["dram__bytes_read.sum"]]
        mul = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "kbyte": 1e3}.get(bu, 1)
        wu = u[ix["dram__bytes_write.sum"]]
        wmul = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "kbyte": 1e3}.get(wu, 1)
        out.append(dict(id=r[ix["ID"]], kernel=r[ix["Kernel Name"]][:48], grid=r[ix["Grid Size"]], block=r[ix["Block Size"]],
                        us=t, rd_MB=g("dram__bytes_read.sum") * mul / 1e6, wr_MB=g("dram__bytes_write.sum") * wmul / 1e6,
                        dram_pct=g("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                        l2_hit=g("lts__t_sector_hit_rate.pct"),
                        sm_active_avg_cycles=g("sm__cycles_active.avg"), warps_active_pct=g("sm__warps_active.avg.pct_of_peak_sustained_active"),
                        occ_limit_regs=g("launch__occupancy_limit_registers"), occ_limit_smem=g("launch__occupancy_limit_shared_mem"),
                        tensor_pct=g("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"),
                        issue_pct=g("sm__issue_active.avg.pct_of_peak_sustained_elapsed"),
                        top_stalls=[(k, round(v, 2)) for k, v in top]))
    return out, u, h

if __name__ == "__main__":
    d = sys.argv[1]
    for name in ["ncu_graph_layer_raw.csv", "ncu_mk_kernel_raw.csv"]:
        t, u, h = table(f"{d}/{name}")
        print("==", name, "units:", {c: u[h.index(c)] for c in ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum"]})
        for i, r in enumerate(t):
            print(i, json.dumps(r))
