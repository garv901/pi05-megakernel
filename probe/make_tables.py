"""Render markdown tables from analysis/summary.json + ptrace summary."""
import json
import numpy as np
from .phases import PHASES, ideal_bytes
A = "/raid/garv901/vibe-megakernel/megabench/sota/pi05/probe/runs/analysis/"
r = json.load(open(A + "summary.json"))
pt = json.loads(open(A + "ptrace_summary.txt").read()[open(A + "ptrace_summary.txt").read().index("{"):])
ib = ideal_bytes()
PEAK, COPY = 8.0e12, 6.567e12
mk = {p: np.mean([r["mk_stamps_stage0"]["phases_us_per_layer"][p]["mean"], r["mk_stamps_stage2_nsys"][p]["mean"]]) for p in PHASES}
gp = r["graph"]["graph-pdl"]["c2c"]; gn = r["graph"]["graph-nopdl"]["c2c"]
lead = r["graph"]["graph-pdl"]["lead"]
ncu = r["ncu_graph_layer1"]
mkobs = r["mk_obs_bytes_per_phase"]; gobs = r["graph_nsys_obs"]["graph-pdl"]["per_phase_B"]
def mbu(b, t, peak=PEAK): return 100 * b / (t * 1e-6) / peak
print("### Table 1: per-phase time and MBU_attr (ideal bytes / phase time / peak)\n")
print("| phase | mk us | graph-pdl us | graph-nopdl us | mk - pdl | ideal MB | MBU_attr mk (8 / 6.57 TB/s) | MBU_attr pdl | MBU_attr nopdl |")
print("|---|---|---|---|---|---|---|---|---|")
for p in PHASES:
    b = ib[p]
    f = lambda t: f"{mbu(b, t):.1f}% / {mbu(b, t, COPY):.1f}%" if b > 5e5 else "n/a (<0.5 MB)"
    f2 = lambda t: f"{mbu(b, t):.1f}%" if b > 5e5 else "n/a"
    print(f"| {p} | {mk[p]:.2f} | {gp[p]:.2f} | {gn[p]:.2f} | {mk[p]-gp[p]:+.2f} | {b/1e6:.2f} | {f(mk[p])} | {f2(gp[p])} | {f2(gn[p])} |")
print(f"| **sum/layer** | **{sum(mk.values()):.2f}** | **{sum(gp.values()):.2f}** | **{sum(gn.values()):.2f}** | **{sum(mk.values())-sum(gp.values()):+.2f}** | {sum(ib.values())/1e6:.2f} | {mbu(sum(ib.values()), sum(mk.values())):.1f}% | {mbu(sum(ib.values()), sum(gp.values())):.1f}% | {mbu(sum(ib.values()), sum(gn.values())):.1f}% |")
print("\n### Table 2: MBU_obs, occupancy, active SMs, dominant stall\n")
print("| phase | mk obs MB in window (nsys-fold, 10us-smoothed) | graph-pdl obs MB (same) | graph kernel ncu isolated: DRAM MB / own us / MBU_kernel | graph grid (CTAs) | graph achieved occ (warps_active %) | graph top stalls (warps/issue) | mk active CTAs (PTRACE) | mk busy frac of interval |")
print("|---|---|---|---|---|---|---|---|---|")
g = r["graph"]["graph-pdl"]["info"]
for p in PHASES:
    n = ncu[p]; ts = ", ".join(f"{k} {v}" for k, v in n["top_stalls"][:2])
    print(f"| {p} | {mkobs[p]/1e6:.1f} | {gobs[p]/1e6:.1f} | {n['rd_MB']+n['wr_MB']/1e6:.1f} / {n['us']:.1f} / {mbu((n['rd_MB']+n['wr_MB']/1e6)*1e6, n['us']):.1f}% | {n['grid']} x {n['block'].strip('()').split(',')[0]} thr | {n['warps_active_pct']:.1f} | {ts} | {pt[p]['active_ctas']:.0f} | {pt[p]['busy_frac_of_interval']:.2f} |")
print("\n### Table 3: mk sync/boundary budget per phase (PTRACE, us)\n")
print("| phase | interval | slowest CTA done | barrier-exit latency | arrive skew | mean work (active CTAs) |")
print("|---|---|---|---|---|---|")
tot = np.zeros(4)
for p in PHASES:
    v = pt[p]; tot += [v['interval_us'], v['slowest_cta_done_us'], v['barrier_exit_latency_us'], v['arrive_skew_us']]
    print(f"| {p} | {v['interval_us']:.2f} | {v['slowest_cta_done_us']:.2f} | {v['barrier_exit_latency_us']:.2f} | {v['arrive_skew_us']:.2f} | {v['mean_work_us_active']:.2f} |")
print(f"| sum | {tot[0]:.2f} | {tot[1]:.2f} | {tot[2]:.2f} | {tot[3]:.2f} | |")
print("\nPDL start-before-prev-end (us): " + ", ".join(f"{p[0]} {lead[p]:.2f}" for p in PHASES))
print("graph windows (nsys, us):", {a: (round(float(np.mean(v['window'])),1), round(float(np.mean(v['prologue'])),1)) for a, v in r['graph'].items()})
