"""Emit markdown tables from per_phase.csv, kernel CSVs and nsys_unmix.json."""
import csv, json, os, numpy as np
P = os.path.dirname(os.path.abspath(__file__)); R = f"{P}/runs/ncu_phase"
pp = list(csv.DictReader(open(f"{P}/per_phase.csv")))
get = lambda s, ph: next(r for r in pp if r["system"] == s and r["phase"] == ph)
f = lambda x, n=1: f"{float(x):.{n}f}"
PH = ["A_qkv", "B_rope", "C_attn", "D_o", "E_res1", "F_gu", "G_act", "H_down", "I_res2"]
out = []
def top(r, k=2): return ", ".join(r["top_stalls"].split(";")[:k]).replace("_", " ")
ker = list(csv.DictReader(open(f"{R}/csv/kernels_graph_layer8_step5_l2cold.csv")))
kw = list(csv.DictReader(open(f"{R}/csv/kernels_graph_layer8_step5_l2warm.csv")))
mkk = list(csv.DictReader(open(f"{R}/csv/kernels_ncu_mk_kernel.csv")))[0]
out.append("### T1 compact table (mk vs graph+PDL, ncu PM, duration-weighted over 180 layer windows, mean of reps)\n")
out.append("| phase | mk time us | mk DRAM rd % (ncu peak) | mk SM act % | mk CTAs | mk top stalls | gr-pdl time us | gr DRAM rd % | gr SM act % | gr CTAs | gr occ ach/theo % (iso L8) | gr top stalls |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, ph in enumerate(PH):
    m, g, k = get("mk", ph), get("graph-pdl", ph), ker[i]
    out.append(f"| {ph} | {f(m['dur_us'],2)} | {f(m['dram_rd_pct'])} | {f(m['sm_active_pct'])} | {f(m['ctas_active'],0)} | {top(m)} | {f(g['dur_us'],2)} | {f(g['dram_rd_pct'])} | {f(g['sm_active_pct'])} | {f(g['ctas_active'],0)} | {f(k['occ_achieved_pct'])}/{f(k['occ_theoretical_pct'])} | {top(g)} |")
tm = sum(float(get('mk', p)['dur_us']) for p in PH); tg = sum(float(get('graph-pdl', p)['dur_us']) for p in PH)
out.append(f"| layer sum | {tm:.2f} | | | | | {tg:.2f} | | | | | |")
out.append(f"\nmk kernel-wide ncu (stage2, one launch): occupancy achieved {f(mkk['occ_achieved_pct'])}% = theoretical {f(mkk['occ_theoretical_pct'])}% (limiter {mkk['occ_limiter']}), grid {mkk['grid']}x{mkk['block']}, {mkk['regs']} regs, {mkk['smem_B']} B smem, SM active {f(mkk['sm_active_pct'])}%.\n")
out.append("### T2 full per-phase table incl. prologue/tail, DRAM bytes, GB/s, % of 8 TB/s, L2, issue, tensor, p10/p90\n")
out.append("| system | phase | time us (p10/p90) | DRAM rd % ncu-peak (p10/p90) | rd GB/s | rd % of 8 TB/s | wr % ncu-peak | wr GB/s | rd MB/window | L2 thr % | L2 hit % | SM act % | CTAs | issue % | IPC | tensor % | top-3 stalls | PTRACE tail share % |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in pp:
    p = "" if r["dur_p10"] in ("", "nan") else f" ({f(r['dur_p10'],1)}/{f(r['dur_p90'],1)})"
    q = f" ({f(r['dram_rd_pct_p10'])}/{f(r['dram_rd_pct_p90'])})" if r["n_windows"] != "1" and r["phase"] != "PRO" else ""
    tl = "" if r["ptrace_tail_share_pct"] in ("", "nan") else f(r["ptrace_tail_share_pct"], 0)
    out.append(f"| {r['system']} | {r['phase']} | {f(r['dur_us'],2)}{p} | {f(r['dram_rd_pct'])}{q} | {f(r['dram_rd_GBps'],0)} | {f(r['dram_rd_pct_of_8TB'])} | {f(r['dram_wr_pct'],2)} | {f(r['dram_wr_GBps'],0)} | {f(r['dram_rd_MB_per_window'],2)} | {f(r['l2_pct'])} | {f(r['l2_hit_pct'])} | {f(r['sm_active_pct'])} | {f(r['ctas_active'],0)} | {f(r['issue_pct'],1)} | {f(r['ipc_active'],2)} | {f(r['tensor_pct'],1)} | {r['top_stalls'].replace(';', '; ')} | {tl} |")
for tag, K in (("L2-cold (--cache-control all)", ker), ("L2-warm (--cache-control none)", kw)):
    out.append(f"\n### T3 graph isolated per-kernel ncu, layer 8 step 5, {tag}\n")
    out.append("| phase | kernel | us | grid x block | regs | smem KB | rd MB | rd GB/s | DRAM % ncu / rd % 8TB | L2 thr % | L2 hit % | SM act % | occ ach/theo % (limiter) | issue act % | tensor % | top stalls |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, k in enumerate(K):
        out.append(f"| {PH[i]} | {k['kernel'][:26]} | {f(k['dur_us'],2)} | {int(float(k['grid']))}x{int(float(k['block']))} | {int(float(k['regs']))} | {float(k['smem_B'])/1024:.0f} | {f(k['dram_rd_MB'],2)} | {f(k['dram_rd_GBps'],0)} | {f(k['dram_pct_ncu'])} / {f(k['dram_rd_pct_8TB'])} | {f(k['l2_thr_pct'])} | {f(k['l2_hit_pct'])} | {f(k['sm_active_pct'])} | {f(k['occ_achieved_pct'])}/{f(k['occ_theoretical_pct'])} ({k['occ_limiter']}) | {f(k['issue_active_pct'])} | {f(k['tensor_pct_active'])} | {k['top_stalls'].replace(';', '; ')} |")
nj = json.load(open(f"{R}/analysis/nsys_unmix.json"))
def x(arm, met, nm): return nj[arm]["metrics"][met]["x"].get(nm)
out.append("\n### T4 nsys GPU-metrics NNLS unmixing (pooled over 180 layers) vs ncu PM\n")
out.append("| phase | mk DRAM rd PM | mk nsys unmix (mean of 3 reps) | gr-pdl DRAM rd ncu-graph PM | gr-pdl nsys unmix | gr-nopdl nsys unmix | mk SMs-active nsys | mk tensor nsys | gr-pdl SMs-active nsys | gr-pdl tensor nsys |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for ph in PH:
    nm = ph + "|all"
    mku = np.mean([x(f"mk_r{r}", "dram_read_pct", nm) for r in (1, 2, 3)])
    s = lambda a, m: f(x(a, m, nm), 1)
    out.append(f"| {ph} | {f(get('mk', ph)['dram_rd_pct'])} | {mku:.1f} | {f(get('graph-pdl', ph)['dram_rd_pct'])} | {s('graph-pdl','dram_read_pct')} | {s('graph-nopdl','dram_read_pct')} | {np.mean([x(f'mk_r{r}','sms_active',nm) for r in (1,2,3)]):.1f} | {np.mean([x(f'mk_r{r}','tensor_active',nm) for r in (1,2,3)]):.1f} | {s('graph-pdl','sms_active')} | {s('graph-pdl','tensor_active')} |")
out.append("\nfit quality (DRAM read, R2 / lag ns): " + "; ".join(f"{k} {nj[k]['metrics']['dram_read_pct']['r2']:.2f} / {nj[k]['lag_ns']:.0f}" for k in nj))
# reps repeatability
out.append("\n### T5 repeatability: sd of the per-rep DRAM-rd % phase means (percentage points)\n")
out.append("| phase | mk (3 reps) | graph-pdl (3 reps) |\n|---|---|---|")
for ph in PH: out.append(f"| {ph} | {f(get('mk', ph)['dram_rd_pct_repsd'],2)} | {f(get('graph-pdl', ph)['dram_rd_pct_repsd'],2)} |")
open(f"{R}/analysis/tables.md", "w").write("\n".join(out)); print("\n".join(out))
