"""Extract the per-kernel hardware metrics of an isolated-kernel ncu report (kernel replay) into rows."""
import csv, sys
from . import ncu_pm as N

REASONS = ["selected", "not_selected", "barrier", "membar", "wait", "long_scoreboard", "short_scoreboard", "mio_throttle", "lg_throttle",
           "math_pipe_throttle", "no_instruction", "branch_resolving", "dispatch_stall", "drain", "misc", "sleeping", "tex_throttle"]


def rows(path, nsm=148):
    import sys as _s
    r = N.ncu_report.load_report(path); rg = r.range_by_idx(0); N._KEEP.extend([r, rg])
    out = []
    for i in range(rg.num_actions()):
        a = rg.action_by_idx(i); g = lambda n, d=float("nan"): N.val(a, n, d)
        dur = g("gpu__time_duration.sum")           # ns
        rd, wr = g("dram__bytes_read.sum"), g("dram__bytes_write.sum", 0.0)
        thr = g("dram__throughput.avg.pct_of_peak_sustained_elapsed")
        peak = (rd + wr) / (dur * 1e-9) / (thr / 100) if thr and (rd + wr) > 0 else float("nan")
        lim = {k: g(f"launch__occupancy_limit_{k}") for k in ("registers", "shared_mem", "blocks", "warps")}
        limiter = ",".join(k for k, v in lim.items() if v == min(lim.values()))
        stall = {s: g(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio", 0.0) for s in REASONS}
        tot = sum(stall.values())
        top = sorted(((v / tot, k) for k, v in stall.items() if k not in ("selected",)), reverse=True)[:3] if tot > 0 else []
        row = dict(idx=i, kernel=a.name()[:70], dur_us=dur / 1e3, grid=g("launch__grid_size"), block=g("launch__block_size"),
                   cluster=g("launch__cluster_size"), regs=g("launch__registers_per_thread"),
                   smem_B=g("launch__shared_mem_per_block"), waves=g("launch__waves_per_multiprocessor"),
                   dram_rd_MB=rd / 1e6, dram_wr_MB=wr / 1e6, dram_rd_GBps=rd / dur, dram_wr_GBps=wr / dur,
                   dram_pct_ncu=thr, ncu_peak_TBps=peak / 1e12, dram_rd_pct_8TB=rd / (dur * 1e-9) / 8e12 * 100,
                   l2_thr_pct=g("lts__throughput.avg.pct_of_peak_sustained_elapsed"), l2_hit_pct=g("lts__t_sector_hit_rate.pct"),
                   sm_active_pct=g("sm__cycles_active.avg") / g("sm__cycles_elapsed.avg") * 100,
                   active_sms_est=g("sm__cycles_active.avg") / g("sm__cycles_elapsed.avg") * nsm,
                   ctas_launched=g("sm__ctas_launched.sum"),
                   occ_achieved_pct=g("sm__warps_active.avg.pct_of_peak_sustained_active"),
                   occ_theoretical_pct=g("sm__maximum_warps_per_active_cycle_pct"), occ_limiter=limiter,
                   issue_active_pct=g("sm__issue_active.avg.pct_of_peak_sustained_active"),
                   ipc_active=g("sm__inst_executed.avg.per_cycle_active"),
                   tensor_pct_active=g("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
                   top_stalls=";".join(f"{k}={v:.2f}" for v, k in top),
                   **{f"stall_{k}_per_issue": v for k, v in stall.items()})
        out.append(row)
    return out


if __name__ == "__main__":
    rs = rows(sys.argv[1])
    w = csv.DictWriter(sys.stdout, fieldnames=list(rs[0].keys())); w.writeheader(); [w.writerow(x) for x in rs]
