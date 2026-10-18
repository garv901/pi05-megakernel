"""Edge-blur bias of the 1 us PM sampling on the burst metric used in mbu_prefetch_analyze (t5..t95 of cumulative bytes; BW = 0.9*bytes/(t95-t5)).
Synthetic rectangular burst of true duration d and rate R, sampled at 1.0 us (start convention) with random phase. Prints measured/true BW."""
import numpy as np
from .mbu_lib import F_of, t_at_cum, mb
rng = np.random.default_rng(0)
print("true_dur_us  measured_dur_us(mean)  BW_measured/BW_true (mean, p10, p90)")
for d in (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0):
    ratios, durs = [], []
    for _ in range(300):
        ph = rng.uniform(0, 1000); t = np.arange(0, 40000, 1000.0) + ph
        R = 60.0
        edges = np.append(t, t[-1] + 1000.0)
        # sample j integrates the true signal over [t_j, t_j+1000)
        lo, hi = 10000.0, 10000.0 + d * 1e3
        v = np.array([R * max(0.0, min(e + 1000.0, hi) - max(e, lo)) / 1000.0 for e in t])
        F = F_of(t, v); tot = mb(F, t[0], t[-1] + 1000)
        t5 = t_at_cum(F, t[0], t[-1] + 1000, 0.05 * tot); t95 = t_at_cum(F, t[0], t[-1] + 1000, 0.95 * tot)
        dd = (t95 - t5) / 1e3; durs.append(dd); ratios.append(0.9 * d / dd)       # BW_meas = 0.9*bytes/dd ; BW_true = bytes/d
    r = np.array(ratios)
    print(f"{d:5.1f}  {np.mean(durs):6.2f}  {r.mean():.2f} {np.percentile(r,10):.2f} {np.percentile(r,90):.2f}")
