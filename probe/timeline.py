"""Fold nsys GPU-metrics samples (10 us) onto the per-layer period -> a ~1 us resolution layer profile."""
import sqlite3
import numpy as np

METRICS = {"gpc_mhz": 0, "gr_active": 2, "sms_active": 3, "sm_issue": 4, "tensor_active": 5, "warps_in_flight_pct": 6,
           "warps_in_flight_avg": 7, "dram_read_pct": 9, "dram_write_pct": 10}


def load_metrics(sqlite_path):
    c = sqlite3.connect(sqlite_path)
    gm = np.array(c.execute("select timestamp,metricId,value from GPU_METRICS").fetchall(), dtype=np.float64)
    out = {}
    for name, mid in METRICS.items():
        m = gm[gm[:, 1] == mid]
        out[name] = (m[:, 0], m[:, 2])
    return out


def fold(metrics, layer_starts_ns, period_us=48.0, bin_us=1.0, lag_ns=0.0):
    """Mean of each metric over samples binned by (t - layer_start) in [0, period)."""
    starts = np.asarray(layer_starts_ns, dtype=np.float64)
    nb = int(period_us / bin_us)
    res = {}
    for name, (t, v) in metrics.items():
        t = t - lag_ns
        idx = np.searchsorted(starts, t, side="right") - 1
        ok = (idx >= 0) & (idx < len(starts))
        rel = (t[ok] - starts[idx[ok]]) / 1e3
        # restrict to samples inside the layer span
        ok2 = rel < period_us
        b = (rel[ok2] / bin_us).astype(int)
        sums = np.bincount(b, weights=v[ok][ok2], minlength=nb)[:nb]
        cnt = np.bincount(b, minlength=nb)[:nb]
        res[name] = (sums / np.maximum(cnt, 1), cnt)
    return res
