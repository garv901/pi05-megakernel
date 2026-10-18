"""Overlap-weighted non-negative least squares: unmix coarse GPU-metric samples (nsys, several us wide) into per-phase means.

Model: sample j (value y_j, covers [t_j+lag, t_j+lag+dt]) = sum_c x_c * f_jc, f_jc = (time of sample j inside windows of class c) / dt.
Windows of different classes tile time (completion-to-completion / stamp-to-stamp) so sum_c f_jc = 1 inside the run.
x >= 0 solved with coordinate descent on the normal equations (no scipy in the venv)."""
from __future__ import annotations
import numpy as np


def design(t, dt, lag, win_cls, win_a, win_b, ncls, t_lo=None, t_hi=None):
    """Return (A dense [nsamp, ncls], keep mask).  win_*: arrays; samples fully inside [t_lo, t_hi] are kept."""
    lo = t.astype(np.float64) + lag      # nsys: sample timestamp = start of its averaging interval (validated against the ncu PM series, corr 0.978)
    hi = lo + dt
    order = np.argsort(win_a)
    a = np.asarray(win_a, float)[order]; b = np.asarray(win_b, float)[order]; c = np.asarray(win_cls)[order]
    keep = np.ones(len(t), bool)
    if t_lo is not None:
        keep &= lo >= t_lo
    if t_hi is not None:
        keep &= hi <= t_hi
    A = np.zeros((len(t), ncls))
    # for every sample find windows that overlap it (windows are sorted & mostly disjoint)
    i0 = np.searchsorted(b, lo, side="right")      # first window whose end > lo
    i1 = np.searchsorted(a, hi, side="left")       # windows with start < hi
    for j in np.nonzero(keep)[0]:
        for w in range(i0[j], i1[j]):
            ov = min(hi[j], b[w]) - max(lo[j], a[w])
            if ov > 0:
                A[j, c[w]] += ov / dt
    return A, keep


def nnls_cd(A, y, iters=0, tol=1e-12, x0=None):
    """Lawson-Hanson active-set NNLS on the normal equations (exact; n columns are few)."""
    G = A.T @ A; b = A.T @ y
    n = G.shape[0]
    P = np.zeros(n, bool); x = np.zeros(n)
    w = b - G @ x
    for _ in range(3 * n + 50):
        if P.all() or not (w[~P] > tol * max(1.0, np.abs(b).max())).any():
            break
        k = np.argmax(np.where(~P, w, -np.inf)); P[k] = True
        while True:
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(G[np.ix_(P, P)], b[P], rcond=None)[0]
            if (z[P] > 0).all():
                x = z; break
            m = P & (z <= 0)
            alpha = np.min(x[m] / np.maximum(x[m] - z[m], 1e-300))
            x = x + alpha * (z - x)
            P &= x > 1e-12
            x[~P] = 0
        w = b - G @ x
    return x


def solve(t, y, dt, lag, win_cls, win_a, win_b, ncls, t_lo=None, t_hi=None):
    A, keep = design(t, dt, lag, win_cls, win_a, win_b, ncls, t_lo, t_hi)
    cov = A.sum(axis=0)
    used = cov > 0
    x = np.zeros(ncls)
    Au = A[keep][:, used]
    x[used] = nnls_cd(Au, y[keep])
    pred = A[keep] @ x
    res = y[keep] - pred
    ss_res = float((res ** 2).sum()); ss_tot = float(((y[keep] - y[keep].mean()) ** 2).sum())
    return dict(x=x, n=int(keep.sum()), rmse=float(np.sqrt(ss_res / max(keep.sum(), 1))), r2=1 - ss_res / max(ss_tot, 1e-12),
                cond_cols=float(np.linalg.cond(Au)) if Au.shape[1] <= 400 else float("nan"))


def scan_lag(t, y, dt, lags, win_cls, win_a, win_b, ncls, t_lo=None, t_hi=None):
    out = []
    for lg in lags:
        r = solve(t, y, dt, lg, win_cls, win_a, win_b, ncls, t_lo, t_hi)
        out.append((lg, r["rmse"], r["r2"]))
    return out
