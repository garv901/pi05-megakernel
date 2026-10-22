"""Per-phase cost breakdown (us/layer) of the CUDA megakernel, default vs -DPI05_FG_UMMA=1, prefix PROBE_PREFIX, k64.
Arms per build: plain (event time), gt (block-0 globaltimer stamps), clk (block-0 clock64 stamps, -DPI05_TS_CLOCK64; ns/cycle
calibrated per build as median(gt span)/median(clk span) from slot 5 to the last stamp), ptrace (-DPI05_PTRACE per-CTA arrive/exit
globaltimer, used for work / skew / exit-latency decomposition only). Measurement only; no kernel change.
Env: PROBE_PREFIX, PROBE_OUT, ROUNDS (12), PER_ROUND (3)."""
import json, os, random, statistics
import numpy as np
import torch
from .common import Recorder, build_mk_arm, pre_flip, l2_flush, make_geo_inputs, out_dir_from_env, setup_torch
from .mk_analyze import analyze
from .phases import PHASES, L, NSTEP, mk_slot, ideal_bytes

ROUNDS, PER = int(os.environ.get("ROUNDS", "12")), int(os.environ.get("PER_ROUND", "3"))
BUILDS = {"default": (), "umma": ("-DPI05_FG_UMMA=1",)}
NB, NCTA = 1700, 148


def timed(arm, fl):
    fl.zero_()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); arm.launch(); b.record(); b.synchronize()
    return a.elapsed_time(b)


def decomp(pt, umma):
    """pt [launch, bar, cta, 2] us. Per phase (F+G merged under umma): interval(block-median), slowest-CTA done, exit latency, skew, work stats."""
    spans = [(p, i, i + 1) for i, p in enumerate(PHASES)]
    if umma:
        spans = [s for s in spans if s[0] not in ("F_gu", "G_act")]
        spans.insert(5, ("FG", 5, 7))
    out = {}
    for name, i0, i1 in spans:
        rows = []
        for li in range(pt.shape[0]):
            for s in range(NSTEP):
                for l in range(L):
                    a = mk_slot(s, l, i0)
                    b = mk_slot(s, l, i1) if i1 < 9 else (mk_slot(s, l + 1, 0) if l < L - 1 else mk_slot(s, l, 9))
                    sx, ea, ex = pt[li, a, :, 1], pt[li, b, :, 0], pt[li, b, :, 1]
                    if (sx < -1e6).any() or (ea < -1e6).any() or (ex < -1e6).any():
                        continue
                    w = ea - sx
                    t0 = np.median(sx)
                    rows.append([np.median(ex) - t0, ea.max() - t0, np.median(ex) - ea.max(), ea.max() - np.median(ea), w.mean(), w.max(),
                                 (w > 0.5 * w.max()).sum()])
        r = np.array(rows).mean(0) if rows else [float("nan")] * 7
        out[name] = dict(zip(["interval", "slowest_done", "exit_latency", "arrive_skew", "mean_work", "max_work", "active_ctas"], map(float, r)))
        out[name]["n"] = len(rows)
    return out


def main():
    out = out_dir_from_env("phase-breakdown")
    rec = Recorder(out)
    setup_torch()
    geo, inputs = make_geo_inputs()
    arms, bufs = {}, {}
    for b, x in BUILDS.items():
        arms[b, "plain"] = build_mk_arm(geo, inputs, "k64", extra=pre_flip(x))
        arms[b, "gt"] = build_mk_arm(geo, inputs, "k64", extra=pre_flip(x))
        arms[b, "clk"] = build_mk_arm(geo, inputs, "k64", extra=pre_flip(x) + ("-DPI05_TS_CLOCK64",))
        print("config", b, arms[b, "clk"].ext.build_config(), flush=True)
        for k in ("gt", "clk"):
            bufs[b, k] = torch.zeros(4096, dtype=torch.int64, device="cuda")
    off = {k: a.no_timing for k, a in arms.items()}
    fl = l2_flush()
    for a in arms.values():
        for _ in range(3):
            a.launch()
    torch.cuda.synchronize()
    t = {k: [] for k in arms}
    st = {k: [] for k in arms if k[1] != "plain"}
    rng = random.Random(7)
    for r in range(ROUNDS):
        order = list(arms)
        rng.shuffle(order)
        for k in order:
            a = arms[k]
            for _ in range(PER):
                if k[1] != "plain":
                    bufs[k].zero_()
                    a.no_timing = bufs[k]
                t[k].append(timed(a, fl))
                if k[1] != "plain":
                    st[k].append(bufs[k].cpu().numpy().copy())
    res = {}
    for b in BUILDS:
        gt, clk = np.stack(st[b, "gt"]).astype(np.int64), np.stack(st[b, "clk"]).astype(np.int64)
        last = mk_slot(NSTEP - 1, L - 1, 9)
        ns_per_cyc = float(np.median(gt[:, last] - gt[:, 5]) / np.median(clk[:, last] - clk[:, 5]))
        per = {}
        for tag, arr, sc in (("gt", gt, 1.0), ("clk", clk, ns_per_cyc)):
            xs = [analyze((s * sc).astype(np.int64)) for s in arr]
            per[tag] = {p: {"mean_over_layers_median_over_launches": float(np.median([x["phases_us_per_layer"][p]["mean"] for x in xs])),
                            "median_layer_median_launch": float(np.median([x["phases_us_per_layer"][p]["median"] for x in xs]))} for p in PHASES}
            per[tag]["tail_us_per_step"] = float(np.median([x["tail_us_per_step"] for x in xs]))
            per[tag]["prologue_us"] = float(np.median([x["prologue_us"] for x in xs]))
            per[tag]["last_stamp_us"] = float(np.median([x["last_stamp_us"] for x in xs]))
        res[b] = {"ns_per_cycle": ns_per_cyc, "stamps": per,
                  "event_ms": {k[1]: statistics.median(v) for k, v in t.items() if k[0] == b}}
        rec("stamps", build=b, **res[b])
    # ptrace arms
    for b, x in BUILDS.items():
        a = build_mk_arm(geo, inputs, "k64", extra=pre_flip(x) + ("-DPI05_PTRACE",))
        buf = torch.zeros(4096 + NB * NCTA * 2, dtype=torch.int64, device="cuda")
        for _ in range(2):
            a.no_timing = off[b, "plain"]; a.launch()
        torch.cuda.synchronize()
        keep, tt = [], []
        a.no_timing = buf
        for _ in range(6):
            buf.zero_()
            tt.append(timed(a, fl))
            keep.append(buf.cpu().numpy().copy())
        arr = np.stack(keep[1:])
        t0 = arr[:, 0:1]
        pt = arr[:, 4096:].reshape(len(arr), NB, NCTA, 2)
        rel = np.where(pt > 0, pt - t0[:, :, None, None], -10**15).astype(np.float64) / 1e3
        nwritten = (rel[:, :, :, 0] > -1e6).all(axis=2)[0]
        per_layer = [int(nwritten[mk_slot(0, l, 0):mk_slot(0, l, 0) + 9].sum()) for l in (3, 4)]
        d = decomp(rel, b == "umma")
        res[b]["ptrace"] = {"traced_event_ms": statistics.median(tt[1:]), "barriers_per_mid_layer_written": per_layer, "phases": d}
        rec("ptrace", build=b, **res[b]["ptrace"])
        del a, buf
    json.dump(res, open(os.path.join(out, "breakdown.json"), "w"), indent=1)
    print("out_dir", out)


main()
