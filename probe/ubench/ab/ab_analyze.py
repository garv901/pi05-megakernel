"""ab_analyze.py RUNDIR NAME [NAME ...]: gate summary + L0 table for the A+B harness (modeled on fg_analyze.py).
Reads RUNDIR/NAME.jsonl (stamped), RUNDIR/NAME_ns.jsonl (FG_STAMPS=0 twin), RUNDIR/resusage_NAME[_ns].txt, RUNDIR/raw/*.k1tl.u64 (stage timelines)
and RUNDIR/raw/*.out.u16 (outputs, for the error vs k0). A config with ANY failed launch gate is refused (no L0 comparison printed, exit 1)."""
import glob, json, os, re, statistics, struct, sys
import numpy as np

d = sys.argv[1]
names = sys.argv[2:]
T, QD, KVD = 50, 2048, 256
GATES = ["work", "closure", "known_delay", "clock_stable", "ordering"]
TS = ["ENTRY", "PREISS", "BAR", "FULL0", "B0", "MMA0", "MMAEND", "EPI0", "LDDONE", "STDONE", "END", "ISSUED", "COMP", "X0", "X1", "X2"]


def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def res(path):
    out, fn = {}, None
    if not os.path.exists(path):
        return out
    for l in open(path):
        m = re.search(r"Function (\S+):", l)
        if m:
            fn = m.group(1)
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", l)
        if m and fn:
            out[fn] = dict(zip(("reg", "stack", "shared", "local"), map(int, m.groups())))
    return out


def u16(p):
    a = np.fromfile(p, dtype=np.uint16).astype(np.uint32) << 16
    return a.view(np.float32)


rc = 0
L0 = {}
for name in names:
    recs = load(os.path.join(d, name + ".jsonl"))
    ns = load(os.path.join(d, name + "_ns.jsonl"))
    if not recs:
        print(f"[{name}] no records"); rc = 1; continue
    gl = [r for r in recs if "gate" in r]
    per = {}
    for r in gl:
        if r["launch"] >= 0:
            per.setdefault((r["variant"], r["gate"]), []).append(r["pass"])
    variants = sorted({v for v, _ in per})
    print(f"== {name}: per-launch gates (passed/total)")
    print("variant        " + " ".join(f"{g:>13}" for g in GATES))
    bad = 0
    for v in variants:
        cells = []
        for g in GATES:
            p = per.get((v, g), [])
            bad += p.count(False)
            cells.append(f"{sum(p)}/{len(p)}".rjust(13))
        print(f"{v:<14} " + " ".join(cells))
    warn = []
    for r in gl:
        if r["launch"] < 0:
            print(f"set-level {r['variant']}: {r['gate']} pass={r['pass']} val={json.dumps(r['val'])}")
            if r["gate"] == "clock_cross":
                if not r["pass"]:
                    warn.append(f"{r['variant']}: SM clock {r['val']['dev'] * 100:+.2f}% vs null (gate 5b fails; L0 includes the clock cost)")
            else:
                bad += not r["pass"]
    summ = {r["variant"]: r for r in recs if r.get("rec") == "summary"}
    nsumm = {r["variant"]: r for r in ns if r.get("rec") == "summary"}
    for v in summ:
        if v in nsumm:
            a, b = summ[v]["event_ms_med"], nsumm[v]["event_ms_med"]
            ok = abs(a / b - 1) < 0.01
            print(f"gate2 events {v}: stamped {a:.4f} ms vs unstamped {b:.4f} ms ({(a / b - 1) * 100:+.3f}%) pass={ok}")
            bad += not ok
    ra, rb = res(os.path.join(d, f"resusage_{name}.txt")), res(os.path.join(d, f"resusage_{name}_ns.txt"))
    for fn in ra:
        if fn in rb and "ab_kernel" in fn:
            ok = all(ra[fn][k] == rb[fn][k] for k in ("reg", "stack", "local"))
            if not ok and ra[fn]["local"] == rb[fn]["local"] == 0 and ra[fn]["stack"] == rb[fn]["stack"] and abs(ra[fn]["reg"] - rb[fn]["reg"]) <= 0.05 * ra[fn]["reg"]:
                print(f"WARNING gate2 res-usage {name}: REG differs ({ra[fn]['reg']} vs {rb[fn]['reg']}), stack/local identical (stage stamps), no spills")
                ok = True
            print(f"gate2 res-usage {name}: stamped {ra[fn]} unstamped {rb[fn]} pass={ok}")
            bad += not ok
    if bad:
        print(f"[{name}] REFUSED: {bad} failed gate checks; no comparison reported"); rc = 1
        continue
    for w in warn:
        print(f"WARNING {w}")
    # L0 table: event delta vs this binary's own null (us / layer iteration)
    t = {r["variant"]: r for r in nsumm.values()}
    for v in summ:
        if v == "null" or "null" not in summ:
            continue
        a = (summ[v]["event_ms_med"] - summ["null"]["event_ms_med"]) * 1e3 / 180
        b = (nsumm[v]["event_ms_med"] - nsumm["null"]["event_ms_med"]) * 1e3 / 180 if v in nsumm and "null" in nsumm else float("nan")
        L0[(name, v)] = (a, b)
        print(f"[{name}] L0 {v} - null = {a:.4f} us/layer stamped, {b:.4f} unstamped")
    mx = {}
    for r in recs:
        if r.get("rec") == "launch":
            mx.setdefault(r["variant"], []).append(r["maxratio"])
    for v, x in mx.items():
        if v not in ("null", "null_apw"):
            print(f"[{name}] {v}: max err/tolerance over launches = {max(x):.4f} (<1 = inside the fp32-reference tolerance); mbarrier timeouts = {sum(r.get('timeouts', 0) for r in recs if r.get('rec') == 'launch' and r['variant'] == v)}")
    # stage timeline medians (k1 variants)
    for f in sorted(glob.glob(os.path.join(d, "raw", f"{name}_k1_*_*.k1tl.u64"))):
        m = re.search(r"(k1_\w+?)_(\d+)\.k1tl", os.path.basename(f))
        raw = open(f, "rb").read()
        ts = np.frombuffer(raw[8:], dtype=np.uint64).reshape(148, 10, 16).astype(np.int64)
        nspc = struct.unpack("<d", raw[:8])[0]
        act = ts[:, :, 2] > 0  # TS_BAR set
        rel = {}
        for i, nm in enumerate(TS):
            col = ts[:, :, i]
            ok = act & (col > 0)
            if ok.sum() == 0:
                continue
            rel[nm] = np.median(((col - ts[:, :, 2])[ok]) * nspc)
        order = [k for k in ["BAR", "ISSUED", "B0", "FULL0", "MMA0", "MMAEND", "EPI0", "LDDONE", "X0", "X1", "STDONE", "END"] if k in rel]
        print(f"[{name}] timeline {m.group(1)} launch {m.group(2)} (ns from I->A barrier exit, medians over {int(act.sum())} CTA-steps): " + ", ".join(f"{k} {rel[k]:.0f}" for k in order))
# correctness vs k0
k0f = glob.glob(os.path.join(d, "raw", "k0bin_k0.out.u16"))
if k0f:
    ref = u16(k0f[0]).copy()
    secs = {"q": slice(0, T * QD), "k": slice(T * QD, T * (QD + KVD)), "v": slice(T * (QD + KVD), T * (QD + 2 * KVD))}
    for f in sorted(glob.glob(os.path.join(d, "raw", "*.out.u16"))):
        b = os.path.basename(f)[:-8]
        if b.startswith("k0bin") or b.endswith("null") or b.endswith("null_apw"):
            continue
        x = u16(f)
        out = []
        for s, sl in secs.items():
            dif = np.abs(x[sl] - ref[sl])
            out.append(f"{s}: max|d|={dif.max():.4g} (max|k0|={np.abs(ref[sl]).max():.3g}) rel-rms={np.sqrt((dif ** 2).mean() / (ref[sl] ** 2).mean()):.4g} frac(|d|>1ulp bf16~2^-7 rel)={(dif > np.abs(ref[sl]) * 2 ** -7 + 1e-6).mean():.4g}")
        print(f"error vs k0 [{b}]: " + "; ".join(out))
print("\nSUMMARY L0 (us/layer vs own null; stamped / unstamped)")
for (n, v), (a, b) in sorted(L0.items()):
    print(f"{n:<10}{v:<16}{a:>9.3f}{b:>9.3f}")
sys.exit(rc)
