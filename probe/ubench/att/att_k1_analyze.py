"""att_k1_analyze.py RUNDIR NAME [NAME ...]: per-variant table for the att_k1 harness runs (RUNDIR/NAME.jsonl stamped, NAME_ns.jsonl unstamped twin,
NAME_mk.jsonl + raw/NAME_mk_<variant>_*.marks.u64 sub-phase marks). A variant with ANY failed per-launch gate (sentinel, work, closure, known_delay,
clock_stable, ordering) or failed set-level gate (clock_cross not counted: caveat only; delta_agreement) is listed as REFUSED and gets no number.
Gate 2 (stamped vs unstamped twin) is the agreement of the two L0 values (< 0.25 us or 5%) plus identical register / stack / local usage."""
import glob, json, os, re, statistics, sys
import numpy as np

d = sys.argv[1]
names = sys.argv[2:] or ["att_k1"]
NIT, NPH, NMARK, SL_IT = 180, 7, 8, 2


def load(p):
    return [json.loads(l) for l in open(p) if l.startswith("{")] if os.path.exists(p) else []


def res(path):
    out = {}
    fn = None
    for l in open(path) if os.path.exists(path) else []:
        m = re.search(r"Function (\S+):", l)
        if m:
            fn = m.group(1)
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", l)
        if m and fn:
            out[fn] = dict(zip(("reg", "stack", "shared", "local"), map(int, m.groups())))
    return out


def variant_table(name):
    recs, ns = load(f"{d}/{name}.jsonl"), load(f"{d}/{name}_ns.jsonl")
    if not recs:
        print(f"[{name}] no records"); return
    gl = [r for r in recs if "gate" in r]
    bad = {}
    for r in gl:
        if r["launch"] >= 0 and not r["pass"]:
            bad.setdefault(r["variant"], set()).add(r["gate"])
    ag = {r["variant"]: r for r in gl if r["gate"] == "delta_agreement"}
    cc = {r["variant"]: r for r in gl if r["gate"] == "clock_cross"}
    for v, r in ag.items():
        if not r["pass"]:
            bad.setdefault(v, set()).add("delta_agreement")
    summ = {r["variant"]: r for r in recs if r.get("rec") == "summary"}
    nsumm = {r["variant"]: r for r in ns if r.get("rec") == "summary"}
    nl = {}
    for r in gl:
        if r["launch"] >= 0:
            nl[r["variant"]] = nl.get(r["variant"], 0) + (r["gate"] == "work")
    ra, rb = res(f"{d}/resusage_{name}.txt"), res(f"{d}/resusage_{name}_ns.txt")
    print(f"== {name}: {len(summ)} variants; reg/stack/local stamped {[ (v['reg'], v['stack'], v['local']) for k, v in ra.items() if 'att_kernel' in k]} "
          f"unstamped {[ (v['reg'], v['stack'], v['local']) for k, v in rb.items() if 'att_kernel' in k]}")
    print(f"{'variant':<20}{'L0 stamped':>11}{'L0 unstamp':>11}{'window dL1':>11}{'L0-L1all':>9}{'clk dev%':>9}{'ev st/ns%':>8}{'launches':>9}  gates")
    nullev = summ["null"]["event_ms_med"] if "null" in summ else None
    nullns = nsumm["null"]["event_ms_med"] if "null" in nsumm else None
    for v in summ:
        if v in ("null", "calib"):
            continue
        L0s = (summ[v]["event_ms_med"] - nullev) * 1e3 / NIT if nullev else float("nan")
        L0n = (nsumm[v]["event_ms_med"] - nullns) * 1e3 / NIT if (v in nsumm and nullns) else float("nan")
        win = ag[v]["val"]["L1_window_us"] if v in ag else float("nan")
        dif = ag[v]["val"]["diff_us"] if v in ag else float("nan")
        cdev = cc[v]["val"]["dev"] * 100 if v in cc else float("nan")
        evp = (summ[v]["event_ms_med"] / nsumm[v]["event_ms_med"] - 1) * 100 if v in nsumm else float("nan")
        g2 = abs(evp) < 1.0  # harness gate 2: stamped vs unstamped event medians agree within 1%
        flag = "ok" if (v not in bad and g2) else ("REFUSED " + ",".join(sorted(bad.get(v, set()))) + ("" if g2 else " gate2"))
        print(f"{v:<20}{L0s:>11.3f}{L0n:>11.3f}{win:>11.3f}{dif:>9.3f}{cdev:>9.2f}{evp:>+8.2f}{nl.get(v, 0):>9}  {flag}")


def timeline(name, variant):
    mf = sorted(glob.glob(f"{d}/raw/{name}_{variant}_*.marks.u64"))
    if not mf:
        return
    recs = load(f"{d}/{name}.jsonl")
    out = []
    for f in mf:
        idx = int(f.rsplit("_", 1)[1].split(".")[0])
        nspc = [r["nspc"] for r in recs if r.get("rec") == "launch" and r["variant"] == variant and r["launch"] == idx][0]
        st = np.fromfile(f.replace(".marks.u64", ".allcta.u64"), dtype=np.uint64).reshape(148, -1).astype(np.int64)
        mk = np.fromfile(f, dtype=np.uint64).reshape(148, NIT, NMARK).astype(np.int64)
        out.append((nspc, st, mk))
    rows = {}
    for nspc, st, mk in out:
        u = lambda x: float(x) * nspc / 1e3
        for it in range(1, NIT):
            sb = SL_IT + it * NPH
            tB, s3 = st[:, sb + 1], st[:, sb + 3]
            m = mk[:, it, :]
            unit, c3 = m[:, 5] > 0, m[:, 6] > 0
            if not unit.any() or not c3.any():
                continue
            def add(k, v): rows.setdefault(k, []).append(v)
            add("unit CTAs", unit.sum()); add("combine CTAs", c3.sum())
            add("barrier exit -> C entry", u((m[:, 0] - tB)[unit].mean()))
            for key, a_, b_ in (("C entry -> inputs landed (Q last)", 0, 1), ("landed -> QK done seen by softmax", 1, 2), ("QK done -> P written (softmax)", 2, 3),
                                ("P written -> PV done", 3, 4), ("PV done -> partials stored + released", 4, 5)):
                ok = unit & (m[:, a_] > 0) & (m[:, b_] > 0)
                if ok.any():
                    dd = (m[:, b_] - m[:, a_])[ok]; add(key + " mean", u(dd.mean())); add(key + " max", u(dd.max()))
            tu = (m[:, 5] - tB)[unit]; add("unit done (release) mean", u(tu.mean())); add("unit done (release) LAST", u(tu.max()))
            w = (m[:, 6] - np.where(unit, m[:, 5], tB))[c3]; add("C3 counter wait mean", u(w.mean())); add("C3 counter wait max", u(w.max()))
            cb = (m[:, 7] - m[:, 6])[c3]; add("C3 combine mean", u(cb.mean())); add("C3 combine max", u(cb.max()))
            add("C3 last combine done (LAST)", u((m[:, 7] - tB)[c3].max()))
            add("C3 end -> D barrier exit (mean)", u((s3 - m[:, 7])[c3].mean()))
            add("window: B->C exit .. C3->D exit (max CTA)", u((s3 - tB).max()))
    print(f"-- {name} / {variant} sub-phase timeline (us from B->C barrier exit; mean over layer iterations x {len(out)} launches; marks perturb codegen)")
    for k, v in rows.items():
        print(f"     {k:<48}{np.mean(v):>9.3f}")


for n in names:
    variant_table(n)
    if os.path.exists(f"{d}/{n}_mk.jsonl"):
        vs = sorted({os.path.basename(f).split("_mk_")[1].rsplit("_", 1)[0] for f in glob.glob(f"{d}/raw/{n}_mk_*.marks.u64")})
        for v in vs:
            timeline(n + "_mk", v)
