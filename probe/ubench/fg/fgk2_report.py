"""fgk2_report.py RUNDIR: K2 report. Per variant: per-launch gate pass counts, L0 = median event(variant) - median event(null) in us/layer over the
launches that passed ALL per-launch gates (stamped binary and unstamped twin), paired-by-launch median diff, error, SM clock, regs, set-level gates.
Launches failing a per-launch gate are excluded and counted (known_delay flakes at ~2-7% of launches on the null variant too)."""
import json, sys, statistics as st, collections, re
d = sys.argv[1]
def load(n):
    try: return [json.loads(l) for l in open(f"{d}/{n}.jsonl") if l.startswith("{")]
    except Exception: return []
def analyze(n):
    R = load(n)
    ev = collections.defaultdict(dict); gp = collections.defaultdict(lambda: collections.defaultdict(dict)); err = {}
    for r in R:
        if r.get("rec") == "launch": ev[r["variant"]][r["launch"]] = r["event_ms"]
        elif "gate" in r and r["launch"] >= 0:
            gp[r["variant"]][r["launch"]][r["gate"]] = r["pass"]
            if r["gate"] == "work": err[r["variant"]] = max(err.get(r["variant"], 0), r["val"]["maxerr"])
    setl = [r for r in R if "gate" in r and r["launch"] < 0]
    ok = {v: {l for l in ev[v] if all(gp[v][l].values())} for v in ev}
    return ev, gp, err, setl, ok
def res(n):
    try: t = open(f"{d}/resusage_{n}.txt").read()
    except Exception: return None
    m = re.search(r"Function \S*fg_kernel\S*:\n\s*(REG:\d+ STACK:\d+ SHARED:\d+ LOCAL:\d+)", t)
    return m.group(1) if m else None
ev, gp, err, setl, ok = analyze("fg_k2")
evn, gpn, errn, setln, okn = analyze("fg_k2_ns")
med = st.median
print("variant | L0 us/layer (stamped, gate-passing launches) | unstamped twin | paired-by-launch | launches passing all gates | gate fails (by gate) | maxerr | work-gate")
for v in ev:
    if v in ("null", "calib"): continue
    base = ok["null"]; use = sorted(ok[v] & base)
    l0 = (med([ev[v][l] for l in ok[v]]) - med([ev["null"][l] for l in base])) * 1e3 / 180 if ok[v] and base else float("nan")
    l0n = float("nan")
    if v in evn and okn.get(v) and okn.get("null"): l0n = (med([evn[v][l] for l in okn[v]]) - med([evn["null"][l] for l in okn["null"]])) * 1e3 / 180
    pr = med([ev[v][l] - ev["null"][l] for l in use]) * 1e3 / 180 if use else float("nan")
    fails = collections.Counter(g for l in ev[v] for g, p in gp[v][l].items() if not p)
    print(f"{v} | {l0:.3f} | {l0n:.3f} | {pr:.3f} | {len(ok[v])}/{len(ev[v])} | {dict(fails)} | {err.get(v, float('nan')):.4f}")
print("null passing all gates:", len(ok["null"]), "/", len(ev["null"]), " null fails:", dict(collections.Counter(g for l in ev["null"] for g, p in gp["null"][l].items() if not p)))
print("set-level gates (stamped):")
for r in setl:
    if r["gate"] == "clock_cross": print(f"  {r['variant']}: clock_cross pass={r['pass']} dev={r['val']['dev'] * 100:+.2f}% nspc={r['val']['nspc']:.4f} (null {r['val']['null_nspc']:.4f}) [5b warning only]")
    if r["gate"] == "delta_agreement": print(f"  {r['variant']}: delta_agreement pass={r['pass']} L0={r['val']['L0_us']} L1_all={r['val']['L1_all_us']} diff={r['val']['diff_us']} F={r['val']['F_us']} stage_delta={r['val']['stage_delta_us']}")
print("gate2: res-usage stamped", res("fg_k2"), "| unstamped", res("fg_k2_ns"))
for v in ev:
    if v in evn and ok[v] and okn.get(v):
        a, b = med([ev[v][l] for l in ok[v]]), med([evn[v][l] for l in okn[v]]); print(f"  gate2 events {v}: stamped {a:.4f} ms unstamped {b:.4f} ms ({(a / b - 1) * 100:+.3f}%)")
