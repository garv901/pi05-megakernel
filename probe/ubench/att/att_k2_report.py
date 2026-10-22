"""att_k2_report.py RUNDIR NAME: per-variant sub-phase timelines (us from the B->C barrier exit) from the ATT_MARKS binary NAME_mk (attribution only) plus a
compact L0 table (event delta vs null, stamped / unstamped) with the per-launch gate pass counts."""
import glob, json, os, sys
import numpy as np
d, name = sys.argv[1], sys.argv[2]
NIT, NPH, NMARK, SL_IT = 180, 7, 8, 2
def load(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
recs, ns = load(f"{d}/{name}.jsonl"), load(f"{d}/{name}_ns.jsonl")
summ = {r["variant"]: r for r in recs if r.get("rec") == "summary"}; nsumm = {r["variant"]: r for r in ns if r.get("rec") == "summary"}
gates = {}
for r in recs:
    if "gate" in r and r["launch"] >= 0: gates.setdefault(r["variant"], []).append(r["pass"])
dl = {r["variant"]: r["val"] for r in recs if r.get("gate") == "delta_agreement"}
print(f"{'variant':<22}{'L0 st':>8}{'L0 ns':>8}{'window':>8}{'stage2':>8}{'stage3':>8}  gates")
for v in summ:
    if v in ("null", "calib"): continue
    L = lambda s: (s[v]["event_ms_med"] - s["null"]["event_ms_med"]) * 1e3 / NIT if v in s and "null" in s else float("nan")
    dd = dl.get(v, {}); sd = dd.get("stage_delta_us", [0] * 7)
    print(f"{v:<22}{L(summ):8.3f}{L(nsumm):8.3f}{dd.get('L1_window_us', float('nan')):8.3f}{sd[2]:8.3f}{sd[3]:8.3f}  {sum(gates.get(v, []))}/{len(gates.get(v, []))}")
LAB = ["loads wait", "QK+S store", "softmax", "PV", "opart store+release"]
mksumm = {r["variant"]: r for r in load(f"{d}/{name}_mk.jsonl") if r.get("rec") == "summary"}
for v in mksumm:
    if v in ("null", "calib"): continue
    mf = sorted(glob.glob(f"{d}/raw/{name}_mk_{v}_*.marks.u64"))
    if not mf: continue
    rows = {}
    for f in mf:
        idx = int(f.rsplit("_", 1)[1].split(".")[0])
        nspc = [r["nspc"] for r in load(f"{d}/{name}_mk.jsonl") if r.get("rec") == "launch" and r["variant"] == v and r["launch"] == idx][0]
        st = np.fromfile(f.replace(".marks.u64", ".allcta.u64"), dtype=np.uint64).reshape(148, -1).astype(np.int64)
        mk = np.fromfile(f, dtype=np.uint64).reshape(148, NIT, NMARK).astype(np.int64)
        for it in range(1, NIT):
            sb = SL_IT + it * NPH; tB = st[:, sb + 1]; s3 = st[:, sb + 3]; m = mk[:, it, :]
            unit = m[:, 5] > 0; c3 = m[:, 6] > 0; u = lambda x: x * nspc / 1e3
            def add(k, val): rows.setdefault(k, []).append(val)
            add("units", unit.sum()); add("c3 tasks", c3.sum())
            for i, lab in enumerate(LAB):
                dd = (m[:, i + 1] - m[:, i if i else 0])[unit] if i else (m[:, 1] - tB)[unit]
                add(lab + " mean", u(dd.mean())); add(lab + " max", u(dd.max()))
            add("unit done mean", u((m[:, 5] - tB)[unit].mean())); add("unit done LAST", u((m[:, 5] - tB)[unit].max()))
            w = (m[:, 6] - np.where(unit, m[:, 5], tB))[c3]; add("C3 wait mean", u(w.mean())); add("C3 wait max", u(w.max()))
            cb = (m[:, 7] - m[:, 6])[c3]; add("C3 combine mean", u(cb.mean())); add("C3 combine max", u(cb.max()))
            if name.endswith("umma") and v.startswith("k2"):
                rgm = np.arange(148) % 13
                pl = []
                for rgi in range(13):
                    idx = np.where((rgm == rgi) & unit)[0]
                    if len(idx): pl.append(((m[idx, 6] - tB[idx]) - (m[idx, 5] - tB[idx]).max()).mean())
                add("poll lat after last release", u(np.mean(pl)))
            add("combine done LAST", u((m[:, 7] - tB)[c3].max())); add("C3 end -> D exit mean", u((s3 - m[:, 7])[c3].mean()))
            add("window (max CTA)", u((s3 - tB).max()))
    print(f"== timeline {v} (us from B->C barrier exit; attribution build)")
    for k, val in rows.items(): print(f"  {k:<28}{np.mean(val):9.3f}")
