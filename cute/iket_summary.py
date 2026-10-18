"""Summarise a run-iket trace of the pi0.5 CuTe megakernel (CPU, stdlib only).

  python iket_summary.py RUN_DIR [--out summary.md] [--all] [--layers 18] [--gemm]

Finds iket.decoded_results.json (or iket_pid_*.trace.json) under RUN_DIR, picks the launches whose kernel name contains `pi05`
(default: the last one; --all: every one), and writes: the launches found; CTAs / warps with ranges and with warp lifetimes; the
range count per name; per name the median / p10 / p90 duration (ns) and the total per warp; for the stage ranges, the median
duration per occurrence index (k-th occurrence of `A` in a warp = layer k mod n_layers of step k div n_layers; `pro_*` and `exit`
occur once); and a nesting check of the sync / wpre / cwait ranges against the stage ranges.
Decoded format: top keys stringTable, locationTable; launches[] with kernelName, gridId, ranges[] {rangeNameIdx|rangeName,
warpLocIdxs|warpLocs, startTs, endTs}, warpLifetimes[] {locIdx|warpLocation, startTs, endTs}. Timestamps are %globaltimer ns.
--gemm adds the gemm_phase sub-stage tables from the g_* point markers of a PI05_IKET_GEMM=1 build (decoded `markers[]`
{markerNameIdx, locIdx, timestamp}), see gemm_section. The JSON helpers are copied from megabench/kernels/cute_fx1/parse_iket.py (that file hard-codes the toy geometry)."""
import argparse
import bisect
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

STAGE_NAMES = ("pro_tab", "pro_t1", "pro_t2", "pro_mod", "pro_act", "A", "AB", "B", "C", "D", "E", "F", "G", "H", "I", "exit")
LAYER_STAGES = ("A", "AB", "B", "C", "D", "E", "F", "G", "H", "I")
SYNC_NAMES = ("sync", "wpre", "cwait")
GEMM_NAMES = ("g_0", "g_wiss", "g_adone", "g_sync", "g_wrdy", "g_mma", "g_st", "g_wnext")
GEMM_STAGES = ("A", "D", "F", "H")  # stages that run gemm_phase
GEMM_ROWS = ("g_0->g_wiss", "wpre", "stage_a", "g_adone->g_sync", "g_sync->g_wrdy", "g_wrdy->g_mma", "g_mma->g_st",
             "g_wnext->g_wrdy", "g_sync->end", "call total")


def find_decoded(d):
    d = Path(d)
    return sorted(d.rglob("iket.decoded_results.json")) or sorted(d.rglob("iket_pid_*.trace.json"))


def _s(d, idx_key, key, row):
    if idx_key in row:
        return d.get("stringTable", [])[row[idx_key]]
    return row.get(key)


def _loc(d, row, idx_key, key):
    if idx_key in row:
        return d.get("locationTable", [])[row[idx_key]]
    return row.get(key)


def range_locs(d, r):
    if "warpLocIdxs" in r:
        # run-iket lists a warp-scope range's warp more than once (warpLocIdxs [i, i]); count each warp once
        return [d["locationTable"][i] for i in dict.fromkeys(r["warpLocIdxs"])]
    return r.get("warpLocs", [])


def warp_key(loc):
    return (tuple(loc["ctaId"]), loc["warpId"])


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def per_warp_ranges(d, ln):
    """{warp: [(start, end, name), ...]} sorted by start."""
    out = defaultdict(list)
    for r in ln.get("ranges", []):
        nm = _s(d, "rangeNameIdx", "rangeName", r)
        for loc in range_locs(d, r):
            out[warp_key(loc)].append((r["startTs"], r["endTs"], nm))
    for v in out.values():
        v.sort()
    return out


def nesting(pw):
    """Per warp: stage ranges must be disjoint; every sync-like range is inside one stage range or between stage ranges."""
    inside, between, straddle, stage_overlap = Counter(), Counter(), Counter(), 0
    examples = []
    for w, rs in pw.items():
        stages = [(a, b) for a, b, n in rs if n in STAGE_NAMES]
        stages.sort()
        for i in range(1, len(stages)):
            if stages[i][0] < stages[i - 1][1]:
                stage_overlap += 1
        starts = [a for a, _ in stages]
        for a, b, n in rs:
            if n not in SYNC_NAMES:
                continue
            i = bisect.bisect_right(starts, a) - 1  # last stage starting at or before a
            if i >= 0 and stages[i][0] <= a and b <= stages[i][1]:
                inside[n] += 1
                continue
            # not inside the stage that starts before it: it is between stages iff it ends before the next stage starts
            # and starts after the previous stage ended
            prev_end = stages[i][1] if i >= 0 else None
            nxt = stages[i + 1][0] if i + 1 < len(stages) else None
            if (prev_end is None or a >= prev_end) and (nxt is None or b <= nxt):
                between[n] += 1
            else:
                straddle[n] += 1
                if len(examples) < 5:
                    examples.append((w, n, a, b))
    return inside, between, straddle, stage_overlap, examples


def gemm_calls(d, ln):
    """Split the g_* markers of every warp into gemm_phase calls. Returns (calls, stats); a call is
    {stage, warp, ts: {name: [ts, ...]}, wpre: (start, end) | None, ntiles, last}. stats counts unknown names, markers outside a
    gemm stage range and malformed calls (the marker order must be g_0 g_wiss g_adone g_sync (g_wrdy g_mma g_st [g_wnext])*)."""
    pw = per_warp_ranges(d, ln)
    mk = defaultdict(list)
    st = Counter()
    for m in ln.get("markers", []):
        nm = _s(d, "markerNameIdx", "markerName", m)
        if nm not in GEMM_NAMES:
            st["unknown marker name " + str(nm)] += 1
            continue
        mk[warp_key(_loc(d, m, "locIdx", "warpLoc"))].append((m["timestamp"], nm))
    calls = []
    for w, ms in mk.items():
        ms.sort()
        stages = [(a, b, n) for a, b, n in pw.get(w, []) if n in GEMM_STAGES]
        wpres = [(a, b) for a, b, n in pw.get(w, []) if n == "wpre"]
        starts = [a for a, _, _ in stages]
        cur = {}
        for ts, nm in ms:
            i = bisect.bisect_right(starts, ts) - 1
            if i < 0 or ts > stages[i][1]:
                st["marker outside a gemm stage range"] += 1
                continue
            a, b, sn = stages[i]
            if nm == "g_0" or cur.get("range") != (a, b):
                cur = {"stage": sn, "warp": w, "range": (a, b), "seq": [], "ts": defaultdict(list)}
                calls.append(cur)
            cur["seq"].append(nm)
            cur["ts"][nm].append(ts)
        for c in calls:
            if c["warp"] == w and "wpre" not in c:
                a, b = c["range"]
                c["wpre"] = next(((x, y) for x, y in wpres if a <= x and y <= b), None)
    good = []
    for c in calls:
        seq = c["seq"]
        ok = re.fullmatch(r"g_0 g_wiss g_adone g_sync( g_wrdy g_mma g_st( g_wnext)?)*", " ".join(seq))
        if ok:
            c["ntiles"] = len(c["ts"].get("g_wrdy", []))
            c["last"] = max(max(v) for v in c["ts"].values())
            good.append(c)
        else:
            st["malformed call (" + c["stage"] + ")"] += 1
    return good, st


def gemm_intervals(c):
    """[(row, tile, ns)] of one call; tile is the 1-based tile index (1 for the per-call rows)."""
    t = {k: v for k, v in c["ts"].items()}
    out = [("g_0->g_wiss", 1, t["g_wiss"][0] - t["g_0"][0])]
    if c["wpre"]:
        out.append(("wpre", 1, c["wpre"][1] - c["wpre"][0]))
        sa0 = c["wpre"][1]
    else:
        sa0 = t["g_wiss"][0]
    out += [("stage_a", 1, t["g_adone"][0] - sa0), ("g_adone->g_sync", 1, t["g_sync"][0] - t["g_adone"][0])]
    for k in range(c["ntiles"]):
        wr, mm, sto = t["g_wrdy"][k], t["g_mma"][k], t["g_st"][k]
        if k == 0:
            out.append(("g_sync->g_wrdy", 1, wr - t["g_sync"][0]))
        else:
            out.append(("g_wnext->g_wrdy", k + 1, wr - t["g_wnext"][k - 1]))
        out += [("g_wrdy->g_mma", k + 1, mm - wr), ("g_mma->g_st", k + 1, sto - mm)]
    out.append(("g_sync->end", 1, c["range"][1] - t["g_sync"][0]))  # loop tail up to the stage range pop (all a LOOP=0 build has)
    out.append(("call total", 1, c["last"] - t["g_0"][0]))
    return out


def _cell(xs):
    if not xs:
        return ""
    return f"{statistics.fmean(xs):.0f} / {statistics.median(xs):.0f} / {pct(xs, .9):.0f} ({len(xs)})"


def gemm_section(d, ln, lines):
    """Tables of the g_* point markers: per interval, columns = stage (and tile index t for F), cells `mean / median / p90 (n)` ns;
    then per stage grouped by tiles per warp."""
    calls, st = gemm_calls(d, ln)
    lines += ["#### gemm_phase sub-stages (g_* markers)", "",
              f"- calls (g_0 .. last marker) {len(calls)}: " + ", ".join(f"{s} {sum(c['stage'] == s for c in calls)}" for s in GEMM_STAGES)]
    lines.append("- unmatched / unknown: " + (", ".join(f"{k}: {v}" for k, v in sorted(st.items())) if st else "none"))
    if not calls:
        lines += ["- no gemm calls (was the build made with PI05_IKET_GEMM=1?)", ""]
        return
    cell = defaultdict(list)
    maxt = {}
    for c in calls:
        for row, tile, ns in gemm_intervals(c):
            cell[(row, c["stage"], tile)].append(ns)
            maxt[c["stage"]] = max(maxt.get(c["stage"], 0), tile)
    cols = [(s, k) for s in GEMM_STAGES if s in maxt for k in range(1, maxt[s] + 1)]
    lab = lambda s, k: s if maxt[s] == 1 else f"{s} t{k}"
    lines += ["", "Intervals in ns, mean / median / p90 (n); per-call rows sit in the t1 column:", "",
              "| interval | " + " | ".join(lab(*x) for x in cols) + " |", "|---|" + "---|" * len(cols)]
    for row in GEMM_ROWS:
        lines.append(f"| {row} | " + " | ".join(_cell(cell.get((row, s, k), [])) for s, k in cols) + " |")
    for s in GEMM_STAGES:
        if s not in maxt:
            continue
        by = defaultdict(lambda: defaultdict(list))
        for c in calls:
            if c["stage"] == s:
                for row, tile, ns in gemm_intervals(c):
                    by[c["ntiles"]][row].append(ns)
        nts = sorted(by)
        lines += ["", f"Stage {s} by tiles per warp (tile rows pooled over the tiles):", "",
                  "| interval | " + " | ".join(f"{n} tiles" for n in nts) + " |", "|---|" + "---|" * len(nts)]
        for row in GEMM_ROWS:
            lines.append(f"| {row} | " + " | ".join(_cell(by[n].get(row, [])) for n in nts) + " |")
    lines.append("")


def summarize_launch(d, ln, n_layers, lines, gemm=False):
    pw = per_warp_ranges(d, ln)
    wl = ln.get("warpLifetimes", [])
    life_warps = {warp_key(_loc(d, w, "locIdx", "warpLocation")) for w in wl}
    life_ctas = {k[0] for k in life_warps}
    r_ctas = {k[0] for k in pw}
    lines += [f"### launch gridId {ln.get('gridId')}: `{ln.get('kernelName')}`", "",
              f"- ranges {len(ln.get('ranges', []))} records; CTAs with ranges {len(r_ctas)}, warps with ranges {len(pw)}",
              f"- warp lifetimes {len(wl)}: CTAs {len(life_ctas)}, warps {len(life_warps)}"]
    if wl:
        lt = [w["endTs"] - w["startTs"] for w in wl]
        lines.append(f"- warp lifetime (ns): median {statistics.median(lt):.0f}, p10 {pct(lt, .1):.0f}, p90 {pct(lt, .9):.0f}")
    dur, cnt, tot = defaultdict(list), Counter(), defaultdict(lambda: defaultdict(int))
    for w, rs in pw.items():
        for a, b, n in rs:
            dur[n].append(b - a)
            cnt[n] += 1
            tot[n][w] += b - a
    names = [n for n in STAGE_NAMES + SYNC_NAMES if n in cnt] + sorted(n for n in cnt if n not in STAGE_NAMES + SYNC_NAMES)
    lines += ["", "| name | ranges | median ns | p10 ns | p90 ns | total per warp, median ns | total per warp, max ns |", "|---|---|---|---|---|---|---|"]
    for n in names:
        x, t = dur[n], list(tot[n].values())
        lines.append(f"| {n} | {cnt[n]} | {statistics.median(x):.0f} | {pct(x, .1):.0f} | {pct(x, .9):.0f} | "
                     f"{statistics.median(t):.0f} | {max(t)} |")
    # occurrence index of the stage ranges
    occ = defaultdict(lambda: defaultdict(list))
    for w, rs in pw.items():
        seen = Counter()
        for a, b, n in rs:
            if n in STAGE_NAMES:
                occ[n][seen[n]].append(b - a)
                seen[n] += 1
    cols = [n for n in LAYER_STAGES if n in occ]
    if cols:
        kmax = max(len(occ[n]) for n in cols)
        lines += ["", f"Median duration (ns) of the k-th occurrence per warp (layer = k mod {n_layers}, step = k div {n_layers}):", "",
                  "| k | step | layer | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
        for k in range(kmax):
            cells = [f"{statistics.median(occ[n][k]):.0f}" if k in occ[n] else "" for n in cols]
            lines.append(f"| {k} | {k // n_layers} | {k % n_layers} | " + " | ".join(cells) + " |")
    once = [n for n in STAGE_NAMES if n in occ and n not in LAYER_STAGES]
    if once:
        lines += ["", "Once-per-launch stages (median ns): " + ", ".join(f"{n} {statistics.median(occ[n][0]):.0f}" for n in once
                                                                         if 0 in occ[n])]
    inside, between, straddle, ov, ex = nesting(pw)
    lines += ["", "Nesting check (sync / wpre / cwait against the stage ranges, per warp):", "",
              f"- stage ranges overlapping another stage range: {ov}"]
    for n in SYNC_NAMES:
        if n in cnt:
            lines.append(f"- {n}: inside a stage {inside[n]}, between stages {between[n]}, straddling a stage edge {straddle[n]}")
    lines.append(f"- RESULT: {'PASS' if not sum(straddle.values()) and not ov else 'FAIL'}")
    for e in ex:
        lines.append(f"  - straddle example: warp {e[0]} {e[1]} [{e[2]}, {e[3]}]")
    lines.append("")
    if gemm:
        gemm_section(d, ln, lines)


def summarize(run_dir, n_layers=18, all_launches=False, gemm=False):
    files = find_decoded(run_dir)
    lines = [f"# IKET summary: {run_dir}", ""]
    if not files:
        return lines + ["NO decoded trace found (iket.decoded_results.json / iket_pid_*.trace.json)."], False
    path = files[-1]
    d = json.loads(path.read_text())
    lines += [f"decoded file: `{path}` ({len(files)} found, last used)", "", "Launches in the trace:", ""]
    pi = []
    for i, ln in enumerate(d.get("launches", [])):
        nm = ln.get("kernelName") or ""
        lines.append(f"- [{i}] gridId {ln.get('gridId')} ranges {len(ln.get('ranges', []))} lifetimes {len(ln.get('warpLifetimes', []))} `{nm}`")
        if "pi05" in nm:
            pi.append(ln)
    lines.append("")
    if not pi:
        return lines + ["NO launch of the pi05 kernel in the trace."], False
    for ln in (pi if all_launches else pi[-1:]):
        summarize_launch(d, ln, n_layers, lines, gemm)
    return lines, True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--all", action="store_true", help="summarise every pi05 launch, not only the last")
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--gemm", action="store_true", help="add the gemm_phase sub-stage tables (PI05_IKET_GEMM=1 traces)")
    args = ap.parse_args(argv)
    lines, ok = summarize(args.run_dir, args.layers, args.all, args.gemm)
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
