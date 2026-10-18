"""CPU checks of the gemm_phase point markers (PI05_IKET_GEMM=1) on the cubins of two builds of the full kernel (stdlib only).

  python gemm_marks_check.py PLAIN.cubin MARK.cubin [--ptx MARK.ptx] [--plain-ptx PLAIN.ptx] [--out FILE]

Both cubins must come from builds with CUTE_DSL_LINEINFO=1 (nvdisasm -g -c gives file:line per instruction; cuobjdump -res-usage
gives REG / STACK). PLAIN = no PI05_IKET_GEMM (stage-only iket build or plain build), MARK = PI05_IKET_GEMM=1.
(a) per build REG, STACK and the LDL / STL counts grouped by source file:line, the plain-vs-marker diff per line, with the
    load_w 128-bit copy line of gemm.py and the kernel signature line kernel.py:279 called out.
    With --ptx / --plain-ptx (iket builds only: the stage-range PMTRIG ids need the PTX) a second table splits LDL / STL by stage,
    i.e. by the last stage range (pro_*, A .. I, exit) pushed before the instruction in program order (stages are inlined and
    contiguous), so the H spill of the load_w line can be compared on its own.
(b) every SR_GLOBALTIMERLO read of the marker build (its PMTRIG event id; with --ptx the name from `__iket_evt_decl_NAME_ID_attrs`
    and the nearest preceding gemm.py line). Checks (exit 1 on a violation, violations are printed):
      1. every STS that comes from the `st.volatile.shared` line of gmark_dep (iket_marks.py) is followed in the same basic block
         by a timer read; with --ptx every g_wrdy timer read has such an STS before it in its basic block;
      2. every timer read right after a BAR.SYNC has an LDS in between (same rule as sass_check in test_pi05_cute_profile.py).
The line info of the iket instrumentation itself (CS2R timer, LOP3, STS, PMTRIG) carries pseudo line numbers of iket_marks.py,
so the marker identity comes from the PMTRIG id, not from a line."""
import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
NVDISASM = "/usr/local/cuda/bin/nvdisasm"
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
FILE_RE = re.compile(r'//## File "(.+)", line (\d+)')
INSTR_RE = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;?\s*$")
LABEL_RE = re.compile(r"^\.L_x_\d+:")
ENDS_BLOCK = ("BRA", "EXIT", "RET", "BRX", "JMP")


def res_usage(cubin):
    out = subprocess.run([CUOBJDUMP, "-res-usage", cubin], capture_output=True, text=True).stdout
    m = re.search(r"REG:(\d+) STACK:(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def disasm(cubin):
    """[(addr, opcode, operands, 'file.py:line', basic block)] in program order."""
    txt = subprocess.run([NVDISASM, "-g", "-c", cubin], capture_output=True, text=True).stdout
    ins, loc, bb = [], "?", 0
    for ln in txt.splitlines():
        m = FILE_RE.search(ln)
        if m:
            loc = f"{Path(m.group(1)).name}:{m.group(2)}"
            continue
        if LABEL_RE.match(ln):
            bb += 1
            continue
        m = INSTR_RE.match(ln)
        if m:
            addr, op, rest = m.groups()
            ins.append((int(addr, 16), op, rest, loc, bb))
            if op.split(".")[0] in ENDS_BLOCK:
                bb += 1
    return ins


def spills(ins):
    ldl, stl = Counter(), Counter()
    for _, op, _, loc, _ in ins:
        if op.startswith("LDL"):
            ldl[loc] += 1
        elif op.startswith("STL"):
            stl[loc] += 1
    return ldl, stl


STAGES = ("pro_tab", "pro_t1", "pro_t2", "pro_mod", "pro_act", "A", "AB", "B", "C", "D", "E", "F", "G", "H", "I", "exit")


def stage_of(ins, ids):
    """Per instruction index: name of the last stage range pushed before it (program order), or None."""
    out, cur = [], None
    for i, (_, op, rest, _, _) in enumerate(ins):
        if op == "PMTRIG":
            m = re.match(r"0x([0-9a-f]+)", rest)
            if m and ids.get(int(m.group(1), 16)) in STAGES:
                cur = ids[int(m.group(1), 16)]
        out.append(cur)
    return out


def stage_report(plain, mark, pids, mids, gl, lines):
    """LDL / STL by stage (all lines, and the load_w line gl) for builds whose PTX is known."""
    rows = {}
    for label, ins, ids in (("plain", plain, pids), ("marker", mark, mids)):
        if not ids:
            continue
        st = stage_of(ins, ids)
        for (_, op, _, loc, _), s in zip(ins, st):
            if op.startswith(("LDL", "STL")):
                k = (s, op[:3])
                rows.setdefault(label, Counter())[k] += 1
                if loc == gl:
                    rows.setdefault(label + " " + gl, Counter())[k] += 1
    if not rows:
        return
    labs = list(rows)
    lines += ["LDL / STL by stage (last stage range pushed before the instruction; None = before the first stage push):", "",
              "| stage | " + " | ".join(f"{l} LDL | {l} STL" for l in labs) + " |", "|---|" + "---|---|" * len(labs)]
    for s in sorted({k[0] for r in rows.values() for k in r}, key=lambda x: STAGES.index(x) if x in STAGES else -1):
        lines.append(f"| {s} | " + " | ".join(f"{rows[l][(s, 'LDL')]} | {rows[l][(s, 'STL')]}" for l in labs) + " |")
    lines.append("")


def src_text(loc):
    f, n = loc.rsplit(":", 1)
    try:
        return (HERE / f).read_text().splitlines()[int(n) - 1].strip()
    except (OSError, IndexError, ValueError):
        return ""


def spill_report(plain, mark, pids, mids, lines):
    (pl, ps), (ml, ms) = spills(plain), spills(mark)
    rp, sp = res_usage(PLAIN)
    rm, sm = res_usage(MARK)
    lines += ["## (a) registers, stack, LDL / STL", "",
              "| build | REG | STACK | LDL | STL |", "|---|---|---|---|---|",
              f"| plain | {rp} | {sp} | {sum(pl.values())} | {sum(ps.values())} |",
              f"| marker | {rm} | {sm} | {sum(ml.values())} | {sum(ms.values())} |", ""]
    gl = next((f"gemm.py:{i + 1}" for i, t in enumerate((HERE / "gemm.py").read_text().splitlines())
               if "nc_atom(BFloat16, 128), src, dst" in t), "gemm.py:?")
    keys = sorted(set(pl) | set(ps) | set(ml) | set(ms) | {gl, "kernel.py:279"},
                  key=lambda k: (k.rsplit(":", 1)[0], int(k.rsplit(":", 1)[1]) if k.rsplit(":", 1)[1].isdigit() else 0))
    lines += [f"LDL / STL per source line (`*` = load_w 128-bit copy {gl}, `K` = kernel signature kernel.py:279); marker minus plain in the last column:", "",
              "| line | plain LDL | plain STL | marker LDL | marker STL | diff LDL | diff STL | source |", "|---|---|---|---|---|---|---|---|"]
    changed = 0
    for k in keys:
        a, b, c, d = pl[k], ps[k], ml[k], ms[k]
        tag = " *" if k == gl else " K" if k == "kernel.py:279" else ""
        if a or b or c or d or tag:
            changed += (c != a) or (d != b)
            lines.append(f"| {k}{tag} | {a} | {b} | {c} | {d} | {c - a:+d} | {d - b:+d} | `{src_text(k)[:60]}` |")
    lines += ["", f"lines whose LDL or STL count differs: {changed}", ""]
    stage_report(plain, mark, pids, mids, gl, lines)


def read_ids(ptx):
    return {int(i): n for n, i in re.findall(r"__iket_evt_decl_(.+?)_(\d+)_attrs", Path(ptx).read_text())}


def timer_check(mark, ids, lines):
    dep_loc = None
    for i, t in enumerate((HERE / "iket_marks.py").read_text().splitlines()):
        if "st.volatile.shared" in t:
            dep_loc = f"iket_marks.py:{i + 1}"
    timers = []  # (index, id)
    for i, (_, op, rest, _, _) in enumerate(mark):
        if op.startswith("CS2R") and "SR_GLOBALTIMERLO" in rest:
            eid = next((int(m.group(1), 16) for _, o, r, _, _ in mark[i + 1:i + 8]
                        for m in [re.match(r"0x([0-9a-f]+)", r)] if o == "PMTRIG" and m), None)
            timers.append((i, eid))
    viol = []
    # 2. LDS between a BAR.SYNC and the next timer read
    pend = lds = False
    after_bar = 0
    tset = {i for i, _ in timers}
    for i, (_, op, rest, loc, bb) in enumerate(mark):
        if "BAR.SYNC" in op:
            pend, lds = True, False
        elif re.match(r"LDS(\.|$)", op):
            lds = True
        elif i in tset and pend:
            after_bar += 1
            if not lds:
                viol.append(f"timer read at {mark[i][0]:#x} follows a BAR.SYNC without an LDS")
            pend = False
    # 1. gmark_dep STS
    deps = [i for i, (_, op, _, loc, _) in enumerate(mark) if op.startswith("STS") and loc == dep_loc]
    for i in deps:
        nxt = next((j for j, _ in timers if j > i and mark[j][4] == mark[i][4]), None)
        if nxt is None:
            viol.append(f"gmark_dep STS at {mark[i][0]:#x}: no timer read after it in its basic block")
    wrdy = [k for k, n in ids.items() if n == "g_wrdy"]
    n_wrdy = 0
    for i, eid in timers:
        if eid in wrdy:
            n_wrdy += 1
            if not any(j < i and mark[j][4] == mark[i][4] for j in deps):
                viol.append(f"g_wrdy timer read at {mark[i][0]:#x}: no gmark_dep STS before it in its basic block")
    lines += ["## (b) timer reads of the marker build", "",
              f"- SR_GLOBALTIMERLO reads {len(timers)}, right after a BAR.SYNC {after_bar}",
              f"- gmark_dep STS sites ({dep_loc}) {len(deps)}, g_wrdy timer reads {n_wrdy if wrdy else 'n/a (no --ptx)'}", "",
              "| PMTRIG id | name | sites | nearest preceding gemm.py line (first site) |", "|---|---|---|---|"]
    cnt = Counter(e for _, e in timers)
    first = {}
    for i, e in timers:
        if e not in first:
            first[e] = next((mark[j][3] for j in range(i, -1, -1) if mark[j][3].startswith("gemm.py:") and mark[j][4] == mark[i][4]), "-")
    for e in sorted(cnt, key=lambda x: (x is None, x)):
        f = first[e]
        lines.append(f"| {e} | {ids.get(e, '?')} | {cnt[e]} | {f} `{src_text(f)[:50] if f != '-' else ''}` |")
    lines += [""]
    lines += [f"- RESULT: {'PASS' if not viol else 'FAIL'}"] + [f"  - VIOLATION: {v}" for v in viol] + [""]
    return viol


def main(argv=None):
    global PLAIN, MARK
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plain")
    ap.add_argument("mark")
    ap.add_argument("--ptx", help="PTX of the marker build (names the PMTRIG ids)")
    ap.add_argument("--plain-ptx", help="PTX of the plain-side build, only if it is an iket build (stage split of the spills)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    PLAIN, MARK = a.plain, a.mark
    lines = [f"# gemm marks check: plain `{a.plain}`, marker `{a.mark}`", ""]
    plain, mark = disasm(a.plain), disasm(a.mark)
    if not plain or not mark:
        print("no SASS found (nvdisasm failed?)")
        return 1
    mids = read_ids(a.ptx) if a.ptx else {}
    spill_report(plain, mark, read_ids(a.plain_ptx) if a.plain_ptx else {}, mids, lines)
    viol = timer_check(mark, mids, lines)
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
