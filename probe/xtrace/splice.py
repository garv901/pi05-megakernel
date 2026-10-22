"""Xtrace (gwatch) SASS splice of a pi05 megakernel cubin, CPU only.

Warp-level probe sites around every grid barrier code site: the opening BAR.SYNC (insert before it = the warp arrives at
the barrier) and the instruction after the closing BAR.SYNC (= the warp leaves it). A grid barrier is found from its
release add: the unpredicated ATOMG of grid_barrier / wpre_arrive (thread 0's branch); CuTe's exit-handshake atomics are
predicated and skipped. The splice is gwatch's TraceInstrumenter_SASS; the kernel's own instructions are untouched.
With gemm=True, also five sites inside gemm_phase (gemm_sites; QKV, gate/up and down GEMMs; MB2 / MB3).
CLI: splice.py IN.cubin OUT_DIR [clock=gpu|anchor] [records=N] [keep TAG] [gemm]
  -> OUT_DIR/xtrace_<clock>[_TAG][_gemm].{cubin,json}
keep: comma list of "b<N>" / "b<A>-<B>" (both sites of barrier N / barriers A..B; suffix a|e for one kind, e.g. "b11a")
and "arrive" / "exit" (that kind at every barrier)."""
import difflib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

import gwatch.cuda.binary as gb
from gwatch.cuda.xtrace._schema import SassTraceSite, TraceSchema_SASS
from gwatch.cuda.xtrace._sass_instrumenter import TraceInstrumenter_SASS

GRID, BLOCK = (148, 1, 1), (256, 1, 1)
RECORDS = 16384  # per-warp record budget (a warp writes ~3.5k: 2 per dynamic grid barrier); overflow corrupts neighbors


class NoBuffer:
    """Stands in for the device buffer during the splice (only collect() reads a buffer)."""
    data_ptr = 0
    device = 0

    def __init__(self, raw=None):
        self.raw = raw

    def to_numpy(self):
        return self.raw


def opcode(ins):
    t = ins.str(flatten=True).split()
    return t[1] if t and t[0].startswith("@") else (t[0] if t else "")


def barrier_sites(kd):
    """[(arrive_index, exit_index)] per grid-barrier code site."""
    ops = [opcode(i) for i in kd.list_instructions]
    out = []
    for a, op in enumerate(ops):
        if not op.startswith("ATOMG") or "Pg=0x7 " not in kd.list_instructions[a].str(flatten=True):
            continue
        o = max(i for i in range(a) if ops[i] == "BAR")
        c = min(i for i in range(a + 1, len(ops)) if ops[i] == "BAR")
        out.append((o, c + 1))
    return out


def site_table(kd, keep=None, gemm=None):
    """keep: optional set of (barrier, kind) to instrument (bisecting a faulting probe); default all. gemm: gemm_sites
    indices to add after the barrier sites (barrier -1, kind from GEMM_KINDS)."""
    sites, table = [], []
    for b, (arr, ext) in enumerate(barrier_sites(kd)):
        for kind, idx in (("arrive", arr), ("exit", ext)):
            if keep is not None and (b, kind) not in keep:
                continue
            table.append({"site_id": len(sites), "barrier": b, "kind": kind, "insert_index": idx,
                          "instr": kd.list_instructions[idx].str(flatten=True)[:80]})
            sites.append(SassTraceSite(idx, warp_level=True))
    for kind, idx in zip(GEMM_KINDS, gemm or []):
        table.append({"site_id": len(sites), "barrier": -1, "kind": kind, "insert_index": idx,
                      "instr": kd.list_instructions[idx].str(flatten=True)[:80]})
        sites.append(SassTraceSite(idx, warp_level=True))
    return sites, table


GEMM_KINDS = ("g_async", "g_tile", "g_m0", "g_wnext", "g_end")


def gemm_sites(image):
    """gemm_phase probe indices from the unspliced cubin (whose function symbols gwatch's image drops), in GEMM_KINDS
    order: before the A-tile CTA sync (the warp has staged its A pieces), the tile-loop head (back-edge target; tile
    start), after the loop's first HMMA (it issues once its first weight and A fragments have landed: weights ready),
    before the first weight LDG after it (next tile's load issue, LOAD_HOIST 0: after the partial stores, 1: before),
    before the closing unguarded RET (the warp leaves gemm_phase; warps with no tile leave by the early guarded RET)."""
    rows, labels = _sass(image)
    funcs = sorted(pc for name, pc in labels.items() if name.startswith("$"))
    f0 = next(pc for name, pc in labels.items() if "_Z10gemm_phase" in name)
    f1 = min([pc for pc in funcs if pc > f0] + [rows[-1][0] + 16])
    body = [(pc, t) for pc, t, _, _ in rows if f0 <= pc < f1]
    hm = next(pc for pc, t in body if t.startswith("HMMA"))
    bar = max(pc for pc, t in body if pc < hm and t.startswith("BAR.SYNC"))
    end = max(pc for pc, t in body if t.startswith("RET"))  # the out-of-line divergent-shuffle paths follow it
    back = [(pc, _target(t, labels)) for pc, t in body if hm < pc < end and re.match(r"(@!?P\d )?BRA ", t)
            and f0 <= (_target(t, labels) or f1) < hm]
    assert len(back) == 1, back
    head = back[0][1]
    assert bar < head < hm, (hex(bar), hex(head), hex(hm))
    wnext = next(pc for pc, t in body if pc > hm and "LDG.E.128.CONSTANT" in t)
    pcs = (bar, head, hm + 16, wnext, end)
    return [pc // 16 for pc in pcs], {k: hex(pc) for k, pc in zip(GEMM_KINDS, pcs)}


def make_schema(sites, clock, records=RECORDS):
    return TraceSchema_SASS(sites=sites, grid_dim=GRID, block_dim=BLOCK, clock_type=clock, per_thread_records=records)


def splice(cubin, clock, records=RECORDS, keep=None, gemm=False):
    """Splice a parsed Cubin in place; returns (kernel name, instrumented image bytes, schema, instrumenter, meta)."""
    kernel = list(cubin.get_map_kernel_def())[0]
    kd = cubin.get_kerneldef_by_name(kernel)
    kd.parse_cfg()
    kd.parse_register_liveness()
    n_instr = len(kd.list_instructions)
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        cubin.dump(f.name)
        orig_image = open(f.name, "rb").read()
    gidx, gpcs = gemm_sites(orig_image) if gemm else (None, None)
    sites, table = site_table(kd, keep, gidx)
    schema = make_schema(sites, clock, records)
    ser0 = gb.Cubin()
    ser0.add_kerneldef(kd)
    plain_image = ser0.serialize()  # the same serializer on the untouched kernel: the reference for fix_relocations
    c4 = sorted({off for _, text, _, _ in _sass(plain_image)[0] for off in _c4_offsets(text)})
    for _ in c4:  # one 8-byte parameter per bank-4 slot the kernel reads, ahead of gwatch's trace pointer (kept last)
        cubin._gw_instance.add_argument(kernel, 8)
    cubin.parse()
    inst = TraceInstrumenter_SASS(kernel_prototype=kernel, kernel_def_sass=cubin.get_kerneldef_by_name(kernel),
                                  cubin=cubin, trace_buffer=NoBuffer(), trace_schema=schema, clock_type=clock)
    inst.instrument()
    ser = gb.Cubin()
    ser.add_kerneldef(inst.kernel_def_sass)
    image, fixes = fix_relocations(plain_image, ser.serialize(), kernel)
    offsets = list(reversed(inst.kernel_def_sass.list_param_offsets_reversed))
    n_params = len(offsets) - len(c4) - 1  # the kernel's own parameters, then ours, then the trace pointer
    assert n_params >= 1, (offsets, c4)
    image, bank4 = retarget_bank4(orig_image, image, kernel, c4, offsets[n_params:n_params + len(c4)])
    probe_pcs = {str(k): v for k, v in inst.map_site_probe_pcs.items()}
    image, renamed = rename_probe_scratch(plain_image, image, kernel, {"probe_pcs": probe_pcs, "sites": table})
    meta = {"kernel": kernel, "n_instr": n_instr, "clock": clock, "records": records, "sites": table,
            "warp_slab_bytes": schema.warp_slab_bytes, "total_bytes": schema.total_bytes, "num_warps": schema.num_warps,
            "header_bytes": schema.warp_header_bytes, "record_bytes": schema.warp_record_bytes,
            "site_id_bits": schema.warp_site_id_bits,
            "probe_pcs": probe_pcs, "renamed_probes": renamed,
            "reports": {str(k): v for k, v in inst.map_site_reports.items()}, "relocation_fixes": fixes,
            "bank4": bank4, "gemm_pcs": gpcs}
    return kernel, image, schema, inst, meta


PARAM_CBANK = 0x380  # sm_100: kernel parameters start at c[0x0][0x380]


def _c4_offsets(text):
    return [0 if m.group(1) == "RZ" else int(m.group(1), 16) for m in re.finditer(r"c\[0x4\]\[(0x[0-9a-f]+|RZ)\]", text)]


def _elf(image):
    """(sections [(name, type, offset, size)], symbols [(shndx, value)]) of an ELF64 image."""
    shoff, = struct.unpack_from("<Q", image, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", image, 0x3A)
    raw = [struct.unpack_from("<IIQQQQIIQQ", image, shoff + i * shentsize) for i in range(shnum)]
    stroff = raw[shstrndx][4]
    secs = [(image[stroff + r[0]:image.index(b"\0", stroff + r[0])].decode(), r[1], r[4], r[5]) for r in raw]
    sym = next(s for s in secs if s[0] == ".symtab")
    syms = [struct.unpack_from("<IBBHQQ", image, sym[2] + i * 24) for i in range(sym[3] // 24)]
    return secs, [(s[3], s[4]) for s in syms]


def retarget_bank4(orig_image, image, kernel, c4, param_offsets):
    """gwatch's image keeps only the kernel's own sections, so constant bank 4 (addresses of module globals: the
    cudart sin/cos and 2/pi tables, filled by .rela.nv.constant4) is gone and those loads read 0. Re-encode every
    LDC / LDCU c[0x4][off] as c[0x0][<the parameter added for off>]; the launcher uploads `init` (.nv.global.init,
    then .nv.global zeroed) and passes base + init_off for each slot, in `slots` order before the trace pointer."""
    secs, syms = _elf(orig_image)
    blob, where = b"", {}
    for i, (name, typ, off, size) in enumerate(secs):
        if name in (".nv.global.init", ".nv.global"):
            where[i] = len(blob)
            blob += orig_image[off:off + size] if typ != 8 else bytes(size)  # 8 = SHT_NOBITS
    rela = next(s for s in secs if s[0] == ".rela.nv.constant4")
    target = {}
    for k in range(rela[3] // 24):
        r_off, r_info, r_add = struct.unpack_from("<QQq", orig_image, rela[2] + k * 24)
        shndx, value = syms[r_info >> 32]
        target[r_off] = where[shndx] + value + r_add
    slots = [{"c4_off": o, "init_off": target[o], "cbank_off": PARAM_CBANK + p} for o, p in zip(c4, param_offsets)]
    cbank = {s["c4_off"]: s["cbank_off"] for s in slots}
    off, _ = _text(image, kernel)
    buf = bytearray(image)
    n = 0
    for pc, text, w0, _ in _sass(image)[0]:
        for o in _c4_offsets(text):
            # constant operand: bank at w0[54:59], byte offset ending at bit 54 from bit 38 (LDC) or 37 (LDCU)
            # (fitted on the shipped kernel's loads; checked below by disassembling the result)
            lo = 37 if "LDCU" in text else 38
            assert (w0 >> 54) & 0x1F == 4 and (w0 >> lo) & ((1 << (54 - lo)) - 1) == o, (hex(pc), text)
            w0 = (w0 & ~(((1 << (59 - lo)) - 1) << lo)) | (cbank[o] << lo)
            struct.pack_into("<Q", buf, off + pc, w0)
            n += 1
    image = bytes(buf)
    rows = _sass(image)[0]
    assert not [t for _, t, _, _ in rows if "c[0x4]" in t]
    # the added parameters lie past the original ones, so every load of them is a retargeted one
    assert n == sum(t.count(f"c[0x0][{hex(c)}]") for _, t, _, _ in rows for c in cbank.values()), n
    return image, {"slots": slots, "patched": n, "init_hex": blob.hex()}


def _text(image, kernel):
    """(file offset, size) of the kernel's .text section in an ELF64 cubin."""
    shoff, = struct.unpack_from("<Q", image, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", image, 0x3A)
    sec = [struct.unpack_from("<IIQQQQ", image, shoff + i * shentsize) for i in range(shnum)]
    stroff = sec[shstrndx][4]
    for name, _, _, _, off, size in sec:
        end = image.index(b"\0", stroff + name)
        if image[stroff + name:end].decode() == f".text.{kernel}":
            return off, size
    raise KeyError(kernel)


def _sass(image):
    """(rows [(pc, text, w0, w1)], labels {name: pc}) from nvdisasm -hex (a spliced image holds only the kernel)."""
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(image)
        f.flush()
        lines = subprocess.run(["nvdisasm", "-c", "-hex", f.name], check=True, capture_output=True,
                               text=True).stdout.splitlines()
    rows, labels, pending = [], {}, []
    for k, line in enumerate(lines):
        m = re.match(r"\s*([.$\w]+):\s*$", line)
        if m:
            pending.append(m.group(1))
            continue
        m = re.search(r"/\*([0-9a-f]{4,})\*/\s+(.*?);\s*/\* 0x([0-9a-f]{16}) \*/", line)
        if m:
            pc = int(m.group(1), 16)
            labels.update((p, pc) for p in pending)
            pending = []
            rows.append((pc, m.group(2).strip(), int(m.group(3), 16),
                         int(re.search(r"0x([0-9a-f]{16})", lines[k + 1]).group(1), 16)))
    return rows, labels


def _target(text, labels):
    m = re.search(r"`\(([^)]*)\)", text)
    return labels.get(m.group(1)) if m else None


def _norm(text):
    return re.sub(r"`\([^)]*\)", "T", re.sub(r"0x[0-9a-f]+", "I", text))


def _encode_rel(w0, w1, rel):
    """CALL.REL / RET.REL offset in bytes from the next instruction: R = rel >> 2 is split over w0[16:24], w0[34:64]
    and w1[0:18] (fitted on every branch of the shipped kernel)."""
    r = (rel >> 2) & ((1 << 56) - 1)
    w0 = (w0 & ~(0xFF << 16) & ((1 << 34) - 1)) | ((r & 0xFF) << 16) | (((r >> 8) & ((1 << 30) - 1)) << 34)
    w1 = (w1 & ~0x3FFFF) | ((r >> 38) & 0x3FFFF)
    return w0, w1


def _branch_errors(orig, new, mp):
    """Branch-like instructions of `orig` whose spliced copy does not target the image of the original target, or the
    start of a probe inserted right before it (a probe at a branch target is entered by the branch)."""
    (a, la), (b, lb) = orig, new
    at = {r[0]: r for r in b}
    kept = set(mp.values())

    def ok(got, want):
        return got is not None and want is not None and got <= want and \
            all(pc not in kept for pc in range(got, want, 16))
    return [(pc, mp[pc]) for pc, text, _, _ in a
            if (t := _target(text, la)) is not None and not ok(_target(at[mp[pc]][1], lb), mp.get(t))]


_NO_DEST = ("ST", "RED", "BRA", "BAR", "EXIT", "RET", "CALL", "BSYNC", "BSSY", "WARPSYNC", "MEMBAR", "ERRBAR",
            "CGAERRBAR", "CCTL", "YIELD", "NOP", "FENCE", "BPT", "DEPBAR")


def _rw(text):
    """(guard, opcode, regs read, regs written unconditionally) of one instruction, over R0..R254 and P0..P6 as bit
    indices (R k = k, P k = 256 + k). Conservative for liveness: any register an operand names is read with the three
    after it (a .64 / .128 / MMA fragment names only its first), and only an unguarded single destination is killed."""
    m = re.match(r"@(!?)(U?P\w+)\s+(.*)", text)
    guard, text = (m.group(2), m.group(3)) if m else (None, text)
    op, _, rest = text.partition(" ")
    ops = [o.strip() for o in rest.split(",")] if rest else []
    nd = 0
    if ops and not op.startswith(_NO_DEST):
        nd = 2 if re.match(r"[IFDH]SETP|PLOP3|R2P|VOTE", op) or (op.startswith("LOP3") and ops[0].startswith("P")) else 1
        while nd < len(ops) and re.fullmatch(r"!?P[T\d]", ops[nd]) and not re.match(r"[IFDH]SETP|PLOP3|F?SEL", op):
            nd += 1  # carry-out predicates (IADD3 R, P0, P1, ...)

    def bits(s, wide):
        out = 0
        for p, k in re.findall(r"(?<![\w.])!?(U?[RP])(\d+)", s):
            if p == "R" and int(k) < 255:
                out |= (0xF if wide else 1) << int(k)
            elif p == "P" and int(k) < 7:
                out |= 1 << (256 + int(k))
        return out & ((1 << 263) - 1)
    reads = bits(", ".join(ops[nd:]), True) | (bits(guard, False) if guard and not guard.startswith("U") else 0)
    if op.startswith(("ST", "RED", "ATOM")) and ops:
        reads |= bits(ops[0], True)
    writes = 0 if guard else bits(", ".join(o for o in ops[:nd] if not re.search(r"\.(64|128)\b", o)), False)
    return guard, op, reads, writes


def caller_live(rows, labels):
    """Registers live before each instruction (pc -> bitset), interprocedurally: a CALL.REL flows into its callee (which
    may read the caller's registers) and past it, a RET to the instruction after every CALL of its own function. gwatch's
    liveness stops at RET, so inside a subroutine it treats registers the caller still holds across the call as free:
    a probe in gemm_phase then clobbers the caller's state (hang / illegal address at the grid barriers inside it)."""
    pcs = [r[0] for r in rows]
    at = {pc: k for k, pc in enumerate(pcs)}
    callers = {}  # subroutine entry -> return sites (gwatch's image drops local function symbols: entries = CALL targets)
    for pc, text, _, _ in rows:
        if "CALL.REL" in text:
            callers.setdefault(_target(text, labels), []).append(pc + 16)
    funcs = sorted(callers)

    def func_of(pc):
        own = [f for f in funcs if f <= pc]
        return own[-1] if own else None
    info, succ = [], []
    for pc, text, _, _ in rows:
        guard, op, reads, writes = _rw(text)
        info.append((reads, writes))
        nxt = [pc + 16] if pc + 16 in at else []
        if op.startswith("RET"):
            s = callers.get(func_of(pc), []) + (nxt if guard else [])
        elif op.startswith("EXIT"):
            s = nxt if guard else []
        elif op.startswith(("BRA", "CALL", "BSSY")) and (t := _target(text, labels)) is not None:
            s = [t] + (nxt if guard or not op.startswith("BRA") else [])
        else:
            s = nxt
        succ.append([at[x] for x in s if x in at])
    live = [0] * len(rows)
    changed = True
    while changed:
        changed = False
        for k in range(len(rows) - 1, -1, -1):
            out = 0
            for j in succ[k]:
                out |= live[j]
            new = info[k][0] | (out & ~info[k][1])
            if new != live[k]:
                live[k], changed = new, True
    return {pc: live[k] for k, pc in enumerate(pcs)}


_FIELDS = {"R": (16, 24, 32), "P": (12, 81)}  # Rd / Ra / Rb (STG data); guard / ELECT's destination (sm_100, fitted)


def _clobbers(text, live):
    _, _, _, w = _rw(text)
    return [f"R{x}" for x in range(255) if w >> x & 1 and live >> x & 1] + \
        [f"P{x}" for x in range(7) if w >> (256 + x) & 1 and live >> (256 + x) & 1]


def rename_probe_scratch(plain_image, image, kernel, meta):
    """gwatch picks each probe's scratch registers from its own liveness, which ends at RET; and it ignores
    exclude_instrument_registers. Re-encode every probe's scratch GPRs / predicates to ones dead at its site by
    caller_live, check the result by disassembly, and assert no probe instruction writes a live register."""
    rows, labels = _sass(plain_image)
    live = caller_live(rows, labels)
    entries = [_target(t, labels) for _, t, _, _ in rows if "CALL.REL" in t]
    first_sub = min(entries) if entries else float("inf")  # subroutines follow the kernel body (ptxas layout)

    def in_sub(sid):  # only there does gwatch's RET-bounded liveness go wrong; the body's is exact (and ours coarser)
        return sid >= 0 and meta["sites"][sid]["insert_index"] * 16 >= first_sub
    off, _ = _text(image, kernel)
    buf = bytearray(image)
    new = {r[0]: r for r in _sass(image)[0]}
    sites = {int(k): pcs for k, pcs in meta["probe_pcs"].items()}
    expect, renamed = {}, 0
    for sid, pcs in sites.items():
        if not in_sub(sid):
            continue
        L = live[meta["sites"][sid]["insert_index"] * 16]
        if not any(_clobbers(new[p][1], L) for p in pcs):
            continue
        used = {t for p in pcs for t in re.findall(r"(?<![\w.])(R\d+|P\d)\b", new[p][1])}
        written = 0
        for p in pcs:
            written |= _rw(new[p][1])[3]
        temps = [f"R{x}" for x in range(255) if written >> x & 1] + [f"P{x}" for x in range(7) if written >> (256 + x) & 1]
        free = {"R": [x for x in range(251, -1, -1) if not L >> x & 1 and x > 1 and f"R{x}" not in used],
                "P": [x for x in range(6, -1, -1) if not L >> (256 + x) & 1 and f"P{x}" not in used]}
        mp = {t: f"{t[0]}{free[t[0]].pop(0)}" for t in temps}
        for p in pcs:
            pc, text, w0, w1 = new[p]
            w = w0 | (w1 << 64)
            for t, u in mp.items():
                for b in _FIELDS[t[0]]:
                    width = 8 if t[0] == "R" else 3
                    if b == 81 and not text.startswith("ELECT") or b == 12 and not text.startswith("@"):
                        continue
                    if (w >> b) & ((1 << width) - 1) == int(t[1:]):
                        w = (w & ~(((1 << width) - 1) << b)) | (int(u[1:]) << b)
            struct.pack_into("<QQ", buf, off + pc, w & (2**64 - 1), w >> 64)
            expect[pc] = re.sub(r"(?<![\w.])(R\d+|P\d)\b", lambda m: mp.get(m.group(1), m.group(1)), text)
        renamed += 1
    image = bytes(buf)
    got = {r[0]: r[1] for r in _sass(image)[0]}
    wrong = [(hex(pc), want, got[pc]) for pc, want in expect.items() if got[pc] != want]
    assert not wrong, wrong[:5]
    bad = [(sid, got[p], c) for sid, pcs in sites.items() if in_sub(sid)
           for p in pcs if (c := _clobbers(got[p], live[meta["sites"][sid]["insert_index"] * 16]))]
    assert not bad, bad[:5]
    return image, renamed


def fix_relocations(plain_image, image, kernel):
    """gwatch moves code but copies CALL.REL / RET.REL verbatim (their pc-relative offsets keep the old distance) and
    leaves the absolute return-address immediates (MOV Rn, <call pc + 16>) alone, so a subroutine call jumps short and
    returns to a stale pc (illegal address / invalid pc). Align the spliced kernel with the untouched one, re-encode those
    offsets and immediates for the new layout, then check every branch-like target."""
    orig, new = _sass(plain_image), _sass(image)
    a, la = orig
    b = new[0]
    sm = difflib.SequenceMatcher(None, [_norm(r[1]) for r in a], [_norm(r[1]) for r in b], autojunk=False)
    mp = {a[i + k][0]: b[j + k][0] for i, j, n in sm.get_matching_blocks() for k in range(n)}
    assert len(mp) == len(a), (len(mp), len(a))  # every original instruction survives, in order
    calls = {r[0] for r in a if "CALL.REL" in r[1]}
    off, _ = _text(image, kernel)
    buf = bytearray(image)
    fixes = {"call_ret_offsets": 0, "return_movs": 0}
    for pc, text, _, _ in a:
        npc = mp[pc]
        if "CALL.REL" in text or "RET.REL" in text:
            w0, w1 = struct.unpack_from("<QQ", buf, off + npc)
            e0, e1 = _encode_rel(w0, w1, mp[_target(text, la)] - (npc + 16))
            if (e0, e1) != (w0, w1):
                struct.pack_into("<QQ", buf, off + npc, e0, e1)
                fixes["call_ret_offsets"] += 1
        m = re.fullmatch(r"(?:@!?U?P\w+ )?MOV R\d+, (0x[0-9a-f]+)", text)
        if m and int(m.group(1), 16) - 16 in calls:
            ret = int(m.group(1), 16)
            assert struct.unpack_from("<I", buf, off + npc + 4)[0] == ret, hex(pc)
            struct.pack_into("<I", buf, off + npc + 4, mp[ret - 16] + 16)
            fixes["return_movs"] += 1
    image = bytes(buf)
    errors = _branch_errors(orig, _sass(image), mp)
    assert not errors, [(hex(x), hex(y)) for x, y in errors[:10]]
    assert fixes["return_movs"] == len(calls), fixes
    return image, fixes


def decode(raw, clock, sites, records=RECORDS):
    """Raw uint8 trace buffer -> gwatch's per-warp records [{global_tid, site_id, smid, timestamp}] (ns)."""
    schema = make_schema(sites, clock, records)
    inst = TraceInstrumenter_SASS(kernel_prototype="", kernel_def_sass=None, cubin=None, trace_buffer=NoBuffer(raw),
                                  trace_schema=schema, clock_type=clock)
    return inst.collect()


def parse_keep(tag):
    """The CLI keep TAG (see the module doc) as a site set; "" = every site (None)."""
    keep = set() if tag else None
    for tok in filter(None, tag.split(",")):
        if tok in ("arrive", "exit"):
            keep |= {(b, tok) for b in range(64)}
            continue
        m = re.fullmatch(r"b?(\d+)(?:-(\d+))?([ae]?)", tok)
        kinds = [k for k in ("arrive", "exit") if not m.group(3) or k[0] == m.group(3)]
        keep |= {(b, k) for b in range(int(m.group(1)), int(m.group(2) or m.group(1)) + 1) for k in kinds}
    return keep


def main():
    src, out = sys.argv[1], sys.argv[2]
    clock = sys.argv[3] if len(sys.argv) > 3 else "gpu"
    records = int(sys.argv[4]) if len(sys.argv) > 4 else RECORDS
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    gemm = len(sys.argv) > 6 and sys.argv[6] == "gemm"
    os.makedirs(out, exist_ok=True)
    cubin = gb.Cubin()
    cubin.fill(src)
    cubin.parse()
    keep = parse_keep(tag)
    _, image, schema, _, meta = splice(cubin, clock, records, keep, gemm)
    meta["src"] = src
    name = f"xtrace_{clock}" + (f"_{tag}" if tag else "") + ("_gemm" if gemm else "")
    path = os.path.join(out, name + ".cubin")
    with open(path, "wb") as f:
        f.write(image)
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print("wrote", path, "sites", len(meta["sites"]), "buffer MiB", schema.total_bytes / 2**20)


if __name__ == "__main__":
    main()
