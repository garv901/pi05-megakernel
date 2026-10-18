"""CPU check of the stage G load batching (Knobs.g_batch) on one cubin of the full kernel (stdlib only).

  python g_batch_check.py CUBIN [--expect N]

The cubin must come from a build with CUTE_DSL_LINEINFO=1. The G loop body is the basic block of the `STG.E.64` instructions whose
line info is in ffn.py (the act stores of stage_g / g_epilogue; nvdisasm -g -c via gemm_marks_check.disasm). In that block:
  loads  = LDG.E.128 (.STRONG.GPU for the .cg atom, whatever the build emits), all of the block (the line info of the
           loads is not reliable: ptxas attributes some to the first sum line, so they are taken by opcode),
  FADD   = FADD whose line is in rowops.part4_sum (source ranges come from the AST of rowops.py / ffn.py next to this file),
  store  = the first STG.E.64 of the block.
Assert (exit 1 on a violation, printed): every load of the block comes before the first reduction FADD and before the first
STG.E.64. --expect N (default: none) also asserts N loads in the block (32 with g_batch 1, 16 with g_batch 0). The same counts
are printed for g_batch 0 as a sanity check (16 LDG before the first FADD)."""
import argparse
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gemm_marks_check import disasm  # noqa: E402


def func_range(src, name):
    for n in ast.walk(ast.parse((HERE / src).read_text())):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n.lineno, n.end_lineno
    raise KeyError(name)


CUDA_SRC = HERE.parent / "mk" / "pi05_mk.cu"


def cuda_ranges():
    """(G region lines, reduce line ranges) in mk/pi05_mk.cu: G region = '---- G (no barrier)' .. STOP_AT(7);
    reduction = part4_sum (g_batch 1/2) or reduce_part4 (g_batch 0), each to its closing '}' at column 0."""
    lines = CUDA_SRC.read_text().splitlines()
    g0 = next(i for i, l in enumerate(lines, 1) if "---- G (no barrier)" in l)
    g1 = next(i for i, l in enumerate(lines, 1) if i > g0 and "STOP_AT(7)" in l)
    red = []
    for name in ("part4_sum(", "reduce_part4("):
        a = next(i for i, l in enumerate(lines, 1) if name in l and l.startswith("__device__"))
        b = next(i for i, l in enumerate(lines, 1) if i > a and l.startswith("}"))
        red.append((a, b))
    return (g0, g1), red


def in_range(loc, fname, rng):
    f, _, ln = loc.partition(":")
    return f == fname and ln.isdigit() and rng[0] <= int(ln) <= rng[1]


def check(cubin, expect=None):
    """Returns (report lines, violations)."""
    sum_rng = func_range("rowops.py", "part4_sum") if (HERE / "rowops.py").exists() else None
    ins = disasm(cubin)
    out, viol = [], []
    stores = [i for i, x in enumerate(ins) if x[1].startswith("STG.E.64") and x[3].startswith("ffn.py:")]
    cuda = False
    if not stores:  # nvcc cubin of mk/pi05_mk.cu (-lineinfo): the G stores are the STG.E.64 whose line is in the G region
        g_rng, red = cuda_ranges()
        stores = [i for i, x in enumerate(ins) if x[1].startswith("STG.E.64") and in_range(x[3], "pi05_mk.cu", g_rng)]
        cuda = True
    if not stores:
        return ["no STG.E.64 with ffn.py / pi05_mk.cu stage-G line info (build without line info?)"], ["no G store found"]
    is_red = (lambda loc: any(in_range(loc, "pi05_mk.cu", r) for r in red)) if cuda else (lambda loc: in_range(loc, "rowops.py", sum_rng))
    blocks = sorted({ins[i][4] for i in stores})
    for bb in blocks:
        body = [(i, x) for i, x in enumerate(ins) if x[4] == bb]
        loads = [i for i, x in body if x[1].startswith("LDG.E.128")]
        fadds = [i for i, x in body if x[1].startswith("FADD") and is_red(x[3])]
        stgs = [i for i, x in body if x[1].startswith("STG.E.64")]
        first_f, first_s = (fadds[0] if fadds else None), (stgs[0] if stgs else None)
        late_f = [i for i in loads if first_f is not None and i > first_f]
        late_s = [i for i in loads if first_s is not None and i > first_s]
        out.append(f"G body bb {bb}: {len(loads)} LDG.E.128, {len(fadds)} reduction FADD, {len(stgs)} STG.E.64; "
                   f"LDG before first FADD: {len(loads) - len(late_f)}, before first STG.E.64: {len(loads) - len(late_s)}; "
                   f"first FADD at instr {first_f}, first STG at {first_s}, loads at {loads[0] if loads else None}..{loads[-1] if loads else None}")
        if expect is not None and len(loads) != expect:
            viol.append(f"bb {bb}: {len(loads)} loads, expected {expect}")
        for i in late_f:
            viol.append(f"bb {bb}: LDG at instr {i} ({hex(ins[i][0])}, {ins[i][3]}) after the first reduction FADD (instr {first_f})")
        for i in late_s:
            viol.append(f"bb {bb}: LDG at instr {i} ({hex(ins[i][0])}, {ins[i][3]}) after the first STG.E.64 (instr {first_s})")
    return out, viol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cubin")
    ap.add_argument("--expect", type=int)
    a = ap.parse_args(argv)
    out, viol = check(a.cubin, a.expect)
    print("\n".join(out))
    for v in viol:
        print("VIOLATION:", v)
    print("RESULT:", "FAIL" if viol else "PASS")
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
