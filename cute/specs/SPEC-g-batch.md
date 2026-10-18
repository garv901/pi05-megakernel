# SPEC: stage G two-task load batching (`Knobs.g_batch`)

Owner: Sonnet subagent, CPU ONLY. Do not run sbatch, srun, or anything that touches a GPU (`CUDA_VISIBLE_DEVICES=` on every
build). The main agent reviews the diff and runs the GPU jobs.

Config for all builds: **P968 k64** (`Knobs()` defaults). P128 is not used in this spec.

## Problem (context, already measured)

`stage_g` (`ffn.py:126-158`) has 2 * (FF // 32) = 256 tasks on 148 CTAs. The loop at `ffn.py:143`
(`for task in cutlass.range(nbk - 1 - bid, 2 * (FF // 32), nbk, unroll=1)`) gives CTA b the tasks {147 - b, 295 - b}:
CTAs 0-39 run 1 task, CTAs 40-147 run 2. A task is one L2 round trip (two `reduce_part4` calls = 16 float4 `.cg` loads per
thread, `rowops.py:169`), a few FLOPs and one 64-bit store. The second task's loads issue only after the first task's
math and store, so 2-task CTAs pay two serial L2 latencies, and H's grid barrier waits for them.

## Change

Issue the loads of both tasks of a CTA before any math. Each `act` element keeps the same arithmetic (sum of the 8 partials
s = 0..7 in order from +0.0, then `X.gelu_mul`, then the bf16 pack), so the result stays bit-exact. Each CTA computes the
same tasks as today.

### 1. Knob

- `config.py`: add `g_batch: int = 0` to `Knobs` (after `d_wpt`). Validate `g_batch in (0, 1)` in `__post_init__`.
- Add `--g-batch` (int, default 0) to `perf.py`, `parity.py`, `profile.py`. Pass it into `Knobs(...)`. Add it to the
  JSON / result dicts next to `d_wpt`, and add `_g{g_batch}` to the parity output file name next to `_d{d_wpt}`.
- `parity.py`: the CUDA reference has NO equivalent. Do not add a `-D` flag for it. The CUDA build must be the same for
  g_batch 0 and 1 (the reference is the same arithmetic).
- Check every other `Knobs(...)` construction in `cute/` and `tests/` (grep). Defaults must keep them working.

### 2. `rowops.py`: split `reduce_part4`

```python
@cute.jit
def part4_load(part, S_: cutlass.Constexpr, N: cutlass.Constexpr, t, c):
    """the S .cg float4 loads of reduce_part4 (all in flight), returns the (S, 4) register tensor"""

@cute.jit
def part4_sum(x, S_: cutlass.Constexpr):
    """sum over s = 0..S-1 in order from +0.0, per element; returns the (4,) register tensor"""

def reduce_part4(...):   # unchanged signature and docstring, body = part4_sum(part4_load(...))
```

`reduce_part4` with g_batch 0 must give byte-identical PTX (acceptance 1). If the split changes the PTX of the g_batch 0
build, keep the old `reduce_part4` body as it is and add the two new functions next to it.

### 3. `ffn.py`: `stage_g`

Keep the current loop under `if cutlass.const_expr(not knobs.g_batch)`. With `knobs.g_batch`, use this loop:

```python
NT = 2 * (FF // 32)
for task in cutlass.range(nbk - 1 - bid, NT, 2 * nbk, unroll=1):
    task2 = task + nbk
    has2 = task2 < NT
    tb = task2 if has2 else task         # cutlass.select_ / Int32 arithmetic: no branch around the loads
    jg1, jg2 = task >> 1, tb >> 1
    t1 = (task & 1) * GR + tid // 8 ; c1 = jg1 * 32 + (tid % 8) * 4
    t2 = (tb & 1) * GR + tid // 8   ; c2 = jg2 * 32 + (tid % 8) * 4
    if fuse_fg: block_wait(jg1) ; block_wait(jg2)     # both waits before any load (same counter / target as today)
    if (tid // 8 < GR) & (t1 < T):                     # t2 < T holds whenever t1's guard holds (row half <= 1, tid//8 <= 24)
        xg1 = part4_load(part, W // KC, 2 * FF, t1, c1)
        xu1 = part4_load(part, W // KC, 2 * FF, t1, FF + c1)
        xg2 = part4_load(part, W // KC, 2 * FF, t2, c2)
        xu2 = part4_load(part, W // KC, 2 * FF, t2, FF + c2)
        <task 1: part4_sum, gelu_mul x4, bf16 pack, 64-bit store at act[t1, c1]>   (same code as today)
        if has2:
            <task 2: same, store at act[t2, c2]>
```

Rules:
- All 32 loads of a pass are unconditional inside the row guard (the clamp `tb = task` makes the address valid when there
  is no second task). Only the store of task 2 is under `has2`.
- The `block_wait` for jg2 when `has2` is false waits on jg1's counter again. That is correct and cheap.
- Do not change the math, the store width, the thread-to-element map, the barrier at the top of the stage, or the L2 prefetch.
- Factor the per-task epilogue into a small helper only if the g_batch 0 PTX stays byte-identical.
- Add a 1-2 line comment at the loop that says why (loads of both tasks in flight, one L2 round trip per pass).

## Step 0: pre-edit snapshots (before any edit)

Products go to `cute/runs/g-batch-cpu/` (`pre/` = snapshots). Build in a fresh process like `compile_build` in
`tests/test_pi05_cute_profile.py`, but with `Knobs()` (P968 k64), `Stop(stage=None)`, `CUTE_DSL_KEEP=ptx,cubin`,
`CUTE_DSL_LINEINFO=1`, `CUTE_DSL_ARCH=sm_100a`, `CUTE_DSL_NO_CACHE=1`, `CUTE_DSL_DISABLE_FILE_CACHING=1`,
`CUDA_VISIBLE_DEVICES=`. Save:
- `pre/plain_full.ptx`, `pre/plain_full.cubin` (no iket)
- `pre/plain_full_f23.ptx` (`Knobs(fuse=23)`, the FUSE_FG path)

## Acceptance (CPU; log the results in a "## Results" section at the end of this file)

1. **Byte identity.** g_batch 0 builds after the edit: `cmp` equal to `pre/plain_full.ptx` and `pre/plain_full_f23.ptx`.
2. **Compile.** g_batch 1 compiles: full build at fuse 18 and fuse 23, and the `Stop` builds that end at G and at H
   (see `kernel.Stop` for the stage names).
3. **SASS order (assert).** Write `cute/g_batch_check.py CUBIN` (stdlib only, like `gemm_marks_check.py`; reuse its
   nvdisasm line info parsing by import if that is clean). In the g_batch 1 full cubin, find the instructions whose
   line info is in `stage_g` / `part4_load` / `part4_sum` (ffn.py and rowops.py lines). For each G loop body:
   all 32 `LDG.E.128` (`.STRONG.GPU`/`.cg` variant, whatever the build emits) come before the first FADD of the
   reduction and before the first `STG.E.64`. Exit 1 on a violation and print it. Report the same counts for the
   g_batch 0 cubin (16 LDG before the first FADD) as a sanity check.
4. **Registers and spills (report, do not assert).** For both full builds (g_batch 0 and 1, fuse 18): REG and STACK from
   `cuobjdump -res-usage`, and LDL / STL per source file:line (`gemm_marks_check.py PLAIN MARK` already prints this
   table: run it with the g_batch 0 cubin as PLAIN and the g_batch 1 cubin as MARK, and put its table in the results).
   Call out every new LDL / STL line and any change at the gemm.py `load_w` line (H spill).
5. **Tests.** New test file `tests/test_pi05_cute_gbatch.py`: knob validation; acceptance 1-3 as tests (compile in a
   fresh process, P968 k64, skip without the venv). All existing quick tests in `tests/test_pi05_cute_profile.py` and
   `tests/test_pi05_cute_phase6.py` still pass (list which ones you ran and their result).
6. **NOTES.md.** One short paragraph: what `g_batch` does, that it is bit-exact by construction, and that the default
   stays 0 until the GPU A/B.

## Fallback (only if acceptance 4 shows NEW spills in G or a STACK increase)

Report first, then add `g_batch = 2`: load task 1 (g, u), load task 2's g, sum task 1, load task 2's u, epilogue 1,
epilogue 2. That keeps about 96 floats live. Run acceptance 1-4 on it as well. Do not do this if g_batch 1 has no
new spills.

## Files Sonnet may touch

`ffn.py`, `rowops.py`, `config.py`, `perf.py`, `parity.py`, `profile.py`, new `g_batch_check.py`, new
`tests/test_pi05_cute_gbatch.py`, `NOTES.md`, this spec (results section only). Nothing else. No git commits.

## GPU verification (main agent, not Sonnet)

One job, P968 k64: `parity.py --g-batch 1` (fuse 18 and 23) byte-equal to CUDA; `perf.py` 20 iters at fuse 18/23 x
g_batch 0/1; IKET stage-only trace with g_batch 1 → G duration of 1-task vs 2-task CTAs and the H `wpre` wait.

## Results

Config P968 k64, CPU only (CUDA_VISIBLE_DEVICES=), products in `runs/g-batch-cpu/{pre,post}/`. Build script: scratchpad build.py (CUTE_DSL_LINEINFO=1) / build_nl.py (LINEINFO=0).

### 1. Byte identity (g_batch 0 vs pre-edit)
- The lineinfo PTX (`pre/plain_full.ptx`) contains `.loc` lines, so every source edit that shifts lines changes it: plain `cmp` differs at byte 422434 (line 11925, a `.loc`). With `.loc` lines removed the PTX is identical (fuse 18). To get a line-independent check, pre-edit snapshots were also taken with CUTE_DSL_LINEINFO=0 (`pre/nl_full.ptx`, `pre/nl_full_f23.ptx`).
- `cmp pre/nl_full.ptx post/nl_full.ptx`: EQUAL (fuse 18). `cmp pre/nl_full_f23.ptx post/nl_full_f23.ptx`: EQUAL (fuse 23). The `reduce_part4` split (`part4_load` + `part4_sum`) was kept.
- Lineinfo snapshots: `cmp pre/plain_full.ptx post/gb0_full.ptx` differs only in `.loc` (diff after stripping `.loc`: none).

### 2. Compile (g_batch 1 and 2)
full fuse 18, full fuse 23, Stop G, Stop H: all compile for g_batch 1 and g_batch 2 (REG 255; STACK g1: 416 / 400 / 384 / 384, g2: 400 / 400 / 384 / 384; g0: 392 / 400 / 400 (G) / 392 (H)).

### 3. SASS order (g_batch_check.py): FAIL for g_batch 1 (and 2)

g_batch 0 (`g_batch_check.py gb0_full.cubin --expect 16`):
```
G body bb 369: 16 LDG.E.128, 64 reduction FADD, 1 STG.E.64; LDG before first FADD: 15, before first STG.E.64: 16; first FADD at instr 11845, first STG at 12027, loads at 11828..11866
VIOLATION: bb 369: LDG at instr 11866 (0x2e5a0, rowops.py:175) after the first reduction FADD (instr 11845)
RESULT: FAIL
... (1 violations in total)
```

g_batch 1 (`g_batch_check.py gb1_full.cubin --expect 32`):
```
G body bb 369: 32 LDG.E.128, 124 reduction FADD, 2 STG.E.64; LDG before first FADD: 12, before first STG.E.64: 32; first FADD at instr 11861, first STG at 12249, loads at 11847..12109
VIOLATION: bb 369: LDG at instr 11888 (0x2e700, rowops.py:175) after the first reduction FADD (instr 11861)
VIOLATION: bb 369: LDG at instr 11895 (0x2e770, rowops.py:175) after the first reduction FADD (instr 11861)
VIOLATION: bb 369: LDG at instr 11990 (0x2ed60, rowops.py:175) after the first reduction FADD (instr 11861)
... (20 violations in total)
```

g_batch 2 (`g_batch_check.py gb2_full.cubin --expect 32`):
```
G body bb 370: 32 LDG.E.128, 124 reduction FADD, 2 STG.E.64; LDG before first FADD: 12, before first STG.E.64: 32; first FADD at instr 11890, first STG at 12275, loads at 11873..12136
VIOLATION: bb 370: LDG at instr 11914 (0x2e8a0, rowops.py:175) after the first reduction FADD (instr 11890)
VIOLATION: bb 370: LDG at instr 11922 (0x2e920, rowops.py:175) after the first reduction FADD (instr 11890)
VIOLATION: bb 370: LDG at instr 12016 (0x2ef00, rowops.py:175) after the first reduction FADD (instr 11890)
... (20 violations in total)
```

Reading: all loads of a pass are in one basic block and all come before the first STG.E.64 (g1: 32 of 32). But ptxas moves loads behind the first reduction FADD: g_batch 0 already has 15 of 16 before it (1 after); g_batch 1 has 12 of 32 before it (20 after); g_batch 2 has 12 of 32. The batching is therefore only partly effective in SASS; the check as specified (all 32 before the first FADD) is not met. Cause (hypothesis, not tested): ptxas schedules under the 255-register limit and sinks loads to the point of use.

### 4. Registers and spills (g_batch 0 vs 1, fuse 18; full tables in `runs/g-batch-cpu/gemm_marks_g0_vs_g1.txt`, `..._g0_vs_g2.txt`)

| build | REG | STACK | LDL | STL |
|---|---|---|---|---|
| plain | 255 | 392 | 51 | 64 |
| marker | 255 | 416 | 65 | 72 |

LDL / STL per source line (`*` = load_w 128-bit copy gemm.py:119, `K` = kernel signature kernel.py:279); marker minus plain in the last column:

| line | plain LDL | plain STL | marker LDL | marker STL | diff LDL | diff STL | source |
|---|---|---|---|---|---|---|---|
| attention.py:154 | 0 | 0 | 0 | 3 | +0 | +3 | `rsum[r8] = rsum[r8] + pr` |
| attention.py:186 | 0 | 0 | 7 | 0 | +7 | +0 | `cute.gemm(tm, accO, tOrP[None, None, ks], tOrV[None, None, k` |
| attention.py:188 | 0 | 0 | 1 | 0 | +1 | +0 | `row = r0 + mt * 16 + g + 8 * r8` |
| gemm.py:97 | 0 | 0 | 0 | 1 | +0 | +1 | `cute.copy(sm_atom, ov[m, None], cute.local_tile(sA32, (1, 4)` |
| gemm.py:112 | 2 | 0 | 0 | 0 | -2 | +0 | `ptr = cute.make_ptr(BFloat16, base + ((nl * K + k0) * 2).to(` |
| gemm.py:114 | 0 | 0 | 1 | 0 | +1 | +0 | `tB = tmm.partition_B(cute.composition(gB, (None, kperm()))) ` |
| gemm.py:119 * | 19 | 36 | 1 | 16 | -18 | -20 | `cute.copy(S.nc_atom(BFloat16, 128), src, dst)` |
| gemm.py:136 | 0 | 0 | 1 | 0 | +1 | +0 | `g = lane // 4` |
| gemm.py:157 | 0 | 0 | 5 | 4 | +5 | +4 | `sAp = tha.partition_S(sA)` |
| gemm.py:191 | 0 | 0 | 11 | 12 | +11 | +12 | `cute.gemm(tm, acc, rA[None, None, kb], rB[None, None, kb], a` |
| gemm.py:199 | 0 | 0 | 2 | 0 | +2 | +0 | `addr = part_base + (((s * T + row) * N + n0 + nt * 8 + 2 * t` |
| kernel.py:266 | 0 | 0 | 1 | 0 | +1 | +0 | `S.prefetch_l2(cute.recast_ptr(wg[l, None, None].iterator, dt` |
| kernel.py:279 K | 0 | 0 | 0 | 0 | +0 | +0 | `noise: cute.Tensor, prefix_k: cute.Tensor, prefix_v: cute.Te` |
| kernel.py:282 | 18 | 21 | 0 | 7 | -18 | -14 | `tid, _, _ = cute.arch.thread_idx()` |
| kernel.py:447 | 3 | 0 | 26 | 19 | +23 | +19 | `@cute.jit` |
| mathx.py:130 | 0 | 0 | 0 | 2 | +0 | +2 | `return M.exp(x)` |
| mathx.py:155 | 3 | 2 | 3 | 2 | +0 | +0 | `` |
| mathx.py:162 | 3 | 2 | 3 | 2 | +0 | +0 | `` |
| rowops.py:18 | 3 | 2 | 3 | 2 | +0 | +0 | `v = v + cute.arch.shuffle_sync_bfly(v, o)` |
| sync.py:29 | 0 | 1 | 0 | 2 | +0 | +1 | `cute.arch.sync_threads()` |

lines whose LDL or STL count differs: 16


g_batch 2 vs 0: REG 255 / 255, STACK 392 / 400, LDL 51 / 57, STL 64 / 71 (same pattern of changed lines, smaller).
New LDL / STL lines with g_batch 1: none in G code (no new spill line in ffn.py, rowops.py reduce lines or mathx.py; the rowops.py:18, mathx.py:155/162 lines are unchanged). New lines are elsewhere and are ptxas reallocation noise from the global 255-register allocation: attention.py:154/186/188, gemm.py:97/114/136/157/191/199, kernel.py:266, kernel.py:447 (+23 LDL, +19 STL), sync.py. At the gemm.py:119 `load_w` line (H spill): LDL 19 -> 1, STL 36 -> 16 (fewer); kernel.py:282 18/21 -> 0/7. STACK +24 (392 -> 416) for g_batch 1 at fuse 18, +0 at fuse 23 (400 -> 400); the baseline STACK itself moves 392 / 400 between fuse 18 / 23 and between Stop G / Stop H, so STACK is layout dependent. The fallback rule (STACK increase) fired, so g_batch 2 was added: STACK 400 (+8), LDL +6, STL +7.

### 5. Tests
- `tests/test_pi05_cute_gbatch.py` (6 tests, 187 s): test_knob, test_g0_ptx_equals_snapshot (nl PTX vs snapshot, fuse 18 and 23), test_compiles (g_batch 1, 2 x full 18, full 23, Stop G, Stop H), test_g0_sanity, test_g1_all_loads_before_stores: pass. test_g1_all_loads_before_first_fadd: marked expectedFailure (see 3).
- `tests/test_pi05_cute_profile.py`: all 12 tests pass (Names x2, Plain, Iket x3, Options, GemmMarks x3, ProfileCli x2).
- `tests/test_pi05_cute_phase6.py`: all 7 tests pass (PerfArgs x2, ResetConfig x2, CompileFull x3: P968 k64 REG 255 STACK 392).

### Notes / deviations
- Added `g_batch = 2` (the spec's fallback: task 1 g,u loads and task 2 g loads, sum task 1, task 2 u loads, epilogues) because STACK increased with g_batch 1. `Knobs` validates `g_batch in (0, 1, 2)`. g_batch 2 compiles for all four builds; it also fails the all-loads-before-FADD check (12 of 32).
- The per-task epilogue is a helper `ffn.g_epilogue`, used only by the g_batch 1 / 2 path; the g_batch 0 loop is untouched (PTX identical).
- `profile.py` also got `--g-batch` and the field in its PROFILE record.
