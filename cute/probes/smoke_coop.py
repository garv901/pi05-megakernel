"""Phase-0 probe: cooperative launch + counter grid barrier / WPRE split barrier / producer-consumer counters on the full machine.

Run on a GPU: python -m megabench.sota.pi05.cute.probes.smoke_coop --out DIR   (writes smoke.json)
CPU check:    CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_100a python -m megabench.sota.pi05.cute.probes.smoke_coop --compile-only"""

import argparse
import json
import os
import time

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cute.arch import nvvm_wrappers as W

from .. import sync as S
from ..config import BAR_WORDS, CNT_STRIDE, NTHREADS, Knobs

MODES = ("barrier", "wpre", "counters")
SLOTW = 256
NCTR = 7
WPRE_WORDS = 16384  # 64 KB of read-only data per block
SMEM = Knobs(968, 64).smem_bytes


@cute.kernel
def smoke_kernel(slot: cute.Tensor, bar: cute.Tensor, cnt: cute.Tensor, rows: cute.Tensor, wsrc: cute.Tensor,
                 err: cute.Tensor, n_epochs: Int32, mode: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    nb, _, _ = cute.arch.grid_dim()
    bad = Uint32(0)

    nwords = SMEM // 4
    sm = cute.make_tensor(cute.arch.get_dyn_smem(Uint32, alignment=16), cute.make_layout(nwords))
    for i in cutlass.range(tid, nwords, NTHREADS, unroll=1):
        sm[i] = Uint32(i * 3 + bid)
    cute.arch.sync_threads()
    for i in cutlass.range(tid, nwords, NTHREADS, unroll=1):
        if sm[i] != Uint32(i * 3 + bid):
            bad += Uint32(1)

    if cutlass.const_expr(mode != "counters"):
        sink = Uint32(0)
        for e in cutlass.range(n_epochs, unroll=1):
            par = e % 2
            val = (e * 1000 + bid).to(Uint32)
            if tid == 0:
                slot[par * SLOTW + bid] = val
            if cutlass.const_expr(mode == "barrier"):
                S.grid_barrier(bar.iterator, e + 1, nb)
            else:
                S.wpre_arrive(bar.iterator)
                for k in cutlass.range_constexpr(WPRE_WORDS // (4 * NTHREADS)):
                    src = cute.local_tile(wsrc, (4,), (tid + k * NTHREADS,))
                    r = cute.make_rmem_tensor((4,), Uint32)
                    cute.copy(S.nc_atom(Uint32, 128), src, r)
                    sink += r[0] + r[1] + r[2] + r[3]
                S.wpre_wait(bar.iterator, ((e + 1) * nb).to(Uint32))
            for j in cutlass.range(tid, nb, NTHREADS, unroll=1):
                v = S.ld_cg_u32(slot.iterator + par * SLOTW + j)
                if v != (e * 1000 + j).to(Uint32):
                    bad += Uint32(1)
        bad += sink
    else:
        np_ = nb // 2
        if bid < np_:
            c = bid % NCTR
            for e in cutlass.range(n_epochs, unroll=1):
                if tid < 32:
                    rows[bid * 32 + tid] = (e * 1000 + bid).to(Uint32)
                cute.arch.sync_threads()
                if tid == 0:
                    S.counter_publish(S.cnt_ptr(cnt.iterator, c))
        else:
            c = (bid - np_) % NCTR
            arr = (np_ - c + NCTR - 1) // NCTR
            for e in cutlass.range(n_epochs, unroll=1):
                S.block_wait(S.cnt_ptr(cnt.iterator, c), ((e + 1) * arr).to(Uint32), True)
                if tid < 32:
                    for p in cutlass.range(c, np_, NCTR, unroll=1):
                        v = S.ld_cg_u32(rows.iterator + p * 32 + tid)
                        if v < (e * 1000 + p).to(Uint32) or v % Uint32(1000) != Uint32(p):
                            bad += Uint32(1)

    if bad != Uint32(0):
        W.atomic_add(err.iterator, bad, sem="relaxed", scope="gpu")


@cute.jit
def host(slot: cute.Tensor, bar: cute.Tensor, cnt: cute.Tensor, rows: cute.Tensor, wsrc: cute.Tensor,
         err: cute.Tensor, n_epochs: Int32, nblocks: Int32, mode: cutlass.Constexpr):
    smoke_kernel(slot, bar, cnt, rows, wsrc, err, n_epochs, mode).launch(
        grid=(nblocks, 1, 1), block=(NTHREADS, 1, 1), smem=SMEM, min_blocks_per_mp=1, cooperative=True)


SHAPES = {"slot": 2 * SLOTW, "bar": BAR_WORDS, "cnt": NCTR * CNT_STRIDE, "rows": SLOTW * 32, "wsrc": WPRE_WORDS, "err": 4}


def fake_args():
    return [cute.runtime.make_fake_compact_tensor(Uint32, (n,), assumed_align=16) for n in SHAPES.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=1634 * 10)
    ap.add_argument("--launches", type=int, default=3)
    args = ap.parse_args()

    if args.compile_only:
        for m in MODES:
            t = time.time()
            cute.compile(host, *fake_args(), args.epochs, 148, m)
            print(f"compiled {m} in {time.time() - t:.1f}s")
        return

    import torch
    from cutlass.cute.runtime import from_dlpack
    nblocks = torch.cuda.get_device_properties(0).multi_processor_count
    res = {"nblocks": nblocks, "smem_bytes": SMEM, "n_epochs": args.epochs, "cooperative": True, "modes": {}}
    for m in MODES:
        t = {k: torch.zeros(n, dtype=torch.uint32, device="cuda") for k, n in SHAPES.items()}
        a = [from_dlpack(v, assumed_align=16) for v in t.values()]
        fn = cute.compile(host, *a, args.epochs, nblocks, m)
        times, errors = [], 0
        for _ in range(args.launches):
            for k in ("slot", "bar", "cnt", "rows", "err"):
                t[k].zero_()
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            ev[0].record()
            fn(*a, args.epochs, nblocks)
            ev[1].record()
            torch.cuda.synchronize()
            times.append(ev[0].elapsed_time(ev[1]))
            errors += int(t["err"][0].item())
        res["modes"][m] = {"errors": errors, "ms": times, "us_per_epoch": min(times) * 1000 / args.epochs,
                           "us_per_barrier": min(times) * 1000 / args.epochs if m != "counters" else None}
        print(m, res["modes"][m])
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "smoke.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
