# pi0.5 action chunk: ported megakernel vs SOTA multi-kernel baselines

B=1, prefix KV = 128 (fully valid), horizon 50, 10 Euler steps x 18 Gemma-300M expert layers.
One launch of every arm = one complete action chunk (`actions` [50, 32] fp32).

    sbatch slurm/pi05.sbatch --tag p128            # full: 3 seeds, 5 rounds
    sbatch slurm/pi05.sbatch --seeds 7301 --rounds 1 --iters 3 --steady 5 --tag smoke

| file | what |
|---|---|
| `mk/pi05_mk.cu` | Keisuke's final megakernel (v18), only change: `PI05_PREFIX` compile-time prefix |
| `megakernel.py` | arm; variants `k64/k32/k16` = `PI05_ATT_KEYS` (attention key chunk); optional suffix `-gb1` / `-gb2` = `PI05_G_BATCH` (stage G loads of a CTA's two tasks batched, bit-exact; CuTe arm: `Knobs.g_batch`). Default 0 until the GPU A/B; `gbatch_equal.py` checks bitwise equality |
| `cute/` | CuTe DSL (cutlass-dsl) port of the megakernel (`cute/NOTES.md`); arm `cute` in `bench.py` (opt-in `--arms megakernel,cute`) |
| `sota_graph.py` | FlashInfer/cuBLAS/cuDNN GEMMs + Triton attention + fused Triton glue, 1 CUDA graph, PDL; autotuned per op (L2-cold) then refined end to end in the full graph |
| `triton_ops.py` | PDL Triton kernels: residual+adaRMS, RoPE+KV append, step boundary, GQA attention |
| `compile_arm.py` | `torch.compile(mode="max-autotune-no-cudagraphs")` step, chunk captured in 1 CUDA graph |
| `bench.py` | correctness (trusted FP32-oracle band, 3 seeds, mutation check), interleaved timing, profiles |
| `reference/` | vendored MegaBench pi0.5 reference + FP32 oracle |

## Result (run `runs/20261016T004914Z-p128`, B200, job 7353)

| arm | correct | ms / chunk (L2 flushed) | kernels | x HBM floor (0.84 ms) |
|---|---|---|---|---|
| sota-graph / graph-pdl | pass | **6.73** | 1638 | 8.0x |
| megakernel / k16 | pass | 7.69 | 1 | 9.1x |
| megakernel / k32 | pass | 7.77 | 1 | 9.2x |
| megakernel / k64 (graded config) | pass | 7.98 | 1 | 9.5x |
| sota-graph / graph-nopdl | pass | 8.03 | 1638 | 9.6x |
| torch.compile max-autotune + graph | pass | 10.08 | 2586 | 12.0x |
| sota-graph / eager | pass | 159.0 | 1637 | |
| eager reference (MegaBench) | pass | 168.2 | | |

Megakernel speedup over the best baseline: **0.88x** (steady and isolated agree).
Chosen baseline: qkv `tgv`, o `cublas`, gate/up `cublaslt`, down `cudnn`, attention Triton
(BM16 BN64, 8 warps). FlashInfer attention: fa2 slower than Triton; cutlass / cute-dsl do not
support head_dim 256 on sm_100; cudnn ragged plan failed.
