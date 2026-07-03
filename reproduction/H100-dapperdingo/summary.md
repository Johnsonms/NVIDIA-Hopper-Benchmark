# H100 results vs paper (H800) — summary

Node `dapperdingo-002`, NVIDIA **H100 80GB HBM3 (SXM5)**, CUDA 12.9, driver 575.x.
Numbers parsed from `raw/` (regenerate with `run_suite.sh`). Paper column = **H800 PCIe**
from Luo et al., arXiv:2501.12084 (Tables II–XI). H800/H100 share the GH100 die, so memory
structure/latency should align; SM count, clock, HBM bandwidth and FP64 differ by SKU.

## Platform (deviceQuery vs paper Table II)

| Property | Paper H800 | This H100 |
|---|---|---|
| Compute capability | 9.0 | 9.0 |
| SMs | 114 | **132** |
| Max clock | 1755 MHz | **1980 MHz** |
| Memory | 80 GB HBM2e | 80 GB **HBM3** |
| Mem bus | 5120-bit | 5120-bit |
| L2 cache | 50 MB | 50 MB |
| L1+SMEM / SM | 256 KB | 256 KB (228 KB max SMEM) |

## Compute throughput (CUDA cores) — H100 only (paper reports no CUDA-core FLOP table)

flop/clk/SM, mean ± std over N=10 (single-run value in parens where N=1):

| dtype | flop/clk/SM |
|---|---|
| FP32 | 248.31 ± 0.05 |
| FP16 | 405.7 ± 9.6 (noisy; peaks ~417) |
| FP64 | 126.25 ± 0.006 |
| INT32 | (126.9) |
| INT64 | (42.4) |

FP16 shows run-to-run variance (~±2.4%); worth an `ncu`-validated re-measure. Others are stable.

## Instruction latency (clk)

| op | latency |
|---|---|
| FP32 | 4.07 |
| FP16 | 8.52 |
| FP64 | 8.04 |
| INT32 | 4.12 |
| INT64 | 10.06 |

## Memory latency (cycles) — vs paper Table III/IV — mean ± std, N=10

| scope | Paper H800 | This H100 | note |
|---|---|---|---|
| L1 | 32.0 | 41.12 ± 0.003 | ~+28%, stable (real, not noise); both measure L1 |
| Shared | 29.0 | **29.015 ± 0.000** | exact match ✅ |
| L2 | 264.5–502 | 272.34 ± 0.07 | in range ✅ |
| Global (DRAM) | 656 | **484 ± 0.3** | resolved by sweep below ✅ |

**Global-latency array-size sweep** (mean ± std, N=3) — the default `mem_lat` 8 MB array fits in
the 50 MB L2 and only measures L2; latency converges to the true DRAM value once the array
exceeds L2:

| array | latency (cyc) |
|---|---|
| 8 MB | 299.8 ± 0.05 (L2-resident) |
| 32 MB | 382.1 ± 2.9 (spilling) |
| 64 MB | 484.0 ± 0.3 |
| 128 MB | 484.1 ± 0.3 |
| 256 MB | 484.4 ± 0.2 |
| 480 MB | 481.7 ± 0.4 |
| 1024 MB | 482.0 ± 0.4 |

Knee sits at the 50 MB L2 boundary as expected; DRAM latency is **stable at ~484 cyc** for any
array ≥ 64 MB. (An earlier one-off reading of ~1047 cyc did not reproduce and is discarded.)
Note H100's 484 cyc @ 1980 MHz ≈ 244 ns vs H800's 656 cyc @ 1755 MHz ≈ 374 ns — H100's absolute
DRAM latency is genuinely lower (HBM3), so normalize for clock when comparing cycle counts.

## Bandwidth — vs paper Table V

Mean ± std, N=10 (except L2-64f, single run):

| level | Paper H800 | This H100 | note |
|---|---|---|---|
| L1 (byte/clk/SM) | 125.8 | 125.70 ± 0.22 | ✅ |
| Shared (byte/clk/SM) | 127.4 | 127.53 ± 0.00 | ✅ |
| L2 (byte/clk) | 4472.3 | 4207 ± 27 (32f); 4808 (64f) | ✅ close |
| Global HBM (GB/s) | 2039 peak / 1407 meas | **3010 ± 1.4** | higher (HBM3), expected ↑ |

## Tensor cores — WGMMA (warpgroup, sm_90a), peak across ss/rs×zero/random

| type | shape | Paper H800 | This H100 | H800×(SM·clk)→H100 predicted | vs H100 spec |
|---|---|---|---|---|---|
| FP16→FP32 | m64n256k16 | 728.5 | **976.6** | ~951 | ~989 dense ✅ |
| TF32→FP32 | m64n256k8 | 373 | **488.2** | ~487 | ~495 ✅ |
| FP8 (e4m3)→FP32 | m64n256k32 | 1448 | **1951.2** | ~1889 | ~1979 ✅ |
| INT8→INT32 | m64n256k32 | 1448 | **1950.7** (TOPS) | ~1889 | ~1979 ✅ |
| B1 (binary) | m64n256k256 | — | 15608 | — | — |

Scaling the paper's H800 by SM count (132/114) × clock (1980/1755) predicts the H100 values well
→ **tensor cores align**. (Note: an early ~484 fp16 reading was GPU contention from a concurrent
job, not the hardware — see METHODOLOGY §2.)

## Tensor cores — MMA (warp-level `mma.sync`), peak mma-inst throughput (TFLOPS/TOPS)

Each `mma/<shape>` binary sweeps several dtypes; value = peak over the dtypes it emits.

| shape | dtypes emitted | peak |
|---|---|---|
| m16n8k4 | tf32→f32 | 264.5 |
| m16n8k8 | f16, tf32 | 535.3 |
| m16n8k16 | f16, s8/int8 | 1062.2 |
| m16n8k32 | e4m3, s4/s8/int | 1423.7 |
| m16n8k128 | b1 | 8465.0 |
| m16n8k256 | b1 | 11389.2 |

`mma.sync` (warp) is expectedly below `wgmma` (warpgroup) for the same dtype — wgmma is the
efficient Hopper path.

## Verdict

Memory hierarchy (latency/bandwidth/structure) reproduces the paper's H800 closely, with only
expected SKU deltas (SMs/clock/HBM). Tensor-core throughput aligns with both the SM·clock-scaled
H800 numbers and H100's published peaks. Values now include N=10 error bars and are highly stable
(most std ≈ 0); the DRAM-latency array sweep is resolved (~484 cyc ≥ 64 MB). Remaining follow-ups:
(a) **L1 latency 41.12 vs 32** — stable and reproducible, so a genuine H100-vs-H800/toolkit
difference to explain (not noise); (b) **FP16 CUDA-core throughput variance** (±2.4%) — re-measure
with `ncu`. Both are minor and documented, not blockers.
