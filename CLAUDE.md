# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## Project goal

This is a fork of the HPMLL **NVIDIA-Hopper-Benchmark** suite (from the paper
*"Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple
Level Analysis"*, Luo et al., arXiv:2501.12084). We are using it as a **validated
baseline** toward a larger goal:

1. **Reproduce** the paper's microbenchmark methodology on hardware we control and
   record the results (starting with **H100 SXM**, this node).
2. **Rewrite / extend the suite for B200 (Blackwell, sm_100a)** on a separate machine.
3. **Write a Hopper-style dissection paper for Blackwell.**

The reference paper benchmarks **H800 PCIe** (not H100) plus A100 and RTX 4090. So our
H100 numbers are compared against the paper's **H800** column, expecting: memory-hierarchy
latencies/structure to align closely (same GH100 die), and SM count / clock / HBM
bandwidth / FP64 to differ (those are expected deltas, not repro failures).

## Repo layout (upstream)

- `RegularUnits/` — arch-agnostic microbenchmarks: compute throughput (`MaxFlops*`),
  instruction latency (`alu_lat_*`), memory latency (`l1_lat`,`l2_lat`,`shared_lat`,`mem_lat`),
  bandwidth (`l1_bw_*`,`l2_bw_*`,`shared_bw*`,`mem_bw`), `deviceQuery`. Each leaf dir has a
  `Makefile` (`make CC=nvcc run`).
- `TensorCores/mma`, `.../mmasp` — warp-level `mma.sync` (dense/sparse); build via `cc.sh`,
  run via `run.sh` (builds `single_SM`+`all_SM`).
- `TensorCores/wgmma` — Hopper warpgroup `wgmma` (sm_90a); build via `compile.sh`.
- `NewFeatures/` — TMA, DPX, distributed shared memory (cluster) benchmarks.
- `TeBenchMark/` — separate Python/Transformer-Engine suite (Docker); NOT part of the C++ build.

## Our additions

- `reproduction/<node>/` — per-node reproduction bundle. See `reproduction/H100-dapperdingo/`:
  - `run_suite.sh` — the reproducible runner (captures env + raw per-benchmark logs).
  - `env/` — `deviceQuery`, `nvidia-smi`, `nvcc` version, GPU spec snapshot.
  - `raw/` — raw stdout of every benchmark (the primary evidence).
  - `summary.md` — parsed numbers + side-by-side vs the paper's H800 (the paper-ready table).
  - `METHODOLOGY.md` — how each metric is measured and the pitfalls that bite.

## Build & run

Toolchain on this node: **CUDA 12.9** (`nvcc` on PATH), driver 575.x, 3× **H100 80GB HBM3** (sm_90).
Unit `Makefile`s hardcode `CC=/usr/local/cuda-12.3/bin/nvcc` (nonexistent here) — always
override with `make CC=nvcc`, or use `../build_all.sh` (passes `CC=nvcc`, `GPUArch=90`).

```bash
./build_all.sh                                   # build all C++ units for sm_90
bash reproduction/H100-dapperdingo/run_suite.sh  # run everything, capture logs
GPU=1 bash reproduction/.../run_suite.sh         # pin to a specific GPU
```

Tensor-core scripts read `GPUArch` (80=A100, 89=Ada, 90=Hopper; default 90). "GPUArch isn't
configured, use the default value 90" is an informational message, not an error.

## Methodology rules (learned the hard way — see METHODOLOGY.md)

1. **`mem_lat` measures L2, not DRAM, by default.** Its array is 8 MB, which fits in the
   50 MB L2. To measure true global latency, rebuild with an array > L2 (the runner does this
   as `mem_lat_dram`). This will matter even more on B200 (bigger L2).
2. **Benchmark on a dedicated GPU; a co-tenant halves your numbers.** We initially misread WGMMA
   fp16 as ~484 TFLOPS ("anomaly") — it was actually GPU contention from our own concurrently
   running `mma` batch. With exclusive access it is **~976 TFLOPS** (on spec). Always set
   `CUDA_VISIBLE_DEVICES` to a free GPU, run one job at a time, take the peak across the binary's
   ss/rs × zero/random configs, and confirm clock+power *during* the kernel before trusting a low
   number.
3. **Latency is in clock cycles; normalize by clock when comparing across GPUs.** A higher-clocked
   part shows more cycles for the same nanosecond latency.
4. **Record clock + power during every run** so throttling is visible after the fact.

## B200 (Blackwell) porting notes — for the next phase

- Target `sm_100a`, CUDA ≥ 12.8. RegularUnits should port with a gencode change **plus array-size
  retuning** for the larger L2/HBM.
- Tensor cores are a **rewrite, not a port**: Hopper `wgmma` → Blackwell **`tcgen05.mma` + Tensor
  Memory (TMEM)**; add **2-SM (CTA-pair) MMA** and **MX block-scaled FP8/FP6/FP4**. Validate against
  CUTLASS 3.8+/CuTeDSL.
- Novel B200 topics worth a paper: **dual-die (NV-HBI) L2 NUMA** (near/far-die latency), TMEM
  latency/bandwidth, MX-format throughput, 2-SM MMA scaling.

## Conventions

- Do not commit build artifacts (compiled unit binaries, `*/bin/`, `single_SM`/`all_SM`). Commit
  sources, scripts, docs, and `reproduction/**` (logs + summaries).
- Keep raw logs immutable; put interpretation in `summary.md`, not in `raw/`.
