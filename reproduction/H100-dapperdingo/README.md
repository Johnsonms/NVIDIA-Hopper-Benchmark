# H100 reproduction bundle — node `dapperdingo-002`

Reproduction of the HPMLL Hopper microbenchmark suite on an **H100 80GB HBM3 (SXM5)**,
CUDA 12.9, as a validated baseline for the B200 effort (see repo-root `CLAUDE.md`).

## Contents

- **`run_suite.sh`** — one-shot reproducible runner. Captures environment and raw
  per-benchmark stdout. Re-run any time: `bash run_suite.sh` (or `GPU=<id> bash run_suite.sh`).
- **`env/`** — machine snapshot: `deviceQuery.txt`, `nvidia-smi.txt`, `gpu-spec.csv`, `nvcc.txt`.
- **`raw/`** — raw output of every benchmark (primary evidence, treat as immutable).
- **`summary.md`** — parsed numbers + side-by-side against the paper's **H800** column
  (arXiv:2501.12084). This is the paper-ready comparison table.
- **`METHODOLOGY.md`** — exactly how each metric is measured and the correctness pitfalls
  (e.g. `mem_lat` measures L2 not DRAM by default; WGMMA under-saturation on CUDA 12.9).

## TL;DR of the H100-vs-paper(H800) alignment

- **Memory hierarchy aligns well**: shared latency 29.0 cyc (exact), L1 BW 125.5 vs 125.8,
  shared BW 127.5 vs 127.4, L2 latency in range, L2 size / SMEM-per-SM identical.
- **Tensor cores align once measured correctly**: WGMMA fp16 ~976 TFLOPS (H100 spec ~989),
  fp8/int8 ~1951 — and scaling the paper's H800 fp16 (728.5) by SM count × clock predicts ~951,
  matching. (An early ~484 "anomaly" was GPU contention from a concurrent job, not hardware —
  see METHODOLOGY §2.)
- **Expected deltas** (different SKU): H100 has 132 SMs vs 114, 1980 vs 1755 MHz, HBM3
  ~3007 GB/s vs HBM2e 2039, and DRAM latency scales with clock.

See `summary.md` for the full table and `METHODOLOGY.md` for how to read/reproduce it.
