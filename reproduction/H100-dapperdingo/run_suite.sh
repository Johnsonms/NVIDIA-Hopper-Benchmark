#!/bin/bash
# Reproducible microbenchmark runner for a single node.
# Captures environment + raw per-benchmark output into ./env and ./raw.
#
# Usage:   bash run_suite.sh
# Requires: nvcc on PATH, benchmarks already built (see repo README / build_all.sh),
#           one visible GPU (set GPU=<id>, default 0).
#
# This script is intentionally read-only w.r.t. the repo's own sources; it only
# writes into reproduction/<node>/{env,raw}. It does NOT lock clocks (the H100
# holds its boost clock under load); instead it records the clock during runs so
# any throttling is visible in the logs.
set -u
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RU="$ROOT/RegularUnits"
TC="$ROOT/TensorCores"
ENVDIR="$HERE/env"; RAW="$HERE/raw"
mkdir -p "$ENVDIR" "$RAW"

echo "### capturing environment ###"
nvidia-smi > "$ENVDIR/nvidia-smi.txt" 2>&1
nvidia-smi -i "$GPU" --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,memory.total \
  --format=csv > "$ENVDIR/gpu-spec.csv" 2>&1
nvcc --version > "$ENVDIR/nvcc.txt" 2>&1
"$RU/deviceQuery/deviceQuery" > "$ENVDIR/deviceQuery.txt" 2>&1

# Helper: run a prebuilt binary, tee to raw log.
runbin() {  # $1 dir  $2 exe
  local d="$1" e="$2"
  [ -x "$d/$e" ] || { echo "  MISSING $d/$e"; return; }
  ( cd "$d" && ./"$e" ) > "$RAW/${e}.log" 2>&1
  echo "  -> raw/${e}.log"
}

echo "### RegularUnits: compute throughput ###"
for u in MaxFlops MaxFlops_16 MaxFlops_64 MaxFlops_int32 MaxFlops_int64; do runbin "$RU/$u" "$u"; done
echo "### RegularUnits: instruction latency ###"
for u in alu_lat_float alu_lat_half alu_lat_double alu_lat_int32 alu_lat_int64; do runbin "$RU/$u" "$u"; done
echo "### RegularUnits: memory latency ###"
for u in l1_lat l2_lat shared_lat mem_lat; do runbin "$RU/$u" "$u"; done
echo "### RegularUnits: bandwidth ###"
for u in l1_bw_32f l1_bw_64f l1_bw_128 l2_bw_32f l2_bw_64f l2_bw_128 shared_bw shared_bw_64 shared_bw_128 mem_bw; do
  runbin "$RU/$u" "$u"; done

# Corrected global-memory latency: default mem_lat uses an 8 MB array that fits
# in the 50 MB L2, so it measures L2 not DRAM. Rebuild with a >L2 array.
echo "### RegularUnits: DRAM latency (array > L2) ###"
if [ -f "$RU/mem_lat/mem_lat.cu" ]; then
  sed 's|#define ARRAY_SIZE (8\*1024\*128)|#define ARRAY_SIZE (62914560)|' \
    "$RU/mem_lat/mem_lat.cu" > "$RAW/mem_lat_dram.cu"
  nvcc "$RAW/mem_lat_dram.cu" -gencode=arch=compute_90,code=sm_90 -o "$RAW/mem_lat_dram" 2>>"$RAW/mem_lat_dram.build.log" \
    && "$RAW/mem_lat_dram" > "$RAW/mem_lat_dram.log" 2>&1 && echo "  -> raw/mem_lat_dram.log"
fi

echo "### TensorCores: MMA (warp-level) ###"
export GPUArch=90
for d in "$TC/mma"/*/; do
  b=$(basename "$d"); [ "$b" = include ] && continue; [ -f "$d/cc.sh" ] || continue
  ( cd "$d"; ./clean.sh >/dev/null 2>&1; GPUArch=90 ./cc.sh >/dev/null 2>&1
    ( [ -x ./single_SM ] && ./single_SM; [ -x ./all_SM ] && ./all_SM ) > "$RAW/mma_${b}.log" 2>&1
    ./clean.sh >/dev/null 2>&1 )
  echo "  -> raw/mma_${b}.log"
done

echo "### TensorCores: WGMMA (warpgroup, sm_90a) throughput ###"
WG="$TC/wgmma/throughput"
if [ -d "$WG" ]; then
  ( cd "$WG" && bash compile.sh ) >>"$RAW/wgmma.build.log" 2>&1
  for t in fp16 tf32 e4m3 s8 b1 fp16fp16 fp16e4m3; do
    [ -x "$WG/${t}_throughput" ] || continue
    ( cd "$WG"; for mode in zero random; do for lay in ss rs; do
        ./"${t}_throughput" "$mode" 256 "$lay"; done; done ) > "$RAW/wgmma_${t}.log" 2>&1
    echo "  -> raw/wgmma_${t}.log"
  done
fi

echo "### done. raw logs in $RAW ; env in $ENVDIR ###"
