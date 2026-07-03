#!/bin/bash
# Repeat key metrics N times -> mean/std/min/max, plus a mem_lat array-size sweep.
# Complements run_suite.sh (which does one full pass). Writes raw/stats.txt.
set -u
GPU="${GPU:-0}"; export CUDA_VISIBLE_DEVICES="$GPU"
N="${N:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"
RU="$ROOT/RegularUnits"; WG="$ROOT/TensorCores/wgmma/throughput"
OUT="$HERE/raw/stats.txt"; : > "$OUT"

stat() { # feed numbers on stdin -> "mean ± std  [min, max]  n=K"
  awk '{a[NR]=$1; s+=$1; if(NR==1||$1<mn)mn=$1; if(NR==1||$1>mx)mx=$1}
       END{m=s/NR; for(i=1;i<=NR;i++)v+=(a[i]-m)^2; sd=(NR>1)?sqrt(v/(NR-1)):0;
           printf "%.3f ± %.3f  [%.3f, %.3f]  n=%d", m, sd, mn, mx, NR}'
}

repeat() { # $1 label  $2 dir  $3 exe  $4 grep-regex (captures the number)
  local label="$1" d="$2" e="$3" re="$4" i v
  [ -x "$d/$e" ] || { printf "%-28s MISSING\n" "$label" | tee -a "$OUT"; return; }
  local nums=""
  for ((i=0;i<N;i++)); do
    v=$(cd "$d" && ./"$e" 2>/dev/null | grep -oE "$re" | grep -oE '[0-9.]+' | tail -1)
    [ -n "$v" ] && nums+="$v"$'\n'
  done
  printf "%-28s %s\n" "$label" "$(printf '%s' "$nums" | stat)" | tee -a "$OUT"
}

echo "### N=$N repeats — RegularUnits scalars ###" | tee -a "$OUT"
repeat "FP32 flop/clk/SM"   "$RU/MaxFlops"      MaxFlops      'FLOP per SM = [0-9.]+'
repeat "FP16 flop/clk/SM"   "$RU/MaxFlops_16"   MaxFlops_16   'FLOP per SM = [0-9.]+'
repeat "FP64 flop/clk/SM"   "$RU/MaxFlops_64"   MaxFlops_64   'FLOP per SM = [0-9.]+'
repeat "L1 latency (cyc)"   "$RU/l1_lat"        l1_lat        'Latency  = +[0-9.]+'
repeat "L2 latency (cyc)"   "$RU/l2_lat"        l2_lat        'Latency = +[0-9.]+'
repeat "Shared latency(cyc)" "$RU/shared_lat"   shared_lat    'Latency  = [0-9.]+'
repeat "L1 BW (byte/clk/SM)" "$RU/l1_bw_32f"    l1_bw_32f     'bandwidth = [0-9.]+'
repeat "L2 BW (byte/clk)"   "$RU/l2_bw_32f"     l2_bw_32f     'bandwidth = [0-9.]+'
repeat "Shared BW(byte/clk/SM)" "$RU/shared_bw" shared_bw     'Bandwidth = [0-9.]+'
repeat "HBM BW (GB/s)"      "$RU/mem_bw"        mem_bw        'GB/sec\)|BW= [0-9.]+'

echo "### N=5 repeats — WGMMA peak TFLOPS (per invocation, exclusive GPU) ###" | tee -a "$OUT"
Nsave=$N; N=5
for t in fp16 tf32 e4m3 s8; do
  [ -x "$WG/$t"_throughput ] || continue
  nums=""; for ((i=0;i<N;i++)); do
    v=$(cd "$WG" && ./"${t}_throughput" zero 256 ss 2>/dev/null | grep -oE '[0-9.]+TFLOPS' | grep -oE '[0-9.]+' | tail -1)
    [ -n "$v" ] && nums+="$v"$'\n'; done
  printf "%-28s %s\n" "wgmma $t (n256 ss)" "$(printf '%s' "$nums" | stat)" | tee -a "$OUT"
done
N=$Nsave

echo "### mem_lat array-size sweep (find L2->DRAM knee, cyc) ###" | tee -a "$OUT"
SRC="$RU/mem_lat/mem_lat.cu"
if [ -f "$SRC" ]; then
  # (elems in 64-bit): 1M=8MB 4M=32MB 8M=64MB 16M=128MB 32M=256MB 62914560=~503MB 134217728=1GB
  for elems in 1048576 4194304 8388608 16777216 33554432 62914560 134217728; do
    mb=$(( elems * 8 / 1048576 ))
    sed "s|#define ARRAY_SIZE (8\*1024\*128)|#define ARRAY_SIZE ($elems)|" "$SRC" > "$HERE/raw/_ml.cu"
    nvcc "$HERE/raw/_ml.cu" -gencode=arch=compute_90,code=sm_90 -o "$HERE/raw/_ml" 2>/dev/null || { echo "  ${mb}MB build-fail"; continue; }
    nums=""; for ((i=0;i<3;i++)); do
      v=$("$HERE/raw/_ml" 2>/dev/null | grep -oE 'latency = +[0-9.]+' | grep -oE '[0-9.]+' | tail -1)
      [ -n "$v" ] && nums+="$v"$'\n'; done
    printf "  %-8s %s\n" "${mb}MB" "$(printf '%s' "$nums" | stat)" | tee -a "$OUT"
  done
  rm -f "$HERE/raw/_ml.cu" "$HERE/raw/_ml"
fi
echo "### wrote $OUT ###"
