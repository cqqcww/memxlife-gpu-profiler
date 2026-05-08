#!/bin/bash
set -e
export PATH=/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib

PROJ=/mnt/v/code_Temp/memxlife-gpu-profiler/memxlife-project
TMPL=$PROJ/probes/templates
BD=$PROJ/build_wsl
mkdir -p $BD

NVCC_FLAGS="-O2 -gencode=arch=compute_89,code=sm_89 -w"

echo "=== Compiling all probes ==="
nvcc $NVCC_FLAGS -DDATA_SIZE_BYTES=8192 -DITERATIONS=5000 -DWARMUP=500 -o $BD/latency_l1 $TMPL/latency_pointer_chase.cu
echo "  latency_l1 OK"
nvcc $NVCC_FLAGS -DDATA_SIZE_BYTES=1048576 -DITERATIONS=2000 -DWARMUP=200 -o $BD/latency_l2 $TMPL/latency_pointer_chase.cu
echo "  latency_l2 OK"
nvcc $NVCC_FLAGS -DDATA_SIZE_BYTES=268435456 -DITERATIONS=1000 -DWARMUP=100 -o $BD/latency_dram $TMPL/latency_pointer_chase.cu
echo "  latency_dram OK"
nvcc $NVCC_FLAGS -o $BD/bw_global $TMPL/bandwidth_global.cu
echo "  bw_global OK"
nvcc $NVCC_FLAGS -o $BD/bw_shared $TMPL/bandwidth_shared.cu
echo "  bw_shared OK"
nvcc $NVCC_FLAGS -o $BD/clock $TMPL/clock_frequency.cu
echo "  clock OK"
nvcc $NVCC_FLAGS -o $BD/cache_sweep $TMPL/cache_size_sweep.cu
echo "  cache_sweep OK"
nvcc $NVCC_FLAGS -o $BD/bank_conflict $TMPL/bank_conflict.cu
echo "  bank_conflict OK"
nvcc $NVCC_FLAGS -o $BD/shmem_cap $TMPL/shmem_capacity.cu
echo "  shmem_cap OK"

echo ""
echo "=== Running all probes ==="
echo "--- L1 Latency ---"
$BD/latency_l1
echo "--- L2 Latency ---"
$BD/latency_l2
echo "--- DRAM Latency ---"
$BD/latency_dram
echo "--- Global BW ---"
$BD/bw_global
echo "--- Shared BW ---"
$BD/bw_shared
echo "--- Clock ---"
$BD/clock
echo "--- Cache Sweep ---"
$BD/cache_sweep 2>&1
echo "--- Bank Conflict ---"
$BD/bank_conflict
echo "--- Shmem Cap ---"
$BD/shmem_cap 2>&1
echo ""
echo "=== ALL DONE ==="
