#!/bin/bash
export PATH=/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib
cat > /tmp/test.cu << 'CUDA'
#include <stdio.h>
#include <cuda_runtime.h>
__global__ void k() { printf("GPU OK\n"); }
int main() { k<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
CUDA
nvcc -O2 -gencode=arch=compute_89,code=sm_89 -o /tmp/test /tmp/test.cu 2>&1
/tmp/test 2>&1
echo "ALL_GOOD"
