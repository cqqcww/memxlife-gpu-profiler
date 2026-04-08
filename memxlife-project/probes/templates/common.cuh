// common.cuh — Shared utilities for all probe templates
#ifndef COMMON_CUH
#define COMMON_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Prevent compiler from optimizing away a value
template<typename T>
__device__ __forceinline__ T use_value(T val) {
    volatile T v = val;
    return v;
}

// GPU-side clock read (returns clock cycles)
__device__ __forceinline__ long long gpu_clock() {
    return clock64();
}

// Print result in standard format for parser
inline void print_result(const char* metric_name, double value, const char* unit,
                         const char* method, int iterations, int warmup) {
    printf("RESULT:%s=%.6f\n", metric_name, value);
    printf("UNIT:%s\n", unit);
    printf("METHOD:%s\n", method);
    printf("ITERATIONS:%d\n", iterations);
    printf("WARMUP:%d\n", warmup);
}

#endif // COMMON_CUH
