// clock_frequency.cu — Measure actual GPU core clock frequency under sustained load
// Uses a known number of FMA instructions timed with clock64()
//
// The idea: execute N FMA ops, measure wall time via CUDA events,
// measure cycle count via clock64(). Frequency = cycles / time.

#include <stdio.h>
#include <cuda_runtime.h>

#ifndef FMA_ITERATIONS
#define FMA_ITERATIONS 1000000
#endif
#ifndef WARMUP_ITERATIONS
#define WARMUP_ITERATIONS 100000
#endif
#ifndef TRIALS
#define TRIALS 5
#endif

// Sustained FMA workload — single thread, known instruction count
__global__ void clock_probe_kernel(
    int fma_iters,
    int warmup_iters,
    long long* __restrict__ out_cycles
) {
    float a = 1.0f, b = 1.0001f, c = 0.0001f;

    // Warmup — get GPU to boost clock
    for (int i = 0; i < warmup_iters; i++) {
        a = __fmaf_rn(a, b, c);
    }

    // Timed section
    long long t0 = clock64();
    for (int i = 0; i < fma_iters; i++) {
        a = __fmaf_rn(a, b, c);
    }
    long long t1 = clock64();

    // Prevent DCE
    if (a == -999.0f) printf("%.6f", a);

    *out_cycles = t1 - t0;
}

// Multi-SM version — saturate the GPU to force sustained boost clock
__global__ void clock_probe_kernel_multi(
    int fma_iters,
    int warmup_iters,
    long long* __restrict__ out_cycles
) {
    float a = 1.0f + threadIdx.x * 0.0001f;
    float b = 1.0001f;
    float c = 0.0001f;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        a = __fmaf_rn(a, b, c);
    }
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < fma_iters; i++) {
        a = __fmaf_rn(a, b, c);
    }
    long long t1 = clock64();

    if (a == -999.0f) printf("%.6f", a);

    // Only thread 0 writes
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_cycles = t1 - t0;
    }
}

int main() {
    int fma_iters = FMA_ITERATIONS;
    int warmup_iters = WARMUP_ITERATIONS;
    int trials = TRIALS;

    long long* d_cycles;
    cudaMalloc(&d_cycles, sizeof(long long));

    // Get device info for multi-SM launch
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    double best_freq_mhz = 0;
    double total_freq = 0;

    for (int t = 0; t < trials; t++) {
        // Use multi-SM kernel to force sustained boost
        int num_blocks = prop.multiProcessorCount;
        int block_size = 128;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        clock_probe_kernel_multi<<<num_blocks, block_size>>>(
            fma_iters, warmup_iters, d_cycles
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        double elapsed_sec = elapsed_ms / 1000.0;

        // Read cycle count from thread 0
        long long h_cycles;
        cudaMemcpy(&h_cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);

        // Frequency = cycles / time
        double freq_hz = (double)h_cycles / elapsed_sec;
        double freq_mhz = freq_hz / 1e6;

        fprintf(stderr, "Trial %d: cycles=%lld, time=%.3f ms, freq=%.1f MHz\n",
                t, h_cycles, elapsed_ms, freq_mhz);

        total_freq += freq_mhz;
        if (freq_mhz > best_freq_mhz) {
            best_freq_mhz = freq_mhz;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    double avg_freq = total_freq / trials;

    printf("RESULT:actual_boost_clock_mhz=%.1f\n", avg_freq);
    printf("UNIT:MHz\n");
    printf("METHOD:fma_clock_probe (sustained FMA, %d trials, %d SMs)\n",
           trials, prop.multiProcessorCount);
    printf("ITERATIONS:%d\n", fma_iters);
    printf("WARMUP:%d\n", warmup_iters);
    printf("BEST_FREQ_MHZ=%.1f\n", best_freq_mhz);
    printf("AVG_FREQ_MHZ=%.1f\n", avg_freq);

    cudaFree(d_cycles);
    return 0;
}
