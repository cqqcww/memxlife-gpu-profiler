// latency_pointer_chase.cu — Measure memory latency via pointer chasing
// Configurable for L1, L2, or DRAM latency depending on data size
//
// Parameters:
//   {{data_size_bytes}} — total array size in bytes (controls cache level hit)
//   {{iterations}}      — number of pointer chase steps
//   {{warmup}}          — warmup iterations

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef DATA_SIZE_BYTES
#define DATA_SIZE_BYTES (256 * 1024 * 1024)  // 256 MB default → DRAM
#endif
#ifndef ITERATIONS
#define ITERATIONS 1000
#endif
#ifndef WARMUP
#define WARMUP 100
#endif

// Single-thread pointer chasing kernel
// Each element stores the index of the next element to visit
__global__ void pointer_chase_kernel(
    const int* __restrict__ chain,
    int start_idx,
    int num_steps,
    int warmup_steps,
    long long* __restrict__ out_cycles
) {
    int idx = start_idx;

    // Warmup — traverse without timing
    for (int i = 0; i < warmup_steps; i++) {
        idx = __ldg(&chain[idx]);
    }

    // Timed traversal
    long long t0 = clock64();
    for (int i = 0; i < num_steps; i++) {
        idx = __ldg(&chain[idx]);
    }
    long long t1 = clock64();

    // Prevent dead code elimination
    if (idx == -999) printf("%d", idx);

    *out_cycles = t1 - t0;
}

// Build a random pointer chase chain (Fisher-Yates shuffle)
void build_random_chain(int* host_chain, int n_elements) {
    // Initialize sequential
    for (int i = 0; i < n_elements; i++) {
        host_chain[i] = (i + 1) % n_elements;
    }
    // Shuffle to create random traversal pattern
    // Use Sattolo's algorithm for a single cycle
    for (int i = n_elements - 1; i > 0; i--) {
        int j = rand() % i;  // j in [0, i-1]
        int tmp = host_chain[i];
        host_chain[i] = host_chain[j];
        host_chain[j] = tmp;
    }
}

// Build a stride-based chain (for more predictable access)
void build_stride_chain(int* host_chain, int n_elements, int stride) {
    for (int i = 0; i < n_elements; i++) {
        host_chain[i] = (i + stride) % n_elements;
    }
}

int main() {
    size_t data_size = DATA_SIZE_BYTES;
    int iterations = ITERATIONS;
    int warmup = WARMUP;

    int n_elements = data_size / sizeof(int);
    if (n_elements < 2) {
        fprintf(stderr, "ERROR:Data size too small\n");
        return 1;
    }

    // Allocate host chain
    int* h_chain = (int*)malloc(data_size);
    if (!h_chain) {
        fprintf(stderr, "ERROR:Host malloc failed\n");
        return 1;
    }

    // Build random chain for cache-defeating access pattern
    srand(42);
    build_random_chain(h_chain, n_elements);

    // Allocate and copy to device
    int* d_chain;
    cudaMalloc(&d_chain, data_size);
    cudaMemcpy(d_chain, h_chain, data_size, cudaMemcpyHostToDevice);

    long long* d_cycles;
    cudaMalloc(&d_cycles, sizeof(long long));

    // Run kernel — single thread for latency measurement
    pointer_chase_kernel<<<1, 1>>>(d_chain, 0, iterations, warmup, d_cycles);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR:Kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Read result
    long long h_cycles;
    cudaMemcpy(&h_cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);

    double avg_latency = (double)h_cycles / (double)iterations;

    // Determine what we're measuring based on data size
    const char* metric_name;
    if (data_size <= 16 * 1024) {
        metric_name = "l1_latency_cycles";
    } else if (data_size <= 4 * 1024 * 1024) {
        metric_name = "l2_latency_cycles";
    } else {
        metric_name = "dram_latency_cycles";
    }

    printf("RESULT:%s=%.2f\n", metric_name, avg_latency);
    printf("UNIT:cycles\n");
    printf("METHOD:pointer_chase (data_size=%zu bytes, random stride)\n", data_size);
    printf("ITERATIONS:%d\n", iterations);
    printf("WARMUP:%d\n", warmup);
    printf("TOTAL_CYCLES:%lld\n", h_cycles);
    printf("N_ELEMENTS:%d\n", n_elements);

    // Cleanup
    cudaFree(d_chain);
    cudaFree(d_cycles);
    free(h_chain);

    return 0;
}
