// bandwidth_shared.cu — Measure shared memory bandwidth
// Uses conflict-free access pattern to measure peak shared memory throughput
//
// Parameters:
//   {{block_size}}  — threads per block
//   {{iterations}}  — measurement iterations
//   {{warmup}}      — warmup iterations

#include <stdio.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef ITERATIONS
#define ITERATIONS 1000
#endif
#ifndef WARMUP
#define WARMUP 100
#endif

// Each thread reads/writes 4 floats per iteration from shared memory
// Access pattern is conflict-free (sequential within warp)
__global__ void shmem_bandwidth_kernel(
    int iterations,
    int warmup,
    float* __restrict__ out,
    long long* __restrict__ out_cycles
) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int n_floats = blockDim.x * 4;  // 4 floats per thread

    // Initialize shared memory
    for (int i = 0; i < 4; i++) {
        smem[tid * 4 + i] = (float)(tid + i);
    }
    __syncthreads();

    // Warmup
    float sum = 0.0f;
    for (int iter = 0; iter < warmup; iter++) {
        for (int i = 0; i < 4; i++) {
            float val = smem[tid * 4 + i];
            sum += val;
            smem[tid * 4 + i] = val + 1.0f;
        }
        __syncthreads();
    }

    // Timed section
    long long t0 = clock64();
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < 4; i++) {
            float val = smem[tid * 4 + i];
            sum += val;
            smem[tid * 4 + i] = val + 1.0f;
        }
        __syncthreads();
    }
    long long t1 = clock64();

    // Prevent DCE
    if (tid == 0) {
        out[0] = sum;
        out_cycles[0] = t1 - t0;
    }
}

int main() {
    int block_size = BLOCK_SIZE;
    int iterations = ITERATIONS;
    int warmup = WARMUP;

    // Shared memory: 4 floats per thread
    int shmem_bytes = block_size * 4 * sizeof(float);

    float* d_out;
    long long* d_cycles;
    cudaMalloc(&d_out, sizeof(float));
    cudaMalloc(&d_cycles, sizeof(long long));

    // Get device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Use CUDA events for wall-clock timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch on all SMs for peak bandwidth
    int num_blocks = prop.multiProcessorCount;

    cudaEventRecord(start);
    shmem_bandwidth_kernel<<<num_blocks, block_size, shmem_bytes>>>(
        iterations, warmup, d_out, d_cycles
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double elapsed_sec = elapsed_ms / 1000.0;

    // Each iteration: each thread does 4 reads + 4 writes = 8 * 4 bytes = 32 bytes
    // Total bytes = num_blocks * block_size * 32 * iterations
    double total_bytes = (double)num_blocks * block_size * 32.0 * iterations;
    double bw_gb_s = total_bytes / elapsed_sec / 1e9;

    // Also get cycle-based measurement
    long long h_cycles;
    cudaMemcpy(&h_cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);

    printf("RESULT:max_shmem_bandwidth_gb_s=%.2f\n", bw_gb_s);
    printf("UNIT:GB/s\n");
    printf("METHOD:shmem_conflict_free (block_size=%d, %d SMs)\n", block_size, num_blocks);
    printf("ITERATIONS:%d\n", iterations);
    printf("WARMUP:%d\n", warmup);
    printf("ELAPSED_MS=%.3f\n", elapsed_ms);
    printf("TOTAL_BYTES=%.0f\n", total_bytes);
    printf("KERNEL_CYCLES=%lld\n", h_cycles);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    cudaFree(d_cycles);

    return 0;
}
