// bank_conflict.cu — Measure shared memory bank conflict penalty
// Compares conflict-free vs all-same-bank access patterns
//
// Parameters:
//   {{block_size}}  — threads per block (should be >= 32)
//   {{iterations}}  — measurement iterations
//   {{warmup}}      — warmup iterations

#include <stdio.h>
#include <cuda_runtime.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef ITERATIONS
#define ITERATIONS 10000
#endif
#ifndef WARMUP
#define WARMUP 1000
#endif

// Conflict-free access: each thread in a warp accesses a different bank
__global__ void no_conflict_kernel(
    int iterations, int warmup,
    float* __restrict__ out,
    long long* __restrict__ out_cycles
) {
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    int lane = tid & 31;

    // Init
    smem[tid] = (float)tid;
    __syncthreads();

    float sum = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        sum += smem[tid];  // Each thread accesses unique bank
    }
    __syncthreads();

    // Timed — conflict-free: thread k reads bank k
    long long t0 = clock64();
    for (int i = 0; i < iterations; i++) {
        sum += smem[tid];
    }
    long long t1 = clock64();

    if (tid == 0) {
        out[0] = sum;
        out_cycles[0] = t1 - t0;
    }
}

// All-same-bank access: all threads in a warp access the same bank
__global__ void conflict_kernel(
    int iterations, int warmup,
    float* __restrict__ out,
    long long* __restrict__ out_cycles
) {
    __shared__ float smem[1024];
    int tid = threadIdx.x;
    int lane = tid & 31;

    // Init
    smem[tid] = (float)tid;
    __syncthreads();

    float sum = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        // All threads in warp access same bank (stride 32 = same bank)
        sum += smem[lane * 32 % 1024];
    }
    __syncthreads();

    // Timed — maximum conflict: all threads read from bank 0
    long long t0 = clock64();
    for (int i = 0; i < iterations; i++) {
        sum += smem[lane * 32 % 1024];  // All map to bank 0
    }
    long long t1 = clock64();

    if (tid == 0) {
        out[0] = sum;
        out_cycles[0] = t1 - t0;
    }
}

int main() {
    int block_size = BLOCK_SIZE;
    int iterations = ITERATIONS;
    int warmup = WARMUP;

    float* d_out;
    long long* d_cycles;
    cudaMalloc(&d_out, sizeof(float));
    cudaMalloc(&d_cycles, sizeof(long long));

    // ── No-conflict measurement ─────────────────────────────
    no_conflict_kernel<<<1, block_size>>>(iterations, warmup, d_out, d_cycles);
    cudaDeviceSynchronize();

    long long no_conflict_cycles;
    cudaMemcpy(&no_conflict_cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);
    double no_conflict_avg = (double)no_conflict_cycles / iterations;

    // ── Conflict measurement ────────────────────────────────
    conflict_kernel<<<1, block_size>>>(iterations, warmup, d_out, d_cycles);
    cudaDeviceSynchronize();

    long long conflict_cycles;
    cudaMemcpy(&conflict_cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);
    double conflict_avg = (double)conflict_cycles / iterations;

    // ── Compute penalty ─────────────────────────────────────
    double penalty = conflict_avg - no_conflict_avg;
    if (penalty < 0) penalty = 0;

    printf("RESULT:bank_conflict_penalty_cycles=%.2f\n", penalty);
    printf("UNIT:cycles\n");
    printf("METHOD:bank_conflict_comparison (no_conflict=%.1f, conflict=%.1f cycles/iter)\n",
           no_conflict_avg, conflict_avg);
    printf("ITERATIONS:%d\n", iterations);
    printf("WARMUP:%d\n", warmup);
    printf("NO_CONFLICT_CYCLES=%lld\n", no_conflict_cycles);
    printf("CONFLICT_CYCLES=%lld\n", conflict_cycles);
    printf("NO_CONFLICT_AVG=%.2f\n", no_conflict_avg);
    printf("CONFLICT_AVG=%.2f\n", conflict_avg);

    cudaFree(d_out);
    cudaFree(d_cycles);

    return 0;
}
