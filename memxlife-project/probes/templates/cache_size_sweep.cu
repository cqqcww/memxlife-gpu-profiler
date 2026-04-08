// cache_size_sweep.cu — Detect L2 cache size via latency cliff
// Sweeps data sizes and measures pointer chase latency at each size.
// The latency cliff indicates the boundary where data exceeds L2.
//
// Parameters:
//   {{min_kb}}     — minimum sweep size in KB
//   {{max_kb}}     — maximum sweep size in KB
//   {{steps}}      — number of sweep steps
//   {{iterations}} — pointer chase iterations per step

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef MIN_KB
#define MIN_KB 64
#endif
#ifndef MAX_KB
#define MAX_KB 65536
#endif
#ifndef STEPS
#define STEPS 24
#endif
#ifndef ITERATIONS
#define ITERATIONS 500
#endif
#ifndef WARMUP
#define WARMUP 100
#endif

// Single-thread pointer chase kernel
__global__ void chase_kernel(
    const int* __restrict__ chain,
    int start_idx,
    int num_steps,
    int warmup_steps,
    long long* __restrict__ out_cycles
) {
    int idx = start_idx;

    // Warmup
    for (int i = 0; i < warmup_steps; i++) {
        idx = __ldg(&chain[idx]);
    }

    // Timed
    long long t0 = clock64();
    for (int i = 0; i < num_steps; i++) {
        idx = __ldg(&chain[idx]);
    }
    long long t1 = clock64();

    if (idx == -999) printf("%d", idx);
    *out_cycles = t1 - t0;
}

void build_random_chain(int* chain, int n) {
    for (int i = 0; i < n; i++) chain[i] = (i + 1) % n;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % i;
        int tmp = chain[i];
        chain[i] = chain[j];
        chain[j] = tmp;
    }
}

int main() {
    int iterations = ITERATIONS;
    int warmup = WARMUP;
    int steps = STEPS;
    size_t min_bytes = (size_t)MIN_KB * 1024;
    size_t max_bytes = (size_t)MAX_KB * 1024;

    // Logarithmic sweep
    double log_min = log2((double)min_bytes);
    double log_max = log2((double)max_bytes);
    double log_step = (log_max - log_min) / (steps - 1);

    // Allocate max size on host and device
    int max_elements = max_bytes / sizeof(int);
    int* h_chain = (int*)malloc(max_bytes);
    int* d_chain;
    cudaMalloc(&d_chain, max_bytes);

    long long* d_cycles;
    cudaMalloc(&d_cycles, sizeof(long long));

    srand(42);

    // Arrays to store sweep results
    double sizes_kb[256];
    double latencies[256];
    int n_points = 0;

    fprintf(stderr, "Sweeping %d sizes from %zu KB to %zu KB...\n",
            steps, min_bytes / 1024, max_bytes / 1024);

    for (int s = 0; s < steps; s++) {
        size_t data_size = (size_t)pow(2.0, log_min + s * log_step);
        // Round to cache line boundary (128 bytes)
        data_size = ((data_size + 127) / 128) * 128;
        if (data_size < 256) data_size = 256;
        if (data_size > max_bytes) data_size = max_bytes;

        int n_elements = data_size / sizeof(int);
        if (n_elements < 2) continue;

        // Build chain for this size
        build_random_chain(h_chain, n_elements);
        cudaMemcpy(d_chain, h_chain, data_size, cudaMemcpyHostToDevice);

        // Run pointer chase
        chase_kernel<<<1, 1>>>(d_chain, 0, iterations, warmup, d_cycles);
        cudaDeviceSynchronize();

        long long cycles;
        cudaMemcpy(&cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost);

        double avg_latency = (double)cycles / (double)iterations;
        double size_kb = (double)data_size / 1024.0;

        sizes_kb[n_points] = size_kb;
        latencies[n_points] = avg_latency;
        n_points++;

        fprintf(stderr, "  size=%8.1f KB  latency=%8.1f cycles\n", size_kb, avg_latency);
    }

    // ── Detect L2 cache boundary ────────────────────────────
    // Find the biggest latency jump (cliff)
    double max_jump = 0;
    int cliff_idx = -1;
    for (int i = 1; i < n_points; i++) {
        double jump = latencies[i] - latencies[i - 1];
        // Relative jump
        double rel_jump = (latencies[i - 1] > 0) ? jump / latencies[i - 1] : 0;
        if (rel_jump > max_jump && rel_jump > 0.3) {  // >30% increase
            max_jump = rel_jump;
            cliff_idx = i;
        }
    }

    double l2_size_kb;
    if (cliff_idx > 0) {
        // L2 size is approximately the size just before the cliff
        l2_size_kb = sizes_kb[cliff_idx - 1];
        // Round to nearest power of 2 or common L2 size
        double log2_size = log2(l2_size_kb);
        l2_size_kb = pow(2.0, round(log2_size));
    } else {
        // No clear cliff found — estimate from where latency starts increasing
        // Use the midpoint where latency exceeds 1.5x the minimum
        double min_lat = latencies[0];
        l2_size_kb = sizes_kb[n_points - 1];  // default to max
        for (int i = 1; i < n_points; i++) {
            if (latencies[i] > min_lat * 1.5) {
                l2_size_kb = sizes_kb[i - 1];
                break;
            }
        }
    }

    printf("RESULT:l2_cache_size_kb=%.0f\n", l2_size_kb);
    printf("UNIT:KB\n");
    printf("METHOD:latency_cliff_sweep (log sweep %d points, %d-%d KB)\n",
           steps, MIN_KB, MAX_KB);
    printf("ITERATIONS:%d\n", iterations);
    printf("WARMUP:%d\n", warmup);

    // Also output raw sweep data for analysis
    for (int i = 0; i < n_points; i++) {
        printf("SWEEP:size_kb=%.1f,latency_cycles=%.1f\n", sizes_kb[i], latencies[i]);
    }

    // Cleanup
    cudaFree(d_chain);
    cudaFree(d_cycles);
    free(h_chain);

    return 0;
}
