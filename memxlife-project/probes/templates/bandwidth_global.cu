// bandwidth_global.cu — Measure peak global memory (VRAM) bandwidth
// Uses streaming read + write with large contiguous buffers
//
// Parameters:
//   {{data_size_bytes}} — buffer size in bytes
//   {{iterations}}      — number of measurement iterations
//   {{warmup}}          — warmup iterations

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef DATA_SIZE_BYTES
#define DATA_SIZE_BYTES (256 * 1024 * 1024)  // 256 MB
#endif
#ifndef ITERATIONS
#define ITERATIONS 20
#endif
#ifndef WARMUP
#define WARMUP 5
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Simple copy kernel — each thread copies one float4 per iteration
__global__ void bandwidth_copy_kernel(
    const float4* __restrict__ src,
    float4* __restrict__ dst,
    int n_float4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n_float4; i += stride) {
        dst[i] = src[i];
    }
}

// Read-only kernel — sum elements to prevent optimization
__global__ void bandwidth_read_kernel(
    const float4* __restrict__ src,
    float* __restrict__ out,
    int n_float4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = idx; i < n_float4; i += stride) {
        float4 v = src[i];
        sum += v.x + v.y + v.z + v.w;
    }
    // Prevent dead code elimination
    if (idx == 0) *out = sum;
}

int main() {
    size_t data_size = DATA_SIZE_BYTES;
    int iterations = ITERATIONS;
    int warmup = WARMUP;

    int n_float4 = data_size / sizeof(float4);

    // Allocate device memory
    float4 *d_src, *d_dst;
    float *d_out;
    cudaMalloc(&d_src, data_size);
    cudaMalloc(&d_dst, data_size);
    cudaMalloc(&d_out, sizeof(float));

    // Initialize source with pattern
    cudaMemset(d_src, 1, data_size);
    cudaMemset(d_dst, 0, data_size);

    // Calculate grid size
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_blocks = prop.multiProcessorCount * 4;

    // ── Warmup ──────────────────────────────────────────────
    for (int i = 0; i < warmup; i++) {
        bandwidth_copy_kernel<<<num_blocks, BLOCK_SIZE>>>(d_src, d_dst, n_float4);
    }
    cudaDeviceSynchronize();

    // ── Measure copy bandwidth (read + write) ───────────────
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        bandwidth_copy_kernel<<<num_blocks, BLOCK_SIZE>>>(d_src, d_dst, n_float4);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float copy_ms;
    cudaEventElapsedTime(&copy_ms, start, stop);
    double copy_sec = copy_ms / 1000.0;

    // Copy moves 2x data (read src + write dst)
    double total_bytes_copy = 2.0 * (double)data_size * iterations;
    double copy_bw_gb_s = total_bytes_copy / copy_sec / 1e9;

    // ── Measure read-only bandwidth ─────────────────────────
    for (int i = 0; i < warmup; i++) {
        bandwidth_read_kernel<<<num_blocks, BLOCK_SIZE>>>(d_src, d_out, n_float4);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        bandwidth_read_kernel<<<num_blocks, BLOCK_SIZE>>>(d_src, d_out, n_float4);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float read_ms;
    cudaEventElapsedTime(&read_ms, start, stop);
    double read_sec = read_ms / 1000.0;

    double total_bytes_read = (double)data_size * iterations;
    double read_bw_gb_s = total_bytes_read / read_sec / 1e9;

    // Report the higher of copy and read bandwidth
    double max_bw = (copy_bw_gb_s > read_bw_gb_s) ? copy_bw_gb_s : read_bw_gb_s;

    printf("RESULT:max_global_mem_bandwidth_gb_s=%.2f\n", max_bw);
    printf("UNIT:GB/s\n");
    printf("METHOD:streaming copy+read (data_size=%zu bytes)\n", data_size);
    printf("ITERATIONS:%d\n", iterations);
    printf("WARMUP:%d\n", warmup);
    printf("COPY_BW_GB_S=%.2f\n", copy_bw_gb_s);
    printf("READ_BW_GB_S=%.2f\n", read_bw_gb_s);
    printf("COPY_TIME_MS=%.3f\n", copy_ms);
    printf("READ_TIME_MS=%.3f\n", read_ms);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_out);

    return 0;
}
