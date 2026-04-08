// shmem_capacity.cu — Detect maximum shared memory per block
// Binary search: try launching kernels with increasing shared memory
// until cudaErrorInvalidConfiguration
//
// Parameters:
//   {{start_kb}} — starting size in KB
//   {{max_kb}}   — maximum size to try in KB
//   {{step_kb}}  — step size in KB

#include <stdio.h>
#include <cuda_runtime.h>

#ifndef START_KB
#define START_KB 16
#endif
#ifndef MAX_KB
#define MAX_KB 228
#endif
#ifndef STEP_KB
#define STEP_KB 4
#endif

// Dummy kernel that uses dynamic shared memory
__global__ void shmem_test_kernel(int n_bytes) {
    extern __shared__ char smem[];
    // Touch shared memory to ensure it's actually allocated
    int tid = threadIdx.x;
    if (tid < n_bytes) {
        smem[tid] = (char)tid;
    }
    __syncthreads();
    // Prevent DCE
    if (tid == 0 && smem[0] == -99) printf("x");
}

int main() {
    int start_kb = START_KB;
    int max_kb = MAX_KB;
    int step_kb = STEP_KB;

    // First, check what the device reports (but don't trust it fully)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int reported_shmem_kb = (int)(prop.sharedMemPerBlock / 1024);
    int reported_shmem_multiprocessor_kb = (int)(prop.sharedMemPerMultiprocessor / 1024);

    fprintf(stderr, "Device reports: sharedMemPerBlock=%d KB, sharedMemPerMultiprocessor=%d KB\n",
            reported_shmem_kb, reported_shmem_multiprocessor_kb);

    // Try to set max dynamic shared memory
    // For newer GPUs, we need to opt in to extended shared memory
    cudaFuncSetAttribute(
        shmem_test_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_kb * 1024
    );

    // Binary search for actual max
    int lo = start_kb;
    int hi = max_kb;
    int actual_max_kb = start_kb;

    // First, linear scan to find approximate range
    for (int size_kb = start_kb; size_kb <= max_kb; size_kb += step_kb) {
        int size_bytes = size_kb * 1024;

        // Try to set the attribute for this size
        cudaError_t attr_err = cudaFuncSetAttribute(
            shmem_test_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            size_bytes
        );

        if (attr_err != cudaSuccess) {
            fprintf(stderr, "  %d KB: cudaFuncSetAttribute failed: %s\n",
                    size_kb, cudaGetErrorString(attr_err));
            break;
        }

        // Try launching
        shmem_test_kernel<<<1, 32, size_bytes>>>(size_bytes);
        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "  %d KB: launch failed: %s\n", size_kb, cudaGetErrorString(err));
            cudaDeviceReset();
            break;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "  %d KB: sync failed: %s\n", size_kb, cudaGetErrorString(err));
            cudaDeviceReset();
            break;
        }

        actual_max_kb = size_kb;
        fprintf(stderr, "  %d KB: OK\n", size_kb);
    }

    printf("RESULT:max_shmem_per_block_kb=%d\n", actual_max_kb);
    printf("UNIT:KB\n");
    printf("METHOD:shmem_alloc_probe (linear scan %d-%d KB, step %d KB)\n",
           start_kb, max_kb, step_kb);
    printf("ITERATIONS:1\n");
    printf("WARMUP:0\n");
    printf("REPORTED_SHMEM_PER_BLOCK_KB=%d\n", reported_shmem_kb);
    printf("REPORTED_SHMEM_PER_SM_KB=%d\n", reported_shmem_multiprocessor_kb);

    return 0;
}
