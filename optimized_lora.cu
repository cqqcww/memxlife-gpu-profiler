#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <sstream>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_2D(x) TORCH_CHECK((x).dim() == 2, #x " must be 2D")

namespace {

inline void validate_inputs(const torch::Tensor& W,
                            const torch::Tensor& X,
                            const torch::Tensor& A,
                            const torch::Tensor& B) {
    CHECK_CUDA(W);
    CHECK_CUDA(X);
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(W);
    CHECK_CONTIGUOUS(X);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    CHECK_FLOAT(W);
    CHECK_FLOAT(X);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);
    CHECK_2D(W);
    CHECK_2D(X);
    CHECK_2D(A);
    CHECK_2D(B);

    TORCH_CHECK(W.size(0) == W.size(1), "W must be square");
    TORCH_CHECK(X.size(0) == X.size(1), "X must be square");
    TORCH_CHECK(W.size(1) == X.size(0), "W and X dimension mismatch");
    TORCH_CHECK(A.size(0) == W.size(0), "A rows must match W rows");
    TORCH_CHECK(B.size(0) == X.size(0), "B rows must match X rows");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B rank mismatch");
    TORCH_CHECK(A.size(1) == 16, "Expected fixed LoRA rank 16");
}

inline void check_cublas(cublasStatus_t status, const char* what) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what);
}

inline void gemm_row_major(cublasHandle_t handle,
                           bool trans_a,
                           bool trans_b,
                           int m,
                           int n,
                           int k,
                           const float* A,
                           const float* B,
                           float* C,
                           float alpha,
                           float beta) {
    const cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = trans_a ? m : k;
    const int ldb = trans_b ? k : n;
    const int ldc = n;
    check_cublas(
        cublasSgemm(handle, op_b, op_a, n, m, k, &alpha, B, ldb, A, lda, &beta, C, ldc),
        "cublasSgemm failed");
}

inline void gemm_ex_row_major(cublasHandle_t handle,
                              bool trans_a,
                              bool trans_b,
                              int m,
                              int n,
                              int k,
                              const float* A,
                              const float* B,
                              float* C,
                              float alpha,
                              float beta,
                              bool allow_tf32) {
    const cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = trans_a ? m : k;
    const int ldb = trans_b ? k : n;
    const int ldc = n;
    const cublasComputeType_t compute_type =
        allow_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    check_cublas(
        cublasGemmEx(handle,
                     op_b,
                     op_a,
                     n,
                     m,
                     k,
                     &alpha,
                     B,
                     CUDA_R_32F,
                     ldb,
                     A,
                     CUDA_R_32F,
                     lda,
                     &beta,
                     C,
                     CUDA_R_32F,
                     ldc,
                     compute_type,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx failed");
}

}  // namespace


torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto WX = at::matmul(W, X);
    auto BX = at::matmul(B.transpose(0, 1).contiguous(), X);
    return at::addmm(WX, A, BX, 1.0, 1.0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (ATen bootstrap)");
}
