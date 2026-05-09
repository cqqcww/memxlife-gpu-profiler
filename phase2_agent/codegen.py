from __future__ import annotations

from phase2_agent.candidate_space import CandidateConfig


def _header() -> str:
    return """#include <torch/extension.h>
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

"""


def _aten_body() -> str:
    return """
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
"""


def _cublas_dispatch(name: str, allow_tf32: bool) -> str:
    if name == "sgemm":
        return "gemm_row_major(handle, trans_a, trans_b, m, n, k, A_ptr, B_ptr, C_ptr, alpha, beta);"
    if name == "gemm_ex_tf32":
        tf32 = "true" if allow_tf32 else "false"
        return (
            "gemm_ex_row_major(handle, trans_a, trans_b, m, n, k, "
            f"A_ptr, B_ptr, C_ptr, alpha, beta, {tf32});"
        )
    raise ValueError(f"Unsupported backend: {name}")


def _cublas_body(candidate: CandidateConfig) -> str:
    main_call = _cublas_dispatch(candidate.main_backend, candidate.allow_tf32)
    low_rank_call = _cublas_dispatch(candidate.low_rank_backend, candidate.allow_tf32)
    wx_first = candidate.accumulation_order == "wx_then_lora"
    first_beta = "0.0f"
    second_beta = "1.0f" if wx_first else "0.0f"
    first_label = "Main GEMM then low-rank update" if wx_first else "Low-rank update then main GEMM"
    first_pass = f"""
    {{
        const bool trans_a = false;
        const bool trans_b = false;
        const int m = d;
        const int n = d;
        const int k = d;
        const float* A_ptr = W_ptr;
        const float* B_ptr = X_ptr;
        float* C_ptr = Y_ptr;
        const float alpha = 1.0f;
        const float beta = {first_beta};
        {main_call}
    }}
"""
    low_rank_prepare = """
    {
        const bool trans_a = true;
        const bool trans_b = false;
        const int m = rank;
        const int n = d;
        const int k = d;
        const float* A_ptr = B_ptr_raw;
        const float* B_ptr = X_ptr;
        float* C_ptr = T_ptr;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        LOW_RANK_BACKEND
    }
    {
        const bool trans_a = false;
        const bool trans_b = false;
        const int m = d;
        const int n = d;
        const int k = rank;
        const float* A_ptr = A_ptr_raw;
        const float* B_ptr = T_ptr;
        float* C_ptr = Y_ptr;
        const float alpha = 1.0f;
        const float beta = SECOND_BETA;
        LOW_RANK_BACKEND
    }
""".replace("LOW_RANK_BACKEND", low_rank_call).replace("SECOND_BETA", second_beta)
    if wx_first:
        ordered = first_pass + low_rank_prepare
    else:
        ordered = low_rank_prepare + first_pass.replace(first_beta, second_beta)
    tf32 = "true" if candidate.allow_tf32 else "false"
    return f"""
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());

    auto Wc = W.contiguous();
    auto Xc = X.contiguous();
    auto Ac = A.contiguous();
    auto Bc = B.contiguous();

    const int d = static_cast<int>(Wc.size(0));
    const int rank = static_cast<int>(Ac.size(1));

    auto Y = torch::empty({{d, d}}, Wc.options());
    auto T = torch::empty({{rank, d}}, Wc.options());

    auto* W_ptr = Wc.data_ptr<float>();
    auto* X_ptr = Xc.data_ptr<float>();
    auto* A_ptr_raw = Ac.data_ptr<float>();
    auto* B_ptr_raw = Bc.data_ptr<float>();
    auto* Y_ptr = Y.data_ptr<float>();
    auto* T_ptr = T.data_ptr<float>();

    auto handle = at::cuda::getDefaultCUDABlasHandle();
    check_cublas(cublasSetStream(handle, at::cuda::getDefaultCUDAStream().stream()), "cublasSetStream failed");
    check_cublas(cublasSetMathMode(handle, {("CUBLAS_TF32_TENSOR_OP_MATH" if candidate.allow_tf32 else "CUBLAS_DEFAULT_MATH")}), "cublasSetMathMode failed");

    // {first_label}
{ordered}
    return Y;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &forward, "LoRA forward (cuBLAS search candidate)");
}}
"""


def render_candidate(candidate: CandidateConfig) -> str:
    if candidate.strategy == "aten":
        return _header() + _aten_body()
    if candidate.strategy == "cublas":
        return _header() + _cublas_body(candidate)
    raise ValueError(f"Unsupported candidate strategy: {candidate.strategy}")
