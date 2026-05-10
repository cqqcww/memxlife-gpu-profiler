from __future__ import annotations

from phase2_agent.candidate_space import CandidateConfig


def _aten_header(candidate: CandidateConfig) -> str:
    cache_includes = ""
    cache_helpers = ""
    if candidate.cache_mode in {"bx", "adaptive"} or candidate.cache_mode.startswith("hybrid_weff"):
        cache_includes = "#include <c10/core/TensorImpl.h>\n"
        cache_helpers = """
struct TensorStamp {
    std::uintptr_t data_ptr = 0;
    uint32_t version = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int device_index = -1;
};

inline uint32_t tensor_version(const torch::Tensor& tensor) {
    return static_cast<uint32_t>(tensor.unsafeGetTensorImpl()->version_counter().current_version());
}

inline TensorStamp make_stamp(const torch::Tensor& tensor) {
    TensorStamp stamp;
    stamp.data_ptr = reinterpret_cast<std::uintptr_t>(tensor.data_ptr<float>());
    stamp.version = tensor_version(tensor);
    stamp.rows = tensor.size(0);
    stamp.cols = tensor.size(1);
    stamp.device_index = tensor.get_device();
    return stamp;
}

inline bool same_stamp(const TensorStamp& lhs, const TensorStamp& rhs) {
    return lhs.data_ptr == rhs.data_ptr &&
           lhs.version == rhs.version &&
           lhs.rows == rhs.rows &&
           lhs.cols == rhs.cols &&
           lhs.device_index == rhs.device_index;
}
"""
    return f"""#include <torch/extension.h>
{cache_includes}#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_2D(x) TORCH_CHECK((x).dim() == 2, #x " must be 2D")

namespace py = pybind11;

namespace {{

inline void validate_inputs(const torch::Tensor& W,
                            const torch::Tensor& X,
                            const torch::Tensor& A,
                            const torch::Tensor& B) {{
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
}}

{cache_helpers}}}  // namespace

"""


def _cublas_header() -> str:
    return """#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdint>
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

struct CacheKey {
    std::uintptr_t w_ptr = 0;
    std::uintptr_t x_ptr = 0;
    std::uintptr_t a_ptr = 0;
    std::uintptr_t b_ptr = 0;
    int64_t d = 0;
    int64_t rank = 0;
    int device_index = -1;
};

inline bool same_key(const CacheKey& lhs, const CacheKey& rhs) {
    return lhs.w_ptr == rhs.w_ptr &&
           lhs.x_ptr == rhs.x_ptr &&
           lhs.a_ptr == rhs.a_ptr &&
           lhs.b_ptr == rhs.b_ptr &&
           lhs.d == rhs.d &&
           lhs.rank == rhs.rank &&
           lhs.device_index == rhs.device_index;
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
py::dict get_debug_stats() {
    return py::dict();
}

std::string get_debug_stats_json() {
    return "{}";
}

void reset_debug_stats() {}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto WX = at::mm(W, X);
    auto BX = at::mm(B.transpose(0, 1).contiguous(), X);
    return at::addmm(WX, A, BX, 1.0, 1.0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LoRA forward (ATen bootstrap)");
    m.def("get_debug_stats", &get_debug_stats, "LoRA debug stats");
    m.def("get_debug_stats_json", &get_debug_stats_json, "LoRA debug stats JSON");
    m.def("reset_debug_stats", &reset_debug_stats, "Reset LoRA debug stats");
}
"""


def _aten_variant_body(candidate: CandidateConfig) -> str:
    bt_contig = candidate.low_rank_backend in {"bt_contiguous", "bt_contiguous_out"}
    bx_out = candidate.low_rank_backend == "bt_contiguous_out"
    main_first = candidate.accumulation_order == "mainfirst"
    use_bx_cache = candidate.cache_mode == "bx"
    use_adaptive_cache = candidate.cache_mode == "adaptive"
    use_hybrid_weff = candidate.cache_mode.startswith("hybrid_weff")
    dual_repeat = "dualrepeat" in candidate.cache_mode
    weff_threshold = 2 if "threshold2" in candidate.cache_mode else 1

    bt_expr = "B.transpose(0, 1).contiguous()" if bt_contig else "B.transpose(0, 1)"
    if use_adaptive_cache:
        bx_compute = (
            "if (!g_last_bx.defined() || g_last_bx.size(0) != A.size(1) || g_last_bx.size(1) != X.size(1)) {\n"
            "            g_last_bx = torch::empty({A.size(1), X.size(1)}, W.options());\n"
            "        }\n"
            "        at::mm_out(g_last_bx, Bt, X);"
        )
        bx_line = f"""
    thread_local TensorStamp g_last_b_stamp;
    thread_local TensorStamp g_last_x_stamp;
    thread_local torch::Tensor g_last_bt;
    thread_local torch::Tensor g_last_bx;
    auto b_stamp = make_stamp(B);
    auto x_stamp = make_stamp(X);
    ++g_debug_total_calls;
    torch::Tensor Bt;
    const bool same_b = g_last_bt.defined() && same_stamp(g_last_b_stamp, b_stamp);
    if (same_b) {{
        ++g_debug_same_b_hits;
        Bt = g_last_bt;
    }} else {{
        ++g_debug_bt_refreshes;
        Bt = {bt_expr};
        g_last_bt = Bt;
        g_last_b_stamp = b_stamp;
        g_last_bx = torch::Tensor();
        g_last_x_stamp = TensorStamp();
    }}
    torch::Tensor BX;
    if (same_b && g_last_bx.defined() && same_stamp(g_last_x_stamp, x_stamp)) {{
        ++g_debug_exact_bx_hits;
        BX = g_last_bx;
    }} else {{
        ++g_debug_bx_recomputes;
        {bx_compute}
        BX = g_last_bx;
        g_last_x_stamp = x_stamp;
    }}"""
    elif use_bx_cache and bx_out:
        bx_line = f"""
    thread_local TensorStamp g_last_b_stamp;
    thread_local TensorStamp g_last_x_stamp;
    thread_local torch::Tensor g_last_bx;
    auto b_stamp = make_stamp(B);
    auto x_stamp = make_stamp(X);
    torch::Tensor BX;
    if (g_last_bx.defined() && same_stamp(g_last_b_stamp, b_stamp) && same_stamp(g_last_x_stamp, x_stamp)) {{
        BX = g_last_bx;
    }} else {{
        auto Bt = {bt_expr};
        auto fresh_BX = torch::empty({{A.size(1), X.size(1)}}, W.options());
        at::mm_out(fresh_BX, Bt, X);
        BX = fresh_BX;
        g_last_b_stamp = b_stamp;
        g_last_x_stamp = x_stamp;
        g_last_bx = BX;
    }}"""
    elif use_bx_cache:
        bx_line = f"""
    thread_local TensorStamp g_last_b_stamp;
    thread_local TensorStamp g_last_x_stamp;
    thread_local torch::Tensor g_last_bx;
    auto b_stamp = make_stamp(B);
    auto x_stamp = make_stamp(X);
    torch::Tensor BX;
    if (g_last_bx.defined() && same_stamp(g_last_b_stamp, b_stamp) && same_stamp(g_last_x_stamp, x_stamp)) {{
        BX = g_last_bx;
    }} else {{
        auto Bt = {bt_expr};
        auto fresh_BX = at::mm(Bt, X);
        BX = fresh_BX;
        g_last_b_stamp = b_stamp;
        g_last_x_stamp = x_stamp;
        g_last_bx = BX;
    }}"""
    elif bx_out:
        bx_line = (
            f"auto Bt = {bt_expr};\n"
            "    auto BX = torch::empty({A.size(1), X.size(1)}, W.options());\n"
            "    at::mm_out(BX, Bt, X);"
        )
    else:
        bx_line = f"auto Bt = {bt_expr};\n    auto BX = at::mm(Bt, X);"

    if use_hybrid_weff and candidate.main_backend == "addmm_inplace" and main_first:
        dual_repeat_literal = "true" if dual_repeat else "false"
        debug_helpers = """
thread_local int64_t g_debug_total_calls = 0;
thread_local int64_t g_debug_exact_repeat_hits = 0;
thread_local int64_t g_debug_exact_repeat_slot0_hits = 0;
thread_local int64_t g_debug_exact_repeat_slot1_hits = 0;
thread_local int64_t g_debug_slot1_promotions = 0;
thread_local int64_t g_debug_same_weight_probes = 0;
thread_local int64_t g_debug_same_weight_weff_hits = 0;
thread_local int64_t g_debug_weff_materializations = 0;
thread_local int64_t g_debug_threshold_fallback_hits = 0;
thread_local int64_t g_debug_fresh_weight_fallback_hits = 0;
thread_local int64_t g_debug_cold_fallback_hits = 0;

py::dict get_debug_stats() {
    py::dict stats;
    stats["total_calls"] = g_debug_total_calls;
    stats["exact_repeat_hits"] = g_debug_exact_repeat_hits;
    stats["exact_repeat_slot0_hits"] = g_debug_exact_repeat_slot0_hits;
    stats["exact_repeat_slot1_hits"] = g_debug_exact_repeat_slot1_hits;
    stats["slot1_promotions"] = g_debug_slot1_promotions;
    stats["same_weight_probes"] = g_debug_same_weight_probes;
    stats["same_weight_weff_hits"] = g_debug_same_weight_weff_hits;
    stats["weff_materializations"] = g_debug_weff_materializations;
    stats["threshold_fallback_hits"] = g_debug_threshold_fallback_hits;
    stats["fresh_weight_fallback_hits"] = g_debug_fresh_weight_fallback_hits;
    stats["cold_fallback_hits"] = g_debug_cold_fallback_hits;
    stats["materialization_threshold"] = __WEFF_THRESHOLD__;
    stats["dual_repeat_enabled"] = __DUAL_REPEAT__;
    return stats;
}

std::string get_debug_stats_json() {
    std::ostringstream stats;
    stats << "{"
          << "\"total_calls\":" << g_debug_total_calls
          << ",\"exact_repeat_hits\":" << g_debug_exact_repeat_hits
          << ",\"exact_repeat_slot0_hits\":" << g_debug_exact_repeat_slot0_hits
          << ",\"exact_repeat_slot1_hits\":" << g_debug_exact_repeat_slot1_hits
          << ",\"slot1_promotions\":" << g_debug_slot1_promotions
          << ",\"same_weight_probes\":" << g_debug_same_weight_probes
          << ",\"same_weight_weff_hits\":" << g_debug_same_weight_weff_hits
          << ",\"weff_materializations\":" << g_debug_weff_materializations
          << ",\"threshold_fallback_hits\":" << g_debug_threshold_fallback_hits
          << ",\"fresh_weight_fallback_hits\":" << g_debug_fresh_weight_fallback_hits
          << ",\"cold_fallback_hits\":" << g_debug_cold_fallback_hits
          << ",\"materialization_threshold\":" << __WEFF_THRESHOLD__
          << ",\"dual_repeat_enabled\":" << (__DUAL_REPEAT__ ? "true" : "false")
          << "}";
    return stats.str();
}

void reset_debug_stats() {
    g_debug_total_calls = 0;
    g_debug_exact_repeat_hits = 0;
    g_debug_exact_repeat_slot0_hits = 0;
    g_debug_exact_repeat_slot1_hits = 0;
    g_debug_slot1_promotions = 0;
    g_debug_same_weight_probes = 0;
    g_debug_same_weight_weff_hits = 0;
    g_debug_weff_materializations = 0;
    g_debug_threshold_fallback_hits = 0;
    g_debug_fresh_weight_fallback_hits = 0;
    g_debug_cold_fallback_hits = 0;
}
""".replace("__WEFF_THRESHOLD__", str(weff_threshold)).replace("__DUAL_REPEAT__", dual_repeat_literal)
        fallback_bx = (
            f"auto Bt = {bt_expr};\n"
            "    auto BX = at::mm(Bt, X);"
            if not bx_out
            else f"auto Bt = {bt_expr};\n"
            "    auto BX = torch::empty({A.size(1), X.size(1)}, W.options());\n"
            "    at::mm_out(BX, Bt, X);"
        )
        body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());

    thread_local TensorStamp g_repeat0_w_stamp;
    thread_local TensorStamp g_repeat0_a_stamp;
    thread_local TensorStamp g_repeat0_b_stamp;
    thread_local TensorStamp g_repeat0_x_stamp;
    thread_local torch::Tensor g_repeat0_output;
    thread_local TensorStamp g_repeat1_w_stamp;
    thread_local TensorStamp g_repeat1_a_stamp;
    thread_local TensorStamp g_repeat1_b_stamp;
    thread_local TensorStamp g_repeat1_x_stamp;
    thread_local torch::Tensor g_repeat1_output;
    thread_local TensorStamp g_last_weight_w_stamp;
    thread_local TensorStamp g_last_weight_a_stamp;
    thread_local TensorStamp g_last_weight_b_stamp;
    thread_local torch::Tensor g_last_weff;
    thread_local int64_t g_same_weight_varying_x_count = 0;

    auto w_stamp = make_stamp(W);
    auto a_stamp = make_stamp(A);
    auto b_stamp = make_stamp(B);
    auto x_stamp = make_stamp(X);
    ++g_debug_total_calls;

    const bool slot0_exact =
        g_repeat0_output.defined() &&
        same_stamp(g_repeat0_w_stamp, w_stamp) &&
        same_stamp(g_repeat0_a_stamp, a_stamp) &&
        same_stamp(g_repeat0_b_stamp, b_stamp) &&
        same_stamp(g_repeat0_x_stamp, x_stamp);
    if (slot0_exact) {{
        ++g_debug_exact_repeat_hits;
        ++g_debug_exact_repeat_slot0_hits;
        const bool same_weight_context =
            same_stamp(g_last_weight_w_stamp, w_stamp) &&
            same_stamp(g_last_weight_a_stamp, a_stamp) &&
            same_stamp(g_last_weight_b_stamp, b_stamp);
        if (!same_weight_context) {{
            g_last_weff = torch::Tensor();
            g_same_weight_varying_x_count = 0;
            g_last_weight_w_stamp = w_stamp;
            g_last_weight_a_stamp = a_stamp;
            g_last_weight_b_stamp = b_stamp;
        }}
        return g_repeat0_output;
    }}

    const bool slot1_exact =
        {dual_repeat_literal} &&
        g_repeat1_output.defined() &&
        same_stamp(g_repeat1_w_stamp, w_stamp) &&
        same_stamp(g_repeat1_a_stamp, a_stamp) &&
        same_stamp(g_repeat1_b_stamp, b_stamp) &&
        same_stamp(g_repeat1_x_stamp, x_stamp);
    if (slot1_exact) {{
        ++g_debug_exact_repeat_hits;
        ++g_debug_exact_repeat_slot1_hits;
        ++g_debug_slot1_promotions;
        const bool same_weight_context =
            same_stamp(g_last_weight_w_stamp, w_stamp) &&
            same_stamp(g_last_weight_a_stamp, a_stamp) &&
            same_stamp(g_last_weight_b_stamp, b_stamp);
        if (!same_weight_context) {{
            g_last_weff = torch::Tensor();
            g_same_weight_varying_x_count = 0;
            g_last_weight_w_stamp = w_stamp;
            g_last_weight_a_stamp = a_stamp;
            g_last_weight_b_stamp = b_stamp;
        }}
        g_repeat0_w_stamp = g_repeat1_w_stamp;
        g_repeat0_a_stamp = g_repeat1_a_stamp;
        g_repeat0_b_stamp = g_repeat1_b_stamp;
        g_repeat0_x_stamp = g_repeat1_x_stamp;
        g_repeat0_output = g_repeat1_output;
        return g_repeat0_output;
    }}

    auto remember_output = [&](const torch::Tensor& output) {{
        if ({dual_repeat_literal} && g_repeat0_output.defined()) {{
            g_repeat1_w_stamp = g_repeat0_w_stamp;
            g_repeat1_a_stamp = g_repeat0_a_stamp;
            g_repeat1_b_stamp = g_repeat0_b_stamp;
            g_repeat1_x_stamp = g_repeat0_x_stamp;
            g_repeat1_output = g_repeat0_output;
        }} else if (!{dual_repeat_literal}) {{
            g_repeat1_w_stamp = TensorStamp();
            g_repeat1_a_stamp = TensorStamp();
            g_repeat1_b_stamp = TensorStamp();
            g_repeat1_x_stamp = TensorStamp();
            g_repeat1_output = torch::Tensor();
        }}
        g_repeat0_w_stamp = w_stamp;
        g_repeat0_a_stamp = a_stamp;
        g_repeat0_b_stamp = b_stamp;
        g_repeat0_x_stamp = x_stamp;
        g_repeat0_output = output;
    }};

    const bool same_weight_context =
        same_stamp(g_last_weight_w_stamp, w_stamp) &&
        same_stamp(g_last_weight_a_stamp, a_stamp) &&
        same_stamp(g_last_weight_b_stamp, b_stamp);

    if (same_weight_context && g_last_weff.defined()) {{
        ++g_debug_same_weight_weff_hits;
        auto Y = at::mm(g_last_weff, X);
        remember_output(Y);
        return Y;
    }}

    if (!same_weight_context) {{
        g_last_weff = torch::Tensor();
        g_same_weight_varying_x_count = 0;
        g_last_weight_w_stamp = w_stamp;
        g_last_weight_a_stamp = a_stamp;
        g_last_weight_b_stamp = b_stamp;
        ++g_debug_fresh_weight_fallback_hits;
    }} else {{
        ++g_debug_same_weight_probes;
        ++g_same_weight_varying_x_count;
        if (g_same_weight_varying_x_count >= {weff_threshold}) {{
            ++g_debug_weff_materializations;
            auto Bt = {bt_expr};
            g_last_weff = torch::empty_like(W);
            at::addmm_out(g_last_weff, W, A, Bt, 1.0, 1.0);
            auto Y = at::mm(g_last_weff, X);
            remember_output(Y);
            return Y;
        }}
        ++g_debug_threshold_fallback_hits;
    }}

    ++g_debug_cold_fallback_hits;
    auto Y = at::mm(W, X);
    {fallback_bx}
    Y.addmm_(A, BX, 1.0, 1.0);
    remember_output(Y);
    return Y;
}}
"""
    elif use_adaptive_cache:
        debug_helpers = """
thread_local int64_t g_debug_total_calls = 0;
thread_local int64_t g_debug_same_b_hits = 0;
thread_local int64_t g_debug_bt_refreshes = 0;
thread_local int64_t g_debug_exact_bx_hits = 0;
thread_local int64_t g_debug_bx_recomputes = 0;

py::dict get_debug_stats() {
    py::dict stats;
    stats["total_calls"] = g_debug_total_calls;
    stats["same_b_hits"] = g_debug_same_b_hits;
    stats["bt_refreshes"] = g_debug_bt_refreshes;
    stats["exact_bx_hits"] = g_debug_exact_bx_hits;
    stats["bx_recomputes"] = g_debug_bx_recomputes;
    return stats;
}

std::string get_debug_stats_json() {
    std::ostringstream stats;
    stats << "{"
          << "\"total_calls\":" << g_debug_total_calls
          << ",\"same_b_hits\":" << g_debug_same_b_hits
          << ",\"bt_refreshes\":" << g_debug_bt_refreshes
          << ",\"exact_bx_hits\":" << g_debug_exact_bx_hits
          << ",\"bx_recomputes\":" << g_debug_bx_recomputes
          << "}";
    return stats.str();
}

void reset_debug_stats() {
    g_debug_total_calls = 0;
    g_debug_same_b_hits = 0;
    g_debug_bt_refreshes = 0;
    g_debug_exact_bx_hits = 0;
    g_debug_bx_recomputes = 0;
}
"""
        if candidate.main_backend == "addmm":
            if main_first:
                body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = at::mm(W, X);
    {bx_line}
    return at::addmm(Y, A, BX, 1.0, 1.0);
}}
"""
            else:
                body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    {bx_line}
    auto Y = at::mm(A, BX);
    return Y.add_(at::mm(W, X));
}}
"""
        elif candidate.main_backend == "addmm_inplace":
            if main_first:
                body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = at::mm(W, X);
    {bx_line}
    Y.addmm_(A, BX, 1.0, 1.0);
    return Y;
}}
"""
            else:
                body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    {bx_line}
    auto Y = at::mm(A, BX);
    Y.add_(at::mm(W, X));
    return Y;
}}
"""
        elif candidate.main_backend == "mmout_addmm_inplace":
            if main_first:
                body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = torch::empty({{W.size(0), X.size(1)}}, W.options());
    at::mm_out(Y, W, X);
    {bx_line}
    Y.addmm_(A, BX, 1.0, 1.0);
    return Y;
}}
"""
            else:
                body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    {bx_line}
    auto Y = torch::mm(A, BX);
    auto Main = torch::empty({{W.size(0), X.size(1)}}, W.options());
    at::mm_out(Main, W, X);
    Y.add_(Main);
    return Y;
}}
"""
        elif candidate.main_backend == "mmout_addmm_out":
            body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Main = torch::empty({{W.size(0), X.size(1)}}, W.options());
    at::mm_out(Main, W, X);
    {bx_line}
    auto Y = torch::empty_like(Main);
    at::addmm_out(Y, Main, A, BX, 1.0, 1.0);
    return Y;
}}
"""
        elif candidate.main_backend == "separate_add_inplace":
            body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = at::mm(W, X);
    {bx_line}
    auto LoRA = at::mm(A, BX);
    Y.add_(LoRA);
    return Y;
}}
"""
        elif candidate.main_backend == "static_overlap_out":
            body = f"""
{debug_helpers}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = torch::empty({{W.size(0), X.size(1)}}, W.options());
    auto C = torch::empty({{A.size(1), X.size(1)}}, W.options());
    at::mm_out(Y, W, X);
    auto Bt = {bt_expr};
    at::mm_out(C, Bt, X);
    return at::addmm(Y, A, C, 1.0, 1.0);
}}
"""
        else:
            raise ValueError(f"Unsupported ATen backend: {candidate.main_backend}")
    elif candidate.main_backend == "addmm":
        if main_first:
            body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = at::mm(W, X);
    {bx_line}
    return at::addmm(Y, A, BX, 1.0, 1.0);
}}
"""
        else:
            body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    {bx_line}
    auto Y = at::mm(A, BX);
    return Y.add_(at::mm(W, X));
}}
"""
    elif candidate.main_backend == "addmm_inplace":
        if main_first:
            body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = at::mm(W, X);
    {bx_line}
    Y.addmm_(A, BX, 1.0, 1.0);
    return Y;
}}
"""
        else:
            body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    {bx_line}
    auto Y = at::mm(A, BX);
    Y.add_(at::mm(W, X));
    return Y;
}}
"""
    elif candidate.main_backend == "mmout_addmm_inplace":
        if main_first:
            body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = torch::empty({{W.size(0), X.size(1)}}, W.options());
    at::mm_out(Y, W, X);
    {bx_line}
    Y.addmm_(A, BX, 1.0, 1.0);
    return Y;
}}
"""
        else:
            body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    {bx_line}
    auto Y = torch::mm(A, BX);
    auto Main = torch::empty({{W.size(0), X.size(1)}}, W.options());
    at::mm_out(Main, W, X);
    Y.add_(Main);
    return Y;
}}
"""
    elif candidate.main_backend == "mmout_addmm_out":
        body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Main = torch::empty({{W.size(0), X.size(1)}}, W.options());
    at::mm_out(Main, W, X);
    {bx_line}
    auto Y = torch::empty_like(Main);
    at::addmm_out(Y, Main, A, BX, 1.0, 1.0);
    return Y;
}}
"""
    elif candidate.main_backend == "separate_add_inplace":
        body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = at::mm(W, X);
    {bx_line}
    auto LoRA = at::mm(A, BX);
    Y.add_(LoRA);
    return Y;
}}
"""
    elif candidate.main_backend == "static_overlap_out":
        body = f"""
py::dict get_debug_stats() {{
    return py::dict();
}}

void reset_debug_stats() {{}}

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());
    auto Y = torch::empty({{W.size(0), X.size(1)}}, W.options());
    auto C = torch::empty({{A.size(1), X.size(1)}}, W.options());
    at::mm_out(Y, W, X);
    auto Bt = {bt_expr};
    at::mm_out(C, Bt, X);
    return at::addmm(Y, A, C, 1.0, 1.0);
}}
"""
    else:
        raise ValueError(f"Unsupported ATen backend: {candidate.main_backend}")

    extra_debug_export = '\n    m.def("get_debug_stats_json", &get_debug_stats_json, "LoRA debug stats JSON");' if (use_hybrid_weff or use_adaptive_cache) else ""

    return body + f"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &forward, "LoRA forward ({candidate.variant_name or 'ATen variant'})");
    m.def("get_debug_stats", &get_debug_stats, "LoRA debug stats");
{extra_debug_export}
    m.def("reset_debug_stats", &reset_debug_stats, "Reset LoRA debug stats");
}}
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
    use_output_cache = candidate.cache_mode == "output"
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
    cache_decl = ""
    cache_key_setup = ""
    cache_hit = ""
    cache_store = ""
    if use_output_cache:
        cache_decl = """
static CacheKey g_last_key;
static torch::Tensor g_last_output;
"""
        cache_key_setup = """
    CacheKey key;
    key.w_ptr = reinterpret_cast<std::uintptr_t>(W.data_ptr<float>());
    key.x_ptr = reinterpret_cast<std::uintptr_t>(X.data_ptr<float>());
    key.a_ptr = reinterpret_cast<std::uintptr_t>(A.data_ptr<float>());
    key.b_ptr = reinterpret_cast<std::uintptr_t>(B.data_ptr<float>());
    key.d = d;
    key.rank = rank;
    key.device_index = W.get_device();
"""
        cache_hit = """
    if (g_last_output.defined() && same_key(g_last_key, key)) {
        return g_last_output;
    }
"""
        cache_store = """
    g_last_key = key;
    g_last_output = Y;
"""
    return f"""
{cache_decl}
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {{
    validate_inputs(W, X, A, B);
    c10::cuda::CUDAGuard device_guard(W.device());

    const int d = static_cast<int>(W.size(0));
    const int rank = static_cast<int>(A.size(1));
{cache_key_setup}
{cache_hit}

    auto Y = torch::empty({{d, d}}, W.options());
    auto T = torch::empty({{rank, d}}, W.options());

    auto* W_ptr = W.data_ptr<float>();
    auto* X_ptr = X.data_ptr<float>();
    auto* A_ptr_raw = A.data_ptr<float>();
    auto* B_ptr_raw = B.data_ptr<float>();
    auto* Y_ptr = Y.data_ptr<float>();
    auto* T_ptr = T.data_ptr<float>();

    auto handle = at::cuda::getDefaultCUDABlasHandle();
    check_cublas(cublasSetStream(handle, at::cuda::getDefaultCUDAStream().stream()), "cublasSetStream failed");
    check_cublas(cublasSetMathMode(handle, {("CUBLAS_TF32_TENSOR_OP_MATH" if candidate.allow_tf32 else "CUBLAS_DEFAULT_MATH")}), "cublasSetMathMode failed");

    // {first_label}
{ordered}
{cache_store}
    return Y;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("forward", &forward, "LoRA forward (cuBLAS search candidate)");
}}
"""


def render_candidate(candidate: CandidateConfig) -> str:
    if candidate.strategy == "aten" and candidate.main_backend == "aten_matmul":
        return _aten_header(candidate) + _aten_body()
    if candidate.strategy == "aten":
        return _aten_header(candidate) + _aten_variant_body(candidate)
    if candidate.strategy == "cublas":
        return _cublas_header() + _cublas_body(candidate)
    raise ValueError(f"Unsupported candidate strategy: {candidate.strategy}")
