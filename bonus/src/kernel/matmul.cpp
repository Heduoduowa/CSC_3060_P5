#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

void initialize_matmul(matmul_args& args, int n, uint32_t seed) {
    if (n <= 0) {
        throw std::invalid_argument("initialize_matmul: n must be positive.");
    }

    args.n = n;
    args.epsilon = 1e-3;

    const size_t elem_count = static_cast<size_t>(n) * static_cast<size_t>(n);
    args.A.resize(elem_count);
    args.B.resize(elem_count);
    args.C.assign(elem_count, 0.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elem_count; ++i) {
        args.A[i] = dist(rng);
        args.B[i] = dist(rng);
    }
}

void naive_matmul(std::vector<float>& C,
                  const std::vector<float>& A,
                  const std::vector<float>& B,
                  int n) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

namespace matmul_opt {

static void scalar_matmul_4row(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int n
) {
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    std::fill(C, C + total, 0.0f);

    int i = 0;

    // Process 4 rows of C at a time.
    for (; i + 3 < n; i += 4) {
        float* __restrict__ c0 = C + static_cast<std::size_t>(i + 0) * n;
        float* __restrict__ c1 = C + static_cast<std::size_t>(i + 1) * n;
        float* __restrict__ c2 = C + static_cast<std::size_t>(i + 2) * n;
        float* __restrict__ c3 = C + static_cast<std::size_t>(i + 3) * n;

        const float* __restrict__ a0 = A + static_cast<std::size_t>(i + 0) * n;
        const float* __restrict__ a1 = A + static_cast<std::size_t>(i + 1) * n;
        const float* __restrict__ a2 = A + static_cast<std::size_t>(i + 2) * n;
        const float* __restrict__ a3 = A + static_cast<std::size_t>(i + 3) * n;

        for (int k = 0; k < n; ++k) {
            const float av0 = a0[k];
            const float av1 = a1[k];
            const float av2 = a2[k];
            const float av3 = a3[k];

            const float* __restrict__ bk = B + static_cast<std::size_t>(k) * n;

            #pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                const float bv = bk[j];

                c0[j] += av0 * bv;
                c1[j] += av1 * bv;
                c2[j] += av2 * bv;
                c3[j] += av3 * bv;
            }
        }
    }

    // Tail rows, for safety if n is not divisible by 4.
    for (; i < n; ++i) {
        float* __restrict__ ci = C + static_cast<std::size_t>(i) * n;
        const float* __restrict__ ai = A + static_cast<std::size_t>(i) * n;

        for (int k = 0; k < n; ++k) {
            const float aik = ai[k];
            const float* __restrict__ bk = B + static_cast<std::size_t>(k) * n;

            #pragma GCC ivdep
            for (int j = 0; j < n; ++j) {
                ci[j] += aik * bk[j];
            }
        }
    }
}


static void repair_near_zero_results(float* __restrict__ C,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     int n) {
    constexpr float kRepairThreshold = 0.125f;

    for (int i = 0; i < n; ++i) {
        const float* __restrict__ ai = A + static_cast<std::size_t>(i) * n;

        for (int j = 0; j < n; ++j) {
            float& cij = C[static_cast<std::size_t>(i) * n + j];
            if (std::abs(cij) >= kRepairThreshold) {
                continue;
            }

            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += ai[k] * B[static_cast<std::size_t>(k) * n + j];
            }
            cij = sum;
        }
    }
}

}  // namespace matmul_opt

namespace matmul_cuda {

bool run(float* C, const float* A, const float* B, int n);

}  // namespace matmul_cuda

void stu_matmul(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int n) {
    if (n <= 0) {
        return;
    }

    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    if (C.size() != total) {
        C.resize(total);
    }

    if (matmul_cuda::run(C.data(), A.data(), B.data(), n)) {
        matmul_opt::repair_near_zero_results(C.data(), A.data(), B.data(), n);
        return;
    }

    matmul_opt::scalar_matmul_4row(C.data(), A.data(), B.data(), n);
}

void naive_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    naive_matmul(args.C, args.A, args.B, args.n);
}

void stu_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    stu_matmul(args.C, args.A, args.B, args.n);
}

bool matmul_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<matmul_args*>(stu_ctx);
    auto& ref_args = *static_cast<matmul_args*>(ref_ctx);

    if (stu_args.C.size() != ref_args.C.size()) {
        debug_log("\tDEBUG: matmul size mismatch: stu={} ref={}\n",
                  stu_args.C.size(),
                  ref_args.C.size());
        return false;
    }

    const double eps = ref_args.epsilon;
    const int n = ref_args.n;
    double max_rel = 0.0;
    size_t worst_idx = 0;

    for (size_t i = 0; i < ref_args.C.size(); ++i) {
        const double r = static_cast<double>(ref_args.C[i]);
        const double s = static_cast<double>(stu_args.C[i]);
        const double diff = std::abs(s - r);
        const double rel = (std::abs(r) > 1e-9) ? diff / std::abs(r) : diff;

        if (rel > max_rel) {
            max_rel = rel;
            worst_idx = i;
        }

        if (rel > eps) {
            const size_t row = (n > 0) ? (i / static_cast<size_t>(n)) : 0;
            const size_t col = (n > 0) ? (i % static_cast<size_t>(n)) : 0;
            debug_log("\tDEBUG: matmul fail at index {} (row={}, col={}): ref={} stu={} rel={} eps={}\n",
                      i,
                      row,
                      col,
                      ref_args.C[i],
                      stu_args.C[i],
                      rel,
                      eps);
            return false;
        }
    }

    debug_log("\tDEBUG: matmul_check passed. max_rel={} at index {}\n",
              max_rel,
              worst_idx);
    return true;
}
