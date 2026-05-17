#include "grff.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <xmmintrin.h>

void initialize_grff(grff_args *args, const size_t size, const std::uint_fast64_t seed) {
    if (!args) return;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    args->a_features.resize(size);
    args->b_features.resize(size);
    args->c_features.resize(size);
    args->f_output.resize(size);

    for (size_t i = 0; i < size; ++i) {
        args->a_features[i] = dist(gen);
        args->b_features[i] = dist(gen);
        args->c_features[i] = dist(gen);
    }
}

// -------------------------------------------------------------------------
// Naive Implementation (A Simplified Gated Residual Feature Fusion (GRFF))
// -------------------------------------------------------------------------
void naive_grff(grff_args& args) {
    size_t n = args.a_features.size();
    
    // Intermediate buffers
    std::vector<float> G(n), A_prime(n), Smooth_A(n), B_prime(n), C_prime(n), H(n), E(n);

    // Stage 1: Gate
    for (size_t i = 0; i < n; ++i) 
        G[i] = 0.5f * ((args.a_features[i] * args.b_features[i]) / (1.0f + std::abs(args.a_features[i] * args.b_features[i])) + 1.0f);

    // Stage 2: Update A (Residual)
    for (size_t i = 0; i < n; ++i) 
        A_prime[i] = args.a_features[i] + G[i];

    // Stage 3: Global Feature Scaling
    float sum_a = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_a += A_prime[i];
    }
    float avg_a = sum_a / static_cast<float>(n);

    // Stage 4: Update A (Smooth)
    Smooth_A[0] = A_prime[0];
    for (size_t i = 1; i < n; ++i) {
        Smooth_A[i] = (A_prime[i] + A_prime[i-1]) * 0.5f; 
    }

    // Stage 5: Update B (Suppression)
    for (size_t i = 0; i < n; ++i) 
        B_prime[i] = args.b_features[i] * (1.0f - G[i]) * avg_a;

    // Stage 6: Context Integration 
    for (size_t i = 0; i < n; ++i) 
        C_prime[i] = args.c_features[i] + (Smooth_A[i] / (1.0f + std::abs(Smooth_A[i])));

    // Stage 7: Hidden Interaction
    for (size_t i = 0; i < n; ++i) 
        H[i] = Smooth_A[i] * C_prime[i];

    // Stage 8: Normalization
    for (size_t i = 0; i < n; ++i) 
        E[i] = (H[i] + B_prime[i]) / (1.0f + std::abs(Smooth_A[i]));

    // Stage 9: Final Output (ReLU)
    for (size_t i = 0; i < n; ++i) {
        float result = C_prime[i] - E[i];
        args.f_output[i] = std::max(result, 0.0f);
    }
}

// -------------------------------------------------------------------------
// TODO: Student Implementation
// -------------------------------------------------------------------------
namespace {

static inline float abs_float(float x) noexcept {
    return std::fabs(x);
}

static inline float relu_float(float x) noexcept {
    return x > 0.0f ? x : 0.0f;
}

static inline __m128 abs_ps(__m128 x) noexcept {
    const __m128 zero = _mm_setzero_ps();
    return _mm_max_ps(x, _mm_sub_ps(zero, x));
}

}  // namespace

void stu_grff(grff_args& args) {
    const std::size_t n = args.a_features.size();

    if (n == 0) {
        return;
    }

    const float* __restrict__ a = args.a_features.data();
    const float* __restrict__ b = args.b_features.data();
    const float* __restrict__ c = args.c_features.data();
    float* __restrict__ out = args.f_output.data();

    static thread_local std::vector<float> a_prime_storage;
    static thread_local std::vector<float> b_gate_storage;
    a_prime_storage.resize(n);
    b_gate_storage.resize(n);

    float* __restrict__ a_prime_buf = a_prime_storage.data();
    float* __restrict__ b_gate = b_gate_storage.data();

    float sum_a = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float prod = a[i] * b[i];
        const float g = 0.5f * (prod / (1.0f + abs_float(prod)) + 1.0f);
        const float a_prime = a[i] + g;

        a_prime_buf[i] = a_prime;
        b_gate[i] = b[i] * (1.0f - g);
        sum_a += a_prime;
    }

    const float avg_a = sum_a / static_cast<float>(n);
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 avg = _mm_set1_ps(avg_a);
    const __m128 zero = _mm_setzero_ps();

    {
        const float smooth = a_prime_buf[0];
        const float b_prime = b_gate[0] * avg_a;
        const float denom = 1.0f + abs_float(smooth);
        const float c_prime = c[0] + smooth / denom;
        const float e = (smooth * c_prime + b_prime) / denom;

        out[0] = relu_float(c_prime - e);
    }

    std::size_t i = 1;
    for (; i + 3 < n; i += 4) {
        const __m128 curr = _mm_loadu_ps(a_prime_buf + i);
        const __m128 prev = _mm_loadu_ps(a_prime_buf + i - 1);
        const __m128 smooth = _mm_mul_ps(_mm_add_ps(curr, prev), half);
        const __m128 denom = _mm_add_ps(one, abs_ps(smooth));
        const __m128 c_prime = _mm_add_ps(_mm_loadu_ps(c + i), _mm_div_ps(smooth, denom));
        const __m128 b_prime = _mm_mul_ps(_mm_loadu_ps(b_gate + i), avg);
        const __m128 e = _mm_div_ps(_mm_add_ps(_mm_mul_ps(smooth, c_prime), b_prime), denom);
        const __m128 result = _mm_max_ps(_mm_sub_ps(c_prime, e), zero);

        _mm_storeu_ps(out + i, result);
    }

    for (; i < n; ++i) {
        const float smooth = (a_prime_buf[i] + a_prime_buf[i - 1]) * 0.5f;
        const float b_prime = b_gate[i] * avg_a;
        const float denom = 1.0f + abs_float(smooth);
        const float c_prime = c[i] + smooth / denom;
        const float e = (smooth * c_prime + b_prime) / denom;

        out[i] = relu_float(c_prime - e);
    }
}

// -------------------------------------------------------------------------
// Wrappers and Checker
// -------------------------------------------------------------------------
void naive_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    naive_grff(args);
}

void stu_grff_wrapper(void *ctx) {
    auto &args = *static_cast<grff_args *>(ctx);
    stu_grff(args);
}

bool grff_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<grff_args *>(stu_ctx);
    auto &ref_args = *static_cast<grff_args *>(ref_ctx);
    const auto eps = ref_args.epsilon;
    const double atol = 1e-6;

    if (stu_args.f_output.size() != ref_args.f_output.size()) return false;

    for (size_t i = 0; i < ref_args.f_output.size(); ++i) {
        double r = static_cast<double>(ref_args.f_output[i]);
        double s = static_cast<double>(stu_args.f_output[i]);
        double err = std::abs(s - r);

        if (err > (atol + eps * std::abs(r))) {
            debug_log("DEBUG: GRFF fail at %zu: ref=%f stu=%f\n", i, r, s);
            return false;
        }
    }
    return true;
}
