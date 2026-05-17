#include "blackscholes.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <random>

#define inv_sqrt_2xPI 0.39894228040143270286f
#define p_val 0.2316419f
#define coefficient_a1 0.319381530f
#define coefficient_a2 -0.356563782f
#define coefficient_a3 1.781477937f
#define coefficient_a4 -1.821255978f
#define coefficient_a5 1.330274429f

void initialize_blackscholes(blackscholes_args &args,
                             std::size_t n,
                             std::uint32_t seed) {
    args.call_option_price.assign(n, 0.0f);
    args.put_option_price.assign(n, 0.0f);
    args.epsilon = 5e-3;

    args.spot_price.resize(n);
    args.strike.resize(n);
    args.rate.resize(n);
    args.volatility.resize(n);
    args.time.resize(n);

    std::mt19937 rng(seed);

    std::uniform_real_distribution<float> spot_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> strike_dist(50.0f, 99.9f);
    std::uniform_real_distribution<float> rate_dist(0.0275f, 0.1f);
    std::uniform_real_distribution<float> vol_dist(0.05f, 0.6f);
    std::uniform_real_distribution<float> time_dist(0.1f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        args.spot_price[i] = spot_dist(rng);
        args.strike[i] = strike_dist(rng);
        args.rate[i] = rate_dist(rng);
        args.volatility[i] = vol_dist(rng);
        args.time[i] = time_dist(rng);
    }
}

void CNDF(float &InputX, float &OutputX) {
    int sign = 0;
    float x = InputX;

    if (x < 0.0f) {
        x = -x;
        sign = 1;
    }

    const float xNPrimeofX =
        std::exp(-0.5f * x * x) * inv_sqrt_2xPI;

    const float k = 1.0f / (1.0f + p_val * x);

    const float k_2 = k * k;
    const float k_3 = k_2 * k;
    const float k_4 = k_3 * k;
    const float k_5 = k_4 * k;

    float local = k * coefficient_a1;
    local += k_2 * coefficient_a2;
    local += k_3 * coefficient_a3;
    local += k_4 * coefficient_a4;
    local += k_5 * coefficient_a5;

    local = 1.0f - local * xNPrimeofX;

    OutputX = sign ? (1.0f - local) : local;
}

static inline void naive_BlkSchls_one(float &CallOptionPrice,
                                      float &PutOptionPrice,
                                      float spotPrice,
                                      float strike,
                                      float rate,
                                      float volatility,
                                      float time) {
    const float xSqrtTime = std::sqrt(time);
    const float xLogTerm = std::log(spotPrice / strike);
    const float xPowerTerm = 0.5f * volatility * volatility;

    float xD1 = (rate + xPowerTerm) * time + xLogTerm;

    const float xDen = volatility * xSqrtTime;

    xD1 = xD1 / xDen;

    const float xD2 = xD1 - xDen;

    float d1 = xD1;
    float d2 = xD2;

    float NofXd1 = 0.0f;
    float NofXd2 = 0.0f;

    CNDF(d1, NofXd1);
    CNDF(d2, NofXd2);

    const float FutureValueX = strike * std::exp(-(rate) * (time));

    CallOptionPrice = (spotPrice * NofXd1) - (FutureValueX * NofXd2);

    const float NegNofXd1 = 1.0f - NofXd1;
    const float NegNofXd2 = 1.0f - NofXd2;

    PutOptionPrice =
        (FutureValueX * NegNofXd2) - (spotPrice * NegNofXd1);
}

void naive_BlkSchls(std::vector<float> &CallOptionPrice,
                    std::vector<float> &PutOptionPrice,
                    const std::vector<float> &spotPrice,
                    const std::vector<float> &strike,
                    const std::vector<float> &rate,
                    const std::vector<float> &volatility,
                    const std::vector<float> &time) {
    const size_t n = spotPrice.size();

    for (size_t i = 0; i < n; ++i) {
        naive_BlkSchls_one(CallOptionPrice[i],
                           PutOptionPrice[i],
                           spotPrice[i],
                           strike[i],
                           rate[i],
                           volatility[i],
                           time[i]);
    }
}

namespace {

static inline float fast_log_scalar(float x) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(x);

    int e = static_cast<int>((bits >> 23) & 0xff) - 127;

    bits = (bits & 0x007fffffu) | 0x3f800000u;
    float m = std::bit_cast<float>(bits);

    constexpr float sqrt2 = 1.4142135623730951f;

    if (m > sqrt2) {
        m *= 0.5f;
        ++e;
    }

    const float z = (m - 1.0f) / (m + 1.0f);
    const float z2 = z * z;

    const float log_m =
        2.0f * z *
        (1.0f + z2 *
        (0.3333333333333333f + z2 *
        (0.2f + z2 *
        (0.14285714285714285f + z2 *
        0.1111111111111111f))));

    return static_cast<float>(e) * 0.6931471805599453f + log_m;
}

static inline float fast_exp5_scalar(float x) noexcept {
    if (x <= -87.0f) {
        return 0.0f;
    }

    if (x >= 88.0f) {
        return std::bit_cast<float>(0x7f800000u);
    }

    constexpr float inv_ln2 = 1.4426950408889634f;
    constexpr float ln2 = 0.6931471805599453f;

    const int k = static_cast<int>(x * inv_ln2);
    const float r = x - static_cast<float>(k) * ln2;

    const float p =
        1.0f + r *
        (1.0f + r *
        (0.5f + r *
        (0.16666666666666666f + r *
        (0.041666666666666664f + r *
        (0.008333333333333333f + r *
        0.001388888888888889f)))));

    const std::uint32_t bits =
        static_cast<std::uint32_t>(k + 127) << 23;

    return p * std::bit_cast<float>(bits);
}

static inline float fast_CNDF_scalar(float x) noexcept {
    const bool sign = x < 0.0f;

    if (sign) {
        x = -x;
    }

    if (x > 8.0f) {
        return sign ? 0.0f : 1.0f;
    }

    const float xNPrimeofX =
        fast_exp5_scalar(-0.5f * x * x) * inv_sqrt_2xPI;

    const float k = 1.0f / (1.0f + p_val * x);

    const float poly =
        (((((coefficient_a5 * k + coefficient_a4) * k
            + coefficient_a3) * k
            + coefficient_a2) * k
            + coefficient_a1) * k);

    const float local = 1.0f - poly * xNPrimeofX;

    return sign ? (1.0f - local) : local;
}

static inline void stu_BlkSchls_one_scalar(float &CallOptionPrice,
                                           float &PutOptionPrice,
                                           float spotPrice,
                                           float strike,
                                           float rate,
                                           float volatility,
                                           float time) noexcept {
    const float sqrtTime = std::sqrt(time);
    const float logTerm = fast_log_scalar(spotPrice / strike);

    const float vol2 = volatility * volatility;
    const float den = volatility * sqrtTime;

    const float d1 =
        ((rate + 0.5f * vol2) * time + logTerm) / den;

    const float d2 = d1 - den;

    const float N1 = fast_CNDF_scalar(d1);
    const float N2 = fast_CNDF_scalar(d2);

    const float futureValue =
        strike * fast_exp5_scalar(-rate * time);

    CallOptionPrice =
        spotPrice * N1 - futureValue * N2;

    PutOptionPrice =
        futureValue * (1.0f - N2) - spotPrice * (1.0f - N1);
}


using v4f = float __attribute__((vector_size(16)));
using v4i = int __attribute__((vector_size(16)));
using v4u = unsigned int __attribute__((vector_size(16)));

static inline v4f splat_f(float x) noexcept {
    return v4f{x, x, x, x};
}

static inline v4i splat_i(int x) noexcept {
    return v4i{x, x, x, x};
}

static inline v4u splat_u(std::uint32_t x) noexcept {
    return v4u{x, x, x, x};
}

static inline v4u as_u(v4f x) noexcept {
    return std::bit_cast<v4u>(x);
}

static inline v4u as_u(v4i x) noexcept {
    return std::bit_cast<v4u>(x);
}

static inline v4f as_f(v4u x) noexcept {
    return std::bit_cast<v4f>(x);
}

static inline v4f select_v(v4i mask, v4f a, v4f b) noexcept {
    const v4u m = as_u(mask);
    return as_f((as_u(a) & m) | (as_u(b) & ~m));
}

static inline v4f abs_v(v4f x) noexcept {
    return as_f(as_u(x) & splat_u(0x7fffffffu));
}

static inline v4f fast_sqrt_v(v4f x) noexcept {
    const v4f half = x * splat_f(0.5f);
    v4u bits = splat_u(0x5f3759dfu) - (as_u(x) >> 1);
    v4f y = as_f(bits);
    y = y * (splat_f(1.5f) - half * y * y);
    y = y * (splat_f(1.5f) - half * y * y);
    return x * y;
}

static inline v4f fast_exp5_v(v4f x) noexcept {
    x = select_v(x < splat_f(-87.0f), splat_f(-87.0f), x);
    x = select_v(x > splat_f(88.0f), splat_f(88.0f), x);

    const v4f inv_ln2 = splat_f(1.4426950408889634f);
    const v4f ln2 = splat_f(0.6931471805599453f);

    const v4i ki = __builtin_convertvector(x * inv_ln2, v4i);
    const v4f kf = __builtin_convertvector(ki, v4f);
    const v4f r = x - kf * ln2;

    v4f p = splat_f(0.001388888888888889f);
    p = splat_f(0.008333333333333333f) + r * p;
    p = splat_f(0.041666666666666664f) + r * p;
    p = splat_f(0.16666666666666666f) + r * p;
    p = splat_f(0.5f) + r * p;
    p = splat_f(1.0f) + r * p;
    p = splat_f(1.0f) + r * p;

    const v4u exp_bits = as_u(ki + splat_i(127)) << 23;
    return p * as_f(exp_bits);
}

static inline v4f fast_log_v(v4f x) noexcept {
    const v4u xi = as_u(x);
    v4i exp_i = std::bit_cast<v4i>(((xi >> 23) & splat_u(0xffu))) - splat_i(127);
    const v4u mant_i = (xi & splat_u(0x007fffffu)) | splat_u(0x3f800000u);

    v4f m = as_f(mant_i);
    v4f e = __builtin_convertvector(exp_i, v4f);

    const v4i mask = m > splat_f(1.4142135623730951f);
    m = select_v(mask, m * splat_f(0.5f), m);
    e += __builtin_convertvector(mask & splat_i(1), v4f);

    const v4f z = (m - splat_f(1.0f)) / (m + splat_f(1.0f));
    const v4f z2 = z * z;

    v4f p = splat_f(0.1111111111111111f);
    p = splat_f(0.14285714285714285f) + z2 * p;
    p = splat_f(0.2f) + z2 * p;
    p = splat_f(0.3333333333333333f) + z2 * p;
    p = splat_f(1.0f) + z2 * p;

    return e * splat_f(0.6931471805599453f) + splat_f(2.0f) * z * p;
}

static inline v4f fast_CNDF_v(v4f x) noexcept {
    const v4f zero = splat_f(0.0f);
    const v4f one = splat_f(1.0f);

    const v4i neg_mask = x < zero;
    x = abs_v(x);

    const v4f xNPrimeofX =
        fast_exp5_v(splat_f(-0.5f) * x * x) * splat_f(inv_sqrt_2xPI);

    const v4f k = one / (one + splat_f(p_val) * x);

    v4f poly = splat_f(coefficient_a5);
    poly = poly * k + splat_f(coefficient_a4);
    poly = poly * k + splat_f(coefficient_a3);
    poly = poly * k + splat_f(coefficient_a2);
    poly = poly * k + splat_f(coefficient_a1);
    poly = poly * k;

    const v4f local = one - poly * xNPrimeofX;
    v4f result = select_v(neg_mask, one - local, local);

    const v4i big_mask = x > splat_f(8.0f);
    const v4f saturated = select_v(neg_mask, zero, one);
    result = select_v(big_mask, saturated, result);

    return result;
}


}  // namespace

void stu_BlkSchls(std::vector<float> &CallOptionPrice,
                  std::vector<float> &PutOptionPrice,
                  const std::vector<float> &spotPrice,
                  const std::vector<float> &strike,
                  const std::vector<float> &rate,
                  const std::vector<float> &volatility,
                  const std::vector<float> &time) {
    const std::size_t n = spotPrice.size();

    std::size_t i = 0;

    const float* __restrict__ S = spotPrice.data();
    const float* __restrict__ K = strike.data();
    const float* __restrict__ R = rate.data();
    const float* __restrict__ V = volatility.data();
    const float* __restrict__ T = time.data();

    float* __restrict__ C = CallOptionPrice.data();
    float* __restrict__ P = PutOptionPrice.data();

    for (; i + 3 < n; i += 4) {
        const v4f s{S[i + 0], S[i + 1], S[i + 2], S[i + 3]};
        const v4f k{K[i + 0], K[i + 1], K[i + 2], K[i + 3]};
        const v4f r{R[i + 0], R[i + 1], R[i + 2], R[i + 3]};
        const v4f v{V[i + 0], V[i + 1], V[i + 2], V[i + 3]};
        const v4f t{T[i + 0], T[i + 1], T[i + 2], T[i + 3]};

        const v4f sqrt_t = fast_sqrt_v(t);
        const v4f log_term = fast_log_v(s / k);
        const v4f v2 = v * v;
        const v4f den = v * sqrt_t;
        const v4f d1 = ((r + splat_f(0.5f) * v2) * t + log_term) / den;
        const v4f d2 = d1 - den;

        const v4f n1 = fast_CNDF_v(d1);
        const v4f n2 = fast_CNDF_v(d2);
        const v4f future = k * fast_exp5_v((splat_f(0.0f) - r) * t);

        const v4f call = s * n1 - future * n2;
        const v4f put = future * (splat_f(1.0f) - n2) -
                        s * (splat_f(1.0f) - n1);

        C[i + 0] = call[0];
        C[i + 1] = call[1];
        C[i + 2] = call[2];
        C[i + 3] = call[3];
        P[i + 0] = put[0];
        P[i + 1] = put[1];
        P[i + 2] = put[2];
        P[i + 3] = put[3];
    }

    for (; i < n; ++i) {
        stu_BlkSchls_one_scalar(CallOptionPrice[i],
                                PutOptionPrice[i],
                                spotPrice[i],
                                strike[i],
                                rate[i],
                                volatility[i],
                                time[i]);
    }
}

void naive_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);

    naive_BlkSchls(args.call_option_price,
                   args.put_option_price,
                   args.spot_price,
                   args.strike,
                   args.rate,
                   args.volatility,
                   args.time);
}

void stu_BlkSchls_wrapper(void *ctx) {
    auto &args = *static_cast<blackscholes_args *>(ctx);

    stu_BlkSchls(args.call_option_price,
                 args.put_option_price,
                 args.spot_price,
                 args.strike,
                 args.rate,
                 args.volatility,
                 args.time);
}

bool BlkSchls_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<blackscholes_args *>(stu_ctx);
    auto &ref_args = *static_cast<blackscholes_args *>(ref_ctx);

    const double eps = ref_args.epsilon;

    if (ref_args.call_option_price.size() !=
            stu_args.call_option_price.size() ||
        ref_args.put_option_price.size() !=
            stu_args.put_option_price.size()) {
        return false;
    }

    const double atol = 1e-5;
    const size_t n = ref_args.call_option_price.size();

    double max_rel = 0.0;
    double max_abs = 0.0;
    size_t max_idx = 0;
    const char *max_leg = "call";

    for (size_t i = 0; i < n; ++i) {
        const double rc =
            static_cast<double>(ref_args.call_option_price[i]);
        const double rp =
            static_cast<double>(ref_args.put_option_price[i]);

        const double sc =
            static_cast<double>(stu_args.call_option_price[i]);
        const double sp =
            static_cast<double>(stu_args.put_option_price[i]);

        const double err_c = std::abs(rc - sc);
        const double err_p = std::abs(rp - sp);

        const double rel_c =
            (err_c - atol) / std::abs(rc);
        const double rel_p =
            (err_p - atol) / std::abs(rp);

        const bool call_ok =
            err_c <= (atol + eps * std::abs(rc));
        const bool put_ok =
            err_p <= (atol + eps * std::abs(rp));

        if (rel_c > max_rel) {
            max_abs = err_c;
            max_rel = rel_c;
            max_idx = i;
            max_leg = "call";
        }

        if (rel_p > max_rel) {
            max_abs = err_p;
            max_rel = rel_p;
            max_idx = i;
            max_leg = "put";
        }

        if (!call_ok || !put_ok) {
            debug_log(
                "\tDEBUG: fail idx={} | call ref={} stu={} err={} thr={} | put ref={} stu={} err={} thr={}\n",
                i,
                rc,
                sc,
                err_c,
                (atol + eps * std::abs(rc)),
                rp,
                sp,
                err_p,
                (atol + eps * std::abs(rp))
            );

            return false;
        }
    }

    debug_log(
        "\tBlkSchls_check passed: n={}, max_rel_err={}, max_abs_err={} at idx={} ({})\n",
        n,
        max_rel,
        max_abs,
        max_idx,
        max_leg
    );

    return true;
}