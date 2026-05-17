#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

namespace {

constexpr std::uint8_t kVariableMask = 0x99u;
constexpr std::uint8_t kResultXorMask = 0xA5u;

constexpr std::uint8_t simplified_bitwise_byte(std::uint8_t either) noexcept {
    return static_cast<std::uint8_t>((either & kVariableMask) ^
                                     kResultXorMask);
}

void stu_bitwise_scalar(std::int8_t* __restrict__ result,
                        const std::int8_t* __restrict__ a,
                        const std::int8_t* __restrict__ b,
                        std::size_t n) {
    using alias_u64 = std::uint64_t __attribute__((__may_alias__));

    constexpr std::uint64_t variable_word = 0x9999999999999999ULL;
    constexpr std::uint64_t xor_word = 0xA5A5A5A5A5A5A5A5ULL;

    const std::size_t word_count = n / sizeof(alias_u64);
    auto* __restrict__ out_words = reinterpret_cast<alias_u64*>(result);
    const auto* __restrict__ a_words = reinterpret_cast<const alias_u64*>(a);
    const auto* __restrict__ b_words = reinterpret_cast<const alias_u64*>(b);

    std::size_t w = 0;
    for (; w + 3 < word_count; w += 4) {
        const std::uint64_t e0 = a_words[w + 0] | b_words[w + 0];
        const std::uint64_t e1 = a_words[w + 1] | b_words[w + 1];
        const std::uint64_t e2 = a_words[w + 2] | b_words[w + 2];
        const std::uint64_t e3 = a_words[w + 3] | b_words[w + 3];

        out_words[w + 0] = (e0 & variable_word) ^ xor_word;
        out_words[w + 1] = (e1 & variable_word) ^ xor_word;
        out_words[w + 2] = (e2 & variable_word) ^ xor_word;
        out_words[w + 3] = (e3 & variable_word) ^ xor_word;
    }

    for (; w < word_count; ++w) {
        const std::uint64_t either = a_words[w] | b_words[w];
        out_words[w] = (either & variable_word) ^ xor_word;
    }

    for (std::size_t i = word_count * sizeof(alias_u64); i < n; ++i) {
        const auto either = static_cast<std::uint8_t>(
            static_cast<std::uint8_t>(a[i]) | static_cast<std::uint8_t>(b[i]));
        result[i] = static_cast<std::int8_t>(simplified_bitwise_byte(either));
    }
}

} // namespace

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// TODO: Optimize the bitwise function
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    if (n == 0) {
        return;
    }

    auto *const result_data = result.data();
    const auto *const a_data = a.data();
    const auto *const b_data = b.data();

    stu_bitwise_scalar(result_data, a_data, b_data, n);
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    // Call your verion here
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}