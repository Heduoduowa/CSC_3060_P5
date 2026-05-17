#ifndef FILTER_GRADIENT_H
#define FILTER_GRADIENT_H

#include "bench.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

inline constexpr std::chrono::nanoseconds BASELINE_FILTER_GRADIENT{25000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_FILTER_GRADIENT{1.45};

struct data_struct {
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> d;
    std::vector<float> e;
    std::vector<float> f;
    std::vector<float> g;
    std::vector<float> h;
    std::vector<float> i;
};

/*
 * Three-channel packed float.
 *
 * abc: x=a, y=b, z=c
 * def: x=d, y=e, z=f
 * ghi: x=g, y=h, z=i
 */
struct fg3 {
    float x;
    float y;
    float z;
};

/*
 * Three-channel packed double.
 * Used only for small row buffers of abc box-filter sums, because the
 * reference implementation accumulates a/b/c 3x3 sums in double.
 */
struct fg3d {
    double x;
    double y;
    double z;
};

struct optimized_filter_gradient_data {
    std::vector<fg3> abc;
    std::vector<fg3> def;
    std::vector<fg3> ghi;
    bool ready = false;
};

struct filter_gradient_args {
    data_struct data;
    std::size_t width;
    std::size_t height;
    float out;
    double epsilon;

    /*
     * Added optimized representation.
     * Built during initialization, outside the timed benchmark region.
     */
    optimized_filter_gradient_data opt;

    explicit filter_gradient_args(double epsilon_in = 1e-6)
        : width(0), height(0), out(0.0f), epsilon(epsilon_in) {}
};

void naive_filter_gradient(float& out,
                           const data_struct& data,
                           std::size_t width,
                           std::size_t height);

void stu_filter_gradient(float& out,
                         const data_struct& data,
                         std::size_t width,
                         std::size_t height);

void convert_filter_gradient(filter_gradient_args* args);

void stu_filter_gradient_optimized(float& out,
                                   const optimized_filter_gradient_data& opt,
                                   std::size_t width,
                                   std::size_t height);

void naive_filter_gradient_wrapper(void* ctx);
void stu_filter_gradient_wrapper(void* ctx);

void initialize_filter_gradient(filter_gradient_args* args,
                                std::size_t width,
                                std::size_t height,
                                std::uint_fast64_t seed);

bool filter_gradient_check(void* stu_ctx,
                           void* ref_ctx,
                           lab_test_func naive_func);

#endif  // FILTER_GRADIENT_H
