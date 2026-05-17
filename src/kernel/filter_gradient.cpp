#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

void convert_filter_gradient(filter_gradient_args* args) {
    if (!args) {
        return;
    }

    const std::size_t count = args->width * args->height;

    args->opt.abc.resize(count);
    args->opt.def.resize(count);
    args->opt.ghi.resize(count);

    const float* const a = args->data.a.data();
    const float* const b = args->data.b.data();
    const float* const c = args->data.c.data();

    const float* const d = args->data.d.data();
    const float* const e = args->data.e.data();
    const float* const f = args->data.f.data();

    const float* const g = args->data.g.data();
    const float* const h = args->data.h.data();
    const float* const ii = args->data.i.data();

    fg3* const abc = args->opt.abc.data();
    fg3* const def = args->opt.def.data();
    fg3* const ghi = args->opt.ghi.data();

    for (std::size_t k = 0; k < count; ++k) {
        abc[k] = fg3{a[k], b[k], c[k]};
        def[k] = fg3{d[k], e[k], f[k]};
        ghi[k] = fg3{g[k], h[k], ii[k]};
    }

    args->opt.ready = true;
}

void initialize_filter_gradient(filter_gradient_args* args,
                                std::size_t width,
                                std::size_t height,
                                std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count);
    args->data.b.resize(count);
    args->data.c.resize(count);
    args->data.d.resize(count);
    args->data.e.resize(count);
    args->data.f.resize(count);
    args->data.g.resize(count);
    args->data.h.resize(count);
    args->data.i.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen);
        args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen);
        args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen);
        args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen);
        args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);
    }

    /*
     * Conversion is outside the timed benchmark region.
     * This follows the data-structure optimization rule.
     */
    convert_filter_gradient(args);
}

// Reference implementation. Keep semantic behavior unchanged.
void naive_filter_gradient(float& out,
                           const data_struct& data,
                           std::size_t width,
                           std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    double total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {
            double sum_a = 0.0;
            double sum_b = 0.0;
            double sum_c = 0.0;

            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;

                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);

                    sum_a += data.a[idx];
                    sum_b += data.b[idx];
                    sum_c += data.c[idx];
                }
            }

            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;

            const float p1 = avg_a * avg_b + avg_c;

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0 = y * W;
            const std::size_t yp1 = (y + 1) * W;

            const std::size_t xm1 = x - 1;
            const std::size_t x0 = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx =
                -data.d[ym1 + xm1] + data.d[ym1 + xp1]
                - 2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1]
                - data.d[yp1 + xm1] + data.d[yp1 + xp1];

            const float sobel_ex =
                -data.e[ym1 + xm1] + data.e[ym1 + xp1]
                - 2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1]
                - data.e[yp1 + xm1] + data.e[yp1 + xp1];

            const float sobel_fx =
                -data.f[ym1 + xm1] + data.f[ym1 + xp1]
                - 2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1]
                - data.f[yp1 + xm1] + data.f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -data.g[ym1 + xm1]
                - 2.0f * data.g[ym1 + x0]
                - data.g[ym1 + xp1]
                + data.g[yp1 + xm1]
                + 2.0f * data.g[yp1 + x0]
                + data.g[yp1 + xp1];

            const float sobel_hy =
                -data.h[ym1 + xm1]
                - 2.0f * data.h[ym1 + x0]
                - data.h[ym1 + xp1]
                + data.h[yp1 + xm1]
                + 2.0f * data.h[yp1 + x0]
                + data.h[yp1 + xp1];

            const float sobel_iy =
                -data.i[ym1 + xm1]
                - 2.0f * data.i[ym1 + x0]
                - data.i[ym1 + xp1]
                + data.i[yp1 + xm1]
                + 2.0f * data.i[yp1 + x0]
                + data.i[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = static_cast<float>(total);
}

namespace {

static inline fg3d abc_hsum3(const fg3* row, std::size_t x) noexcept {
    const fg3& l = row[x - 1];
    const fg3& m = row[x];
    const fg3& r = row[x + 1];

    return fg3d{
        static_cast<double>(l.x) + static_cast<double>(m.x) + static_cast<double>(r.x),
        static_cast<double>(l.y) + static_cast<double>(m.y) + static_cast<double>(r.y),
        static_cast<double>(l.z) + static_cast<double>(m.z) + static_cast<double>(r.z)
    };
}

static inline fg3 horizontal_weight3(const fg3* row, std::size_t x) noexcept {
    const fg3& l = row[x - 1];
    const fg3& m = row[x];
    const fg3& r = row[x + 1];

    return fg3{
        l.x + 2.0f * m.x + r.x,
        l.y + 2.0f * m.y + r.y,
        l.z + 2.0f * m.z + r.z
    };
}

static inline fg3 vertical_weight3(const fg3* top,
                                   const fg3* mid,
                                   const fg3* bot,
                                   std::size_t x) noexcept {
    return fg3{
        top[x].x + 2.0f * mid[x].x + bot[x].x,
        top[x].y + 2.0f * mid[x].y + bot[x].y,
        top[x].z + 2.0f * mid[x].z + bot[x].z
    };
}

static void compute_abc_hsum_row(const fg3* row,
                                 fg3d* out,
                                 std::size_t W) {
    for (std::size_t x = 1; x + 1 < W; ++x) {
        out[x] = abc_hsum3(row, x);
    }
}

static void compute_ghi_hweight_row(const fg3* row,
                                    fg3* out,
                                    std::size_t W) {
    for (std::size_t x = 1; x + 1 < W; ++x) {
        out[x] = horizontal_weight3(row, x);
    }
}

static void compute_def_vweight_row(const fg3* top,
                                    const fg3* mid,
                                    const fg3* bot,
                                    fg3* out,
                                    std::size_t W) {
    for (std::size_t x = 0; x < W; ++x) {
        out[x] = vertical_weight3(top, mid, bot, x);
    }
}

}  // namespace

void stu_filter_gradient_optimized(float& out,
                                   const optimized_filter_gradient_data& opt,
                                   std::size_t width,
                                   std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;

    if (W < 3 || H < 3) {
        out = 0.0f;
        return;
    }

    constexpr float inv9 = 1.0f / 9.0f;

    const fg3* const abc = opt.abc.data();
    const fg3* const def = opt.def.data();
    const fg3* const ghi = opt.ghi.data();

    /*
     * Row buffers are small: width = 1024.
     * They are reused across benchmark iterations to avoid allocation noise.
     */
    static thread_local std::vector<fg3d> abc_h0;
    static thread_local std::vector<fg3d> abc_h1;
    static thread_local std::vector<fg3d> abc_h2;

    static thread_local std::vector<fg3> ghi_h0;
    static thread_local std::vector<fg3> ghi_h1;
    static thread_local std::vector<fg3> ghi_h2;

    static thread_local std::vector<fg3> def_v;

    abc_h0.resize(W);
    abc_h1.resize(W);
    abc_h2.resize(W);

    ghi_h0.resize(W);
    ghi_h1.resize(W);
    ghi_h2.resize(W);

    def_v.resize(W);

    compute_abc_hsum_row(abc + 0 * W, abc_h0.data(), W);
    compute_abc_hsum_row(abc + 1 * W, abc_h1.data(), W);
    compute_abc_hsum_row(abc + 2 * W, abc_h2.data(), W);

    compute_ghi_hweight_row(ghi + 0 * W, ghi_h0.data(), W);
    compute_ghi_hweight_row(ghi + 1 * W, ghi_h1.data(), W);
    compute_ghi_hweight_row(ghi + 2 * W, ghi_h2.data(), W);

    double total = 0.0;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        if (y > 1) {
            abc_h0.swap(abc_h1);
            abc_h1.swap(abc_h2);

            ghi_h0.swap(ghi_h1);
            ghi_h1.swap(ghi_h2);

            compute_abc_hsum_row(abc + (y + 1) * W, abc_h2.data(), W);
            compute_ghi_hweight_row(ghi + (y + 1) * W, ghi_h2.data(), W);
        }

        const fg3* const def_top = def + (y - 1) * W;
        const fg3* const def_mid = def + y * W;
        const fg3* const def_bot = def + (y + 1) * W;

        compute_def_vweight_row(def_top, def_mid, def_bot, def_v.data(), W);

        const fg3d* const a_top = abc_h0.data();
        const fg3d* const a_mid = abc_h1.data();
        const fg3d* const a_bot = abc_h2.data();

        const fg3* const g_top = ghi_h0.data();
        const fg3* const g_bot = ghi_h2.data();

        const fg3* const dv = def_v.data();

        for (std::size_t x = 1; x + 1 < W; ++x) {
            /*
             * Box filter for a,b,c:
             * horizontal sums are already in double, then vertical sum in double.
             */
            const double sum_a =
                a_top[x].x + a_mid[x].x + a_bot[x].x;
            const double sum_b =
                a_top[x].y + a_mid[x].y + a_bot[x].y;
            const double sum_c =
                a_top[x].z + a_mid[x].z + a_bot[x].z;

            const float avg_a = static_cast<float>(sum_a * inv9);
            const float avg_b = static_cast<float>(sum_b * inv9);
            const float avg_c = static_cast<float>(sum_c * inv9);

            const float p1 = avg_a * avg_b + avg_c;

            /*
             * Sobel-x for d,e,f:
             * vertical weighted column, then right minus left.
             */
            const float sobel_dx = dv[x + 1].x - dv[x - 1].x;
            const float sobel_ex = dv[x + 1].y - dv[x - 1].y;
            const float sobel_fx = dv[x + 1].z - dv[x - 1].z;

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            /*
             * Sobel-y for g,h,i:
             * horizontal weighted row, then bottom minus top.
             */
            const float sobel_gy = g_bot[x].x - g_top[x].x;
            const float sobel_hy = g_bot[x].y - g_top[x].y;
            const float sobel_iy = g_bot[x].z - g_top[x].z;

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            /*
             * Same global accumulation style as reference:
             * p1+p2+p3 is float expression, then added to double total.
             */
            total += p1 + p2 + p3;
        }
    }

    out = static_cast<float>(total);
}

/*
 * Keep original signature for compatibility.
 * The real optimized timed path is stu_filter_gradient_wrapper.
 */
void stu_filter_gradient(float& out,
                         const data_struct& data,
                         std::size_t width,
                         std::size_t height) {
    naive_filter_gradient(out, data, width, height);
}

void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}

void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;

    if (!args.opt.ready) {
        convert_filter_gradient(&args);
    }

    stu_filter_gradient_optimized(args.out, args.opt, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx,
                           void* ref_ctx,
                           lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);

    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;

    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}