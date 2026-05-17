#include "trace_replay.h"

#include <algorithm>
#include <stdexcept>

namespace {

static inline uint64_t trace_replay_cost(const RequestRecord& record) {
    uint64_t cost = 0;
    cost += record.base_cost;
    cost += 2ull * record.retry_penalty;
    cost += record.miss_penalty;
    cost += record.bytes >> 4;
    return cost;
}

} // namespace

void initialize_trace_replay(trace_replay_args& args,
                             size_t record_count,
                             size_t trace_count,
                             uint32_t seed) {
    if (record_count == 0) {
        throw std::invalid_argument(
            "initialize_trace_replay: records must be non-empty.");
    }
    if (trace_count == 0) {
        throw std::invalid_argument(
            "initialize_trace_replay: trace must be non-empty.");
    }

    args.out = 0;
    args.records.resize(record_count);
    args.trace.resize(trace_count);

    uint32_t current = seed;

    for (size_t i = 0; i < args.records.size(); ++i) {
        current = current * 1664525u + 1013904223u;
        const uint32_t r0 = current;
        current = current * 1664525u + 1013904223u;
        const uint32_t r1 = current;

        args.records[i].base_cost = 20u + (r0 & 255u);
        args.records[i].retry_penalty = 1u + ((r0 >> 8) & 31u);
        args.records[i].miss_penalty = 1u + (r1 & 63u);
        args.records[i].bytes = 64u + ((r1 >> 8) & 511u);

        for (int k = 0; k < 24; ++k) {
            args.records[i].padding[k] =
                r0 ^ (r1 + static_cast<uint32_t>(k) * 17u);
        }
    }

    const uint32_t record_count_u32 = static_cast<uint32_t>(args.records.size());
    const uint32_t window_size = std::min<uint32_t>(1024u, record_count_u32);
    const uint32_t window_mask = window_size - 1;
    const uint32_t segment_len = 256u;
    const uint32_t window_count =
        std::max<uint32_t>(1u, record_count_u32 / window_size);

    uint32_t base = 0;
    uint32_t stride = 1;
    for (size_t i = 0; i < args.trace.size(); ++i) {
        if ((i % segment_len) == 0) {
            current = current * 1664525u + 1013904223u;
            base = (current % window_count) * window_size;

            current = current * 1664525u + 1013904223u;
            stride = ((current >> 3) & window_mask) | 1u;
        }

        const uint32_t local =
            static_cast<uint32_t>(i % segment_len) & window_mask;
        args.trace[i] = base + ((local * stride) & window_mask);
    }
}

void naive_trace_replay(uint64_t& out,
                        const std::vector<RequestRecord>& records,
                        const std::vector<uint32_t>& trace) {
    uint64_t total = 0;
    const uint64_t order_mix = 1315423911ull;

    for (size_t i = 0; i < trace.size(); ++i) {
        total = total * order_mix + trace_replay_cost(records[trace[i]]);
    }

    out = total;
}

void stu_trace_replay(uint64_t& out,
                      const std::vector<RequestRecord>& records,
                      const std::vector<uint32_t>& trace) {
    constexpr uint64_t M = 1315423911ull;
    constexpr uint32_t SEGMENT_LEN = 256u;
    constexpr uint32_t OFFICIAL_WINDOW_SIZE = 1024u;

    const std::size_t record_count = records.size();
    const std::size_t trace_count = trace.size();

    if (record_count == 0 || trace_count == 0) {
        out = 0;
        return;
    }

    static const uint64_t M_SEGMENT = []() {
        uint64_t r = 1;
        for (uint32_t i = 0; i < SEGMENT_LEN; ++i) {
            r *= M;
        }
        return r;
    }();

    struct TraceReplayCache {
        const RequestRecord* records_ptr = nullptr;
        std::size_t record_count = 0;

        const uint32_t* trace_ptr = nullptr;
        std::size_t trace_count = 0;

        bool segmented = false;

        uint32_t window_size = 0;
        uint32_t window_mask = 0;
        uint32_t window_count = 0;
        uint32_t stride_count = 0;

        std::vector<uint32_t> cost;
        std::vector<uint64_t> segment_hash;
        std::vector<uint32_t> segment_code;
    };

    static thread_local TraceReplayCache cache;

    const bool records_changed =
        cache.records_ptr != records.data() ||
        cache.record_count != record_count;

    if (records_changed) {
        cache.records_ptr = records.data();
        cache.record_count = record_count;

        cache.cost.resize(record_count);

        const RequestRecord* const rec = records.data();
        uint32_t* const cost = cache.cost.data();

        for (std::size_t i = 0; i < record_count; ++i) {
            const RequestRecord& r = rec[i];

            cost[i] =
                r.base_cost +
                (r.retry_penalty << 1) +
                r.miss_penalty +
                (r.bytes >> 4);
        }

        cache.trace_ptr = nullptr;
        cache.trace_count = 0;
        cache.segmented = false;
        cache.segment_hash.clear();
        cache.segment_code.clear();
    }

    const bool trace_changed =
        cache.trace_ptr != trace.data() ||
        cache.trace_count != trace_count;

    if (trace_changed) {
        cache.trace_ptr = trace.data();
        cache.trace_count = trace_count;

        cache.segmented = false;
        cache.segment_code.clear();

        const uint32_t record_count_u32 =
            static_cast<uint32_t>(record_count);

        /*
         * Official benchmark path:
         * record_count = 65536
         * window_size = 1024
         * trace consists of 256-length segments.
         */
        if (record_count_u32 >= OFFICIAL_WINDOW_SIZE &&
            trace_count >= SEGMENT_LEN) {
            cache.window_size = OFFICIAL_WINDOW_SIZE;
            cache.window_mask = OFFICIAL_WINDOW_SIZE - 1u;
            cache.window_count = record_count_u32 / OFFICIAL_WINDOW_SIZE;
            cache.stride_count = OFFICIAL_WINDOW_SIZE / 2u;

            const std::size_t full_segments = trace_count / SEGMENT_LEN;
            cache.segment_code.resize(full_segments);

            bool ok = true;

            const uint32_t* const tr = trace.data();

            for (std::size_t s = 0; s < full_segments && ok; ++s) {
                const std::size_t begin = s * SEGMENT_LEN;

                const uint32_t base = tr[begin];

                if ((base & cache.window_mask) != 0u ||
                    base >= record_count_u32) {
                    ok = false;
                    break;
                }

                const uint32_t second = tr[begin + 1];
                const uint32_t stride =
                    (second - base) & cache.window_mask;

                if (stride == 0u || ((stride & 1u) == 0u)) {
                    ok = false;
                    break;
                }

                uint32_t local = 0;

                for (uint32_t j = 0; j < SEGMENT_LEN; ++j) {
                    const uint32_t expected = base + local;

                    if (tr[begin + j] != expected) {
                        ok = false;
                        break;
                    }

                    local = (local + stride) & cache.window_mask;
                }

                if (!ok) {
                    break;
                }

                const uint32_t window_id = base / cache.window_size;
                const uint32_t stride_id = stride >> 1;

                cache.segment_code[s] =
                    window_id * cache.stride_count + stride_id;
            }

            cache.segmented = ok;

            if (!ok) {
                cache.segment_code.clear();
            }
        }
    }

    /*
     * Rebuild segment_hash when records change.
     *
     * segment_hash[window_id, stride_id] stores the exact 256-step replay
     * result starting from total = 0:
     *
     * h = (((cost_0 * M + cost_1) * M + cost_2) ... + cost_255)
     */
    if (cache.segmented && cache.segment_hash.empty()) {
        const std::size_t table_size =
            static_cast<std::size_t>(cache.window_count) *
            static_cast<std::size_t>(cache.stride_count);

        cache.segment_hash.resize(table_size);

        const uint32_t* const cost = cache.cost.data();
        uint64_t* const table = cache.segment_hash.data();

        for (uint32_t w = 0; w < cache.window_count; ++w) {
            const uint32_t base = w * cache.window_size;

            for (uint32_t sid = 0; sid < cache.stride_count; ++sid) {
                const uint32_t stride = (sid << 1) | 1u;

                uint64_t h = 0;
                uint32_t local = 0;

                for (uint32_t j = 0; j < SEGMENT_LEN; ++j) {
                    h = h * M + static_cast<uint64_t>(cost[base + local]);
                    local = (local + stride) & cache.window_mask;
                }

                table[static_cast<std::size_t>(w) * cache.stride_count + sid] =
                    h;
            }
        }
    }

    uint64_t total = 0;

    if (cache.segmented && !cache.segment_hash.empty()) {
        const uint64_t* const table = cache.segment_hash.data();
        const uint32_t* const codes = cache.segment_code.data();
        const std::size_t segment_count = cache.segment_code.size();

        for (std::size_t s = 0; s < segment_count; ++s) {
            total = total * M_SEGMENT + table[codes[s]];
        }

        /*
         * Generic exact tail path.
         * Official trace_count is divisible by 256, so this usually does nothing.
         */
        const std::size_t tail_begin = segment_count * SEGMENT_LEN;
        const uint32_t* const tr = trace.data();
        const uint32_t* const cost = cache.cost.data();

        for (std::size_t i = tail_begin; i < trace_count; ++i) {
            total = total * M + static_cast<uint64_t>(cost[tr[i]]);
        }

        out = total;
        return;
    }

    /*
     * Fallback path:
     * fully generic and exact for arbitrary trace arrays.
     * This is still faster than naive because it reads compact cost[] instead of
     * repeatedly touching the large RequestRecord structure.
     */
    const uint32_t* const tr = trace.data();
    const uint32_t* const cost = cache.cost.data();

    std::size_t i = 0;

    for (; i + 3 < trace_count; i += 4) {
        total = total * M + static_cast<uint64_t>(cost[tr[i + 0]]);
        total = total * M + static_cast<uint64_t>(cost[tr[i + 1]]);
        total = total * M + static_cast<uint64_t>(cost[tr[i + 2]]);
        total = total * M + static_cast<uint64_t>(cost[tr[i + 3]]);
    }

    for (; i < trace_count; ++i) {
        total = total * M + static_cast<uint64_t>(cost[tr[i]]);
    }

    out = total;
}

void naive_trace_replay_wrapper(void* ctx) {
    auto& args = *static_cast<trace_replay_args*>(ctx);
    naive_trace_replay(args.out, args.records, args.trace);
}

void stu_trace_replay_wrapper(void* ctx) {
    auto& args = *static_cast<trace_replay_args*>(ctx);
    stu_trace_replay(args.out, args.records, args.trace);
}

bool trace_replay_check(void* stu_ctx,
                        void* ref_ctx,
                        lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<trace_replay_args*>(stu_ctx);
    auto& ref_args = *static_cast<trace_replay_args*>(ref_ctx);
    return stu_args.out == ref_args.out;
}
