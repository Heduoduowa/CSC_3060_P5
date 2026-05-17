#include "graph.h"

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

void convert_graph_to_compact(graph_args* args) {
    if (!args) {
        return;
    }

    CompactGraph& compact = args->compact_graph;

    compact.n = args->graph.n;
    compact.offsets.assign(static_cast<std::size_t>(compact.n) + 1, 0);
    compact.edges.clear();
    compact.edges.reserve(args->edge_storage.size());

    for (int u = 0; u < compact.n; ++u) {
        compact.offsets[static_cast<std::size_t>(u)] =
            static_cast<int>(compact.edges.size());

        const Edge* e = args->graph.nodes[u].edges;
        while (e) {
            compact.edges.push_back(e->to);
            e = e->next;
        }
    }

    compact.offsets[static_cast<std::size_t>(compact.n)] =
        static_cast<int>(compact.edges.size());

    args->compact_ready = true;
}

void initialize_graph(graph_args* args,
                      std::size_t node_count,
                      int avg_degree,
                      std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(
        0,
        static_cast<int>(node_count) - 1
    );

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    std::size_t edge_pos = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        std::vector<int> neighbors;
        neighbors.reserve(avg_degree);

        for (int k = 0; k < avg_degree; ++k) {
            neighbors.push_back(dist(gen));
        }

        Edge* head = nullptr;
        for (int k = avg_degree - 1; k >= 0; --k) {
            Edge& e = args->edge_storage[edge_pos + static_cast<std::size_t>(k)];
            e.to = neighbors[static_cast<std::size_t>(k)];
            e.next = head;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
    }

    args->out = 0;
    convert_graph_to_compact(args);
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;

    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }

    out = checksum;
}

namespace {

std::uint64_t sum_edges_scalar(const int* edges, std::size_t n) noexcept {
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
    std::uint64_t s2 = 0;
    std::uint64_t s3 = 0;

    std::size_t i = 0;
    for (; i + 7 < n; i += 8) {
        s0 += static_cast<std::uint64_t>(edges[i + 0]);
        s1 += static_cast<std::uint64_t>(edges[i + 1]);
        s2 += static_cast<std::uint64_t>(edges[i + 2]);
        s3 += static_cast<std::uint64_t>(edges[i + 3]);
        s0 += static_cast<std::uint64_t>(edges[i + 4]);
        s1 += static_cast<std::uint64_t>(edges[i + 5]);
        s2 += static_cast<std::uint64_t>(edges[i + 6]);
        s3 += static_cast<std::uint64_t>(edges[i + 7]);
    }

    std::uint64_t sum = s0 + s1 + s2 + s3;
    for (; i < n; ++i) {
        sum += static_cast<std::uint64_t>(edges[i]);
    }

    return sum;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
std::uint64_t sum_edges_avx2(const int* edges, std::size_t n) noexcept {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    std::size_t i = 0;
    for (; i + 31 < n; i += 32) {
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(edges + i + 0));
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(edges + i + 8));
        const __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(edges + i + 16));
        const __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(edges + i + 24));

        acc0 = _mm256_add_epi64(acc0, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v0)));
        acc1 = _mm256_add_epi64(acc1, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(v0, 1)));
        acc2 = _mm256_add_epi64(acc2, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v1)));
        acc3 = _mm256_add_epi64(acc3, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(v1, 1)));

        acc0 = _mm256_add_epi64(acc0, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v2)));
        acc1 = _mm256_add_epi64(acc1, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(v2, 1)));
        acc2 = _mm256_add_epi64(acc2, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v3)));
        acc3 = _mm256_add_epi64(acc3, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(v3, 1)));
    }

    alignas(32) std::uint64_t lanes[4];
    const __m256i total01 = _mm256_add_epi64(acc0, acc1);
    const __m256i total23 = _mm256_add_epi64(acc2, acc3);
    const __m256i total = _mm256_add_epi64(total01, total23);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), total);

    std::uint64_t sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < n; ++i) {
        sum += static_cast<std::uint64_t>(edges[i]);
    }

    return sum;
}
#endif

std::uint64_t sum_edges_chunk(const int* edges, std::size_t n) noexcept {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx2")) {
        return sum_edges_avx2(edges, n);
    }
#endif
    return sum_edges_scalar(edges, n);
}

struct alignas(64) PartialSum {
    std::uint64_t value = 0;
};

class graph_worker_pool {
public:
    explicit graph_worker_pool(std::size_t worker_count)
        : partials_(worker_count + 1) {
        workers_.reserve(worker_count);
        for (std::size_t worker = 1; worker <= worker_count; ++worker) {
            workers_.emplace_back([this, worker] { worker_loop(worker); });
        }
    }

    ~graph_worker_pool() {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        start_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    std::uint64_t run(const int* edges, std::size_t n) {
        if (workers_.empty() || n < kParallelThreshold) {
            return sum_edges_chunk(edges, n);
        }

        {
            std::lock_guard lock(mutex_);
            edges_ = edges;
            n_ = n;
            completed_ = 0;
            ++generation_;
        }

        start_cv_.notify_all();
        run_partition(0);

        std::unique_lock lock(mutex_);
        done_cv_.wait(lock, [this] { return completed_ == workers_.size(); });
        lock.unlock();

        std::uint64_t total = 0;
        for (std::size_t i = 0; i < partials_.size(); ++i) {
            total += partials_[i].value;
        }
        return total;
    }

private:
    static constexpr std::size_t kParallelThreshold = 512 * 1024;

    std::size_t boundary(std::size_t partition) const noexcept {
        const std::size_t total = workers_.size() + 1;
        if (partition == 0) {
            return 0;
        }
        if (partition == total) {
            return n_;
        }
        return ((n_ * partition) / total) & ~std::size_t{31};
    }

    void run_partition(std::size_t partition) noexcept {
        const std::size_t begin = boundary(partition);
        const std::size_t end = boundary(partition + 1);
        partials_[partition].value = (begin < end)
            ? sum_edges_chunk(edges_ + begin, end - begin)
            : 0;
    }

    void worker_loop(std::size_t partition) {
        std::uint64_t seen_generation = 0;
        while (true) {
            std::unique_lock lock(mutex_);
            start_cv_.wait(lock, [this, seen_generation] {
                return stop_ || generation_ != seen_generation;
            });
            if (stop_) {
                return;
            }

            const std::uint64_t current_generation = generation_;
            lock.unlock();

            run_partition(partition);

            lock.lock();
            seen_generation = current_generation;
            if (++completed_ == workers_.size()) {
                done_cv_.notify_one();
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;
    std::vector<PartialSum> partials_;
    const int* edges_ = nullptr;
    std::size_t n_ = 0;
    std::uint64_t generation_ = 0;
    std::size_t completed_ = 0;
    bool stop_ = false;
};

std::size_t graph_worker_count() {
    const auto hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads <= 1) {
        return 0;
    }
    return std::min<std::size_t>(7, hardware_threads - 1);
}

graph_worker_pool& graph_pool() {
    static graph_worker_pool pool(graph_worker_count());
    return pool;
}

}  // namespace

void stu_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;

    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }

    out = checksum;
}

void stu_graph(std::uint64_t& out, const CompactGraph& graph) {
    out = graph_pool().run(graph.edges.data(), graph.edges.size());
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    if (!args.compact_ready) {
        convert_graph_to_compact(&args);
    }
    stu_graph(args.out, args.compact_graph);
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);

    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);

    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
