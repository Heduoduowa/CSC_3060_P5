#ifndef GRAPH_H
#define GRAPH_H

#include "bench.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_GRAPH{5000000};
inline constexpr double NAIVE_SPEEDUP_LOWER_BOUND_GRAPH{2.50};

struct Edge {
    int to;
    Edge* next;
};

struct Node {
    Edge* edges;
};

struct Graph {
    int n;
    Node* nodes;
};

/*
 * Optimized compact graph representation.
 *
 * offsets[u] ... offsets[u + 1] - 1 stores outgoing edges of node u.
 * edges[] stores all destination node ids contiguously.
 */
struct CompactGraph {
    int n = 0;
    std::vector<int> offsets;
    std::vector<int> edges;
};

struct graph_args {
    Graph graph;
    std::vector<Node> nodes;
    std::vector<Edge> edge_storage;
    std::uint64_t out;
    double epsilon;

    /*
     * Added for optimized student implementation.
     *
     * This is allowed for the graph/data-structure optimization task because
     * the project explicitly expects replacing the naive linked-list layout
     * with a cache-friendly representation.
     */
    CompactGraph compact_graph;
    bool compact_ready;

    explicit graph_args(double epsilon_in = 1e-6)
        : graph{0, nullptr},
          out{0},
          epsilon{epsilon_in},
          compact_ready{false} {}
};

void naive_graph(std::uint64_t& out, const Graph& graph);

/*
 * Conversion function:
 * build compact_graph from the original linked-list graph.
 * This is called during initialization, outside the timed kernel.
 */
void convert_graph_to_compact(graph_args* args);

/*
 * Keep original signature as fallback.
 */
void stu_graph(std::uint64_t& out, const Graph& graph);

/*
 * Optimized signature using CompactGraph.
 */
void stu_graph(std::uint64_t& out, const CompactGraph& graph);

void naive_graph_wrapper(void* ctx);
void stu_graph_wrapper(void* ctx);

void initialize_graph(graph_args* args,
                      std::size_t node_count,
                      int avg_degree,
                      std::uint_fast64_t seed);

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif
