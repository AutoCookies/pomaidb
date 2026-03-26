#include "pomai/database.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace std::chrono;

int main(int argc, char** argv) {
    uint32_t num_vectors = 10000;
    uint32_t num_vertices = 10000;
    uint32_t num_edges = 50000;
    uint32_t dim = 128;
    uint32_t k_hops = 2;

    auto parse_pos_u32 = [](const char* s, uint32_t fallback) -> uint32_t {
        if (!s || !*s) return fallback;
        char* end = nullptr;
        unsigned long v = std::strtoul(s, &end, 10);
        if (end == s || *end != '\0') return fallback;
        return static_cast<uint32_t>(v);
    };
    if (argc > 1) num_vectors = parse_pos_u32(argv[1], num_vectors);
    if (argc > 2) num_vertices = parse_pos_u32(argv[2], num_vertices);
    if (argc > 3) num_edges = parse_pos_u32(argv[3], num_edges);

    printf("=============================================================\n");
    printf("                 PomaiDB GraphRAG Benchmark\n");
    printf("=============================================================\n");
    printf("Vectors:    %u\n", num_vectors);
    printf("Vertices:   %u\n", num_vertices);
    printf("Edges:      %u\n", num_edges);
    printf("Dims:       %u\n", dim);
    printf("KHops:      %u\n", k_hops);
    printf("=============================================================\n\n");

    std::filesystem::remove_all("/tmp/graph_bench");
    
    pomai::EmbeddedOptions opts;
    opts.path = "/tmp/graph_bench";
    opts.dim = dim;
    opts.fsync = pomai::FsyncPolicy::kNever;

    pomai::Database db;
    auto st = db.Open(opts);
    if (!st.ok()) {
        fprintf(stderr, "Open failed: %s\n", st.message());
        return 1;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> v_dist(0.0f, 1.0f);

    // 1. Vector Ingestion
    printf("[1/4] Ingesting %u vectors...\n", num_vectors);
    auto t0 = high_resolution_clock::now();
    for (uint32_t i = 0; i < num_vectors; ++i) {
        std::vector<float> vec(dim);
        for (auto& v : vec) v = v_dist(rng);
        db.AddVector(i, vec);
    }
    auto t1 = high_resolution_clock::now();
    double vec_sec = duration<double>(t1 - t0).count();
    printf("  Vector throughput: %.1f ops/sec\n", num_vectors / vec_sec);

    // 2. Vertex Ingestion
    printf("[2/4] Ingesting %u vertices...\n", num_vertices);
    auto t2 = high_resolution_clock::now();
    for (uint32_t i = 0; i < num_vertices; ++i) {
        db.AddVertex(i, 1, pomai::Metadata());
    }
    auto t3 = high_resolution_clock::now();
    double vtx_sec = duration<double>(t3 - t2).count();
    printf("  Vertex throughput: %.1f ops/sec\n", num_vertices / vtx_sec);

    // 3. Edge Ingestion
    printf("[3/4] Ingesting %u random edges...\n", num_edges);
    std::uniform_int_distribution<uint64_t> id_dist(0, num_vertices - 1);
    auto t4 = high_resolution_clock::now();
    for (uint32_t i = 0; i < num_edges; ++i) {
        db.AddEdge(id_dist(rng), id_dist(rng), 1, 0, pomai::Metadata());
    }
    auto t5 = high_resolution_clock::now();
    double edge_sec = duration<double>(t5 - t4).count();
    printf("  Edge throughput:   %.1f ops/sec\n", num_edges / edge_sec);

    // 4. GraphRAG Search
    printf("[4/4] Benchmarking SearchGraphRAG (100 queries)...\n");
    std::vector<double> latencies;
    for (int i = 0; i < 100; ++i) {
        std::vector<float> query(dim);
        for (auto& v : query) v = v_dist(rng);
        
        std::vector<pomai::SearchResult> results;
        auto q0 = high_resolution_clock::now();
        db.SearchGraphRAG(query, 10, pomai::SearchOptions(), k_hops, &results);
        auto q1 = high_resolution_clock::now();
        latencies.push_back(duration<double, std::milli>(q1 - q0).count());
    }

    std::sort(latencies.begin(), latencies.end());
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();

    printf("\n=============================================================\n");
    printf("                      LATENCY RESULTS\n");
    printf("=============================================================\n");
    printf("Avg latency:   %.3f ms\n", avg);
    printf("P50 latency:   %.3f ms\n", latencies[latencies.size() / 2]);
    printf("P99 latency:   %.3f ms\n", latencies[static_cast<size_t>(latencies.size() * 0.99)]);
    printf("=============================================================\n");

    db.Close();
    std::filesystem::remove_all("/tmp/graph_bench");
    return 0;
}
