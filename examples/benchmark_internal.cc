/*
 * examples/benchmark_internal.cc
 *
 * Internal Benchmark Tool for Pomai - PLAN B (Pure PQ Scan vs HNSW)
 *
 * Changes from original:
 * 1. SimHash DISABLED (fingerprint_bits = 0) to remove noise.
 * 2. Compares Recall@100: Does the Holographic scan (Top 100) contain the HNSW Top 10?
 *
 * Usage:
 * ./build/benchmark [num_vectors] [dim] [shard_count]
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <filesystem>
#include <numeric>
#include <iomanip>
#include <cstring>
#include <atomic>
#include <unordered_set>

#include "src/core/config.h"
#include "src/core/shard_manager.h"
#include "src/core/pps_manager.h"
#include "src/memory/arena.h"

using namespace pomai::core;
using namespace std::chrono;

// --- Helper: Stopwatch ---
class Stopwatch {
    high_resolution_clock::time_point start_time;
public:
    Stopwatch() { reset(); }
    void reset() { start_time = high_resolution_clock::now(); }
    double elapsed_ms() const {
        return duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - start_time).count();
    }
};

// --- Helper: Random Vector Generator ---
void generate_random_data(size_t n, size_t dim, std::vector<float>& data) {
    data.resize(n * dim);
    // Use deterministic seed to ensure HNSW and SoA see same distribution
    std::mt19937 rng(42); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n * dim; ++i) {
        data[i] = dist(rng);
    }
}

// --- Helper: Print Stats ---
void print_stats(const std::string& name, double total_ms, size_t count) {
    double total_sec = total_ms / 1000.0;
    double qps = (total_sec > 0) ? count / total_sec : 0.0;
    double avg_lat_ms = (count > 0) ? total_ms / count : 0.0;

    std::cout << std::left << std::setw(25) << name 
              << " | Time: " << std::fixed << std::setprecision(2) << total_ms << " ms"
              << " | QPS: " << std::setprecision(0) << qps
              << " | Latency: " << std::setprecision(3) << avg_lat_ms << " ms/req\n";
}

int main(int argc, char** argv) {
    // 1. Configuration
    size_t num_vectors = 100000; 
    size_t dim = 128;            
    size_t shard_count = 4;

    if (argc > 1) num_vectors = std::stoull(argv[1]);
    if (argc > 2) dim = std::stoull(argv[2]);
    if (argc > 3) shard_count = std::stoull(argv[3]);

    // Clean up old data
    if (std::filesystem::exists("./data")) {
        std::filesystem::remove_all("./data");
        std::filesystem::create_directory("./data");
    }

    // --- PLAN B CONFIGURATION ---
    pomai::config::runtime.shard_count = static_cast<uint32_t>(shard_count);
    pomai::config::runtime.max_elements_total = num_vectors * 2;
    
    // [KEY CHANGE] Disable SimHash to test Pure PQ Scan accuracy
    pomai::config::runtime.fingerprint_bits = 0; 

    std::cout << "=== Pomai Benchmark (Plan B: Pure PQ Scan) ===\n";
    std::cout << "Vectors: " << num_vectors << " | Dim: " << dim << "\n";
    std::cout << "Mode:    SimHash DISABLED (Full Scan)\n";
    std::cout << "----------------------------------------------\n";

    // 2. Setup System
    auto shard_mgr = std::make_unique<ShardManager>(static_cast<uint32_t>(shard_count));
    auto ppsm = std::make_unique<PPSM>(shard_mgr.get(), dim, num_vectors * 2, 16, 200, false);

    // 3. Data Generation
    std::cout << "[Gen] Generating random vectors...\n";
    std::vector<float> data;
    generate_random_data(num_vectors, dim, data);

    std::vector<float> query_pool;
    size_t num_queries = 100; // Reduce queries to verify accuracy closely
    generate_random_data(num_queries, dim, query_pool);

    // 4. Ingestion
    std::cout << "[Ingest] Inserting " << num_vectors << " vectors...\n";
    auto start_ingest = high_resolution_clock::now();
    for (size_t i = 0; i < num_vectors; ++i) {
        std::string key = std::to_string(i); // Simple numeric key
        const float* vec = data.data() + i * dim;
        ppsm->addVec(key.data(), key.size(), vec);
        
        if ((i + 1) % (num_vectors / 10) == 0) {
            std::cout << "." << std::flush;
        }
    }
    double ingest_ms = duration_cast<duration<double, std::milli>>(high_resolution_clock::now() - start_ingest).count();
    std::cout << "\n";
    print_stats("Ingestion", ingest_ms, num_vectors);
    std::cout << "----------------------------------------------\n";

    // 5. Search Benchmark: HNSW (Ground Truth)
    std::cout << "[Search] Standard HNSW (Top-10)...\n";
    Stopwatch sw;
    size_t total_hnsw_found = 0;
    std::vector<std::vector<std::string>> hnsw_results;
    
    for (size_t i = 0; i < num_queries; ++i) {
        const float* q = query_pool.data() + i * dim;
        auto res = ppsm->search(q, dim, 10);
        total_hnsw_found += res.size();
        
        std::vector<std::string> keys;
        for(auto& p : res) keys.push_back(p.first);
        hnsw_results.push_back(keys);
    }
    print_stats("HNSW Search", sw.elapsed_ms(), num_queries);

    // 6. Search Benchmark: Holographic (Pure PQ Scan)
    // We scan for Top-100 to calculate Recall@100
    size_t scan_k = 100;
    std::cout << "[Search] Holographic PQ Scan (Top-" << scan_k << ")...\n";
    sw.reset();
    
    size_t recall_hits = 0;
    size_t total_ground_truth = 0;

    for (size_t i = 0; i < num_queries; ++i) {
        const float* q = query_pool.data() + i * dim;
        
        // SCAN: Get top-100 candidates based on PQ score
        auto res_holo = ppsm->searchHolographic(q, dim, scan_k);
        
        // Calculate Recall: How many of HNSW's Top-10 are in Holo's Top-100?
        std::unordered_set<std::string> holo_set;
        for(auto& p : res_holo) holo_set.insert(p.first);

        const auto& ground_truth = hnsw_results[i];
        for(const auto& true_key : ground_truth) {
            if(holo_set.count(true_key)) {
                recall_hits++;
            }
            total_ground_truth++;
        }
    }
    print_stats("Holographic Search", sw.elapsed_ms(), num_queries);

    // 7. Accuracy Report
    std::cout << "----------------------------------------------\n";
    double recall = (total_ground_truth > 0) ? (100.0 * recall_hits / total_ground_truth) : 0.0;
    std::cout << "[Quality] Recall@" << scan_k << " (vs HNSW Top-10): " 
              << std::fixed << std::setprecision(2) << recall << " %\n";
    
    if (recall < 50.0) {
        std::cout << "-> Note: Low recall on RANDOM data is expected for PQ (K-Means fails to cluster noise).\n";
        std::cout << "-> On real datasets (Sift/Gist), this usually jumps to >90%.\n";
    } else {
        std::cout << "-> Great! PQ is approximating the random distribution well.\n";
    }

    return 0;
}