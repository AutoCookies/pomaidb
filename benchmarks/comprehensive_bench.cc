// PomaiDB Comprehensive Benchmark Suite
// Industry-standard metrics: Latency (p50/p90/p99), Throughput, Recall@k
//
// Usage: ./comprehensive_bench [options]
//   --dataset <small|medium|large>  Dataset size (default: small)
//   --threads <N>                   Concurrent search threads (default: 1)
//   --output <path>                 JSON output path (default: stdout)

#include "pomai/pomai.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using namespace std::chrono;

// ============================================================================
// Configuration
// ============================================================================

struct BenchmarkConfig {
    std::string dataset_size = "small";  // small, medium, large
    int num_threads = 1;
    std::string output_path = "";
    
    // Dataset parameters
    uint32_t num_vectors = 10000;
    uint32_t dim = 128;
    uint32_t num_queries = 1000;
    uint32_t topk = 10;
    
    void configure() {
        if (dataset_size == "small") {
            num_vectors = 10000;
            dim = 128;
            num_queries = 1000;
        } else if (dataset_size == "medium") {
            num_vectors = 100000;
            dim = 256;
            num_queries = 5000;
        } else if (dataset_size == "large") {
            num_vectors = 1000000;
            dim = 768;
            num_queries = 10000;
        }
    }
};

// ============================================================================
// Dataset Generation
// ============================================================================

struct Dataset {
    std::vector<std::vector<float>> vectors;
    std::vector<std::vector<float>> queries;
    std::vector<std::vector<pomai::VectorId>> ground_truth;  // Ground truth top-k IDs
    
    void generate(uint32_t num_vecs, uint32_t dim, uint32_t num_q, uint32_t k) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        printf("Generating dataset: %u vectors @ %u dims, %u queries...\n",
               num_vecs, dim, num_q);
        
        // Generate vectors
        vectors.resize(num_vecs);
        for (auto& vec : vectors) {
            vec.resize(dim);
            for (auto& val : vec) {
                val = dist(rng);
            }
            normalize(vec);
        }
        
        // Generate queries
        queries.resize(num_q);
        for (auto& q : queries) {
            q.resize(dim);
            for (auto& val : q) {
                val = dist(rng);
            }
            normalize(q);
        }
        
        // Compute ground truth (brute-force)
        printf("Computing ground truth (brute-force)...\n");
        ground_truth.resize(num_q);
        for (uint32_t qi = 0; qi < num_q; ++qi) {
            auto& gt = ground_truth[qi];
            std::vector<std::pair<float, pomai::VectorId>> scores;
            scores.reserve(num_vecs);
            
            for (uint32_t i = 0; i < num_vecs; ++i) {
                float score = dot(queries[qi], vectors[i]);
                scores.emplace_back(score, i);
            }
            
            std::nth_element(scores.begin(), scores.begin() + k, scores.end(),
                           [](const auto& a, const auto& b) { return a.first > b.first; });
            scores.resize(k);
            std::sort(scores.begin(), scores.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
            
            gt.resize(k);
            for (uint32_t j = 0; j < k; ++j) {
                gt[j] = scores[j].second;
            }
            
            if ((qi + 1) % 100 == 0) {
                printf("  Ground truth progress: %u/%u\r", qi + 1, num_q);
                fflush(stdout);
            }
        }
        printf("\nGround truth complete.\n");
    }
    
private:
    void normalize(std::vector<float>& vec) {
        float norm = 0.0f;
        for (float v : vec) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-6f) {
            for (auto& v : vec) v /= norm;
        }
    }
    
    float dot(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
};

// ============================================================================
// Metrics
// ============================================================================

struct LatencyStats {
    std::vector<double> latencies_us;  // microseconds
    
    void record(double us) {
        latencies_us.push_back(us);
    }
    
    double percentile(double p) const {
        if (latencies_us.empty()) return 0.0;
        auto sorted = latencies_us;
        std::sort(sorted.begin(), sorted.end());
        size_t idx = static_cast<size_t>(p * sorted.size());
        if (idx >= sorted.size()) idx = sorted.size() - 1;
        return sorted[idx];
    }
    
    double mean() const {
        if (latencies_us.empty()) return 0.0;
        double sum = 0.0;
        for (double lat : latencies_us) sum += lat;
        return sum / latencies_us.size();
    }
};

struct BenchmarkResults {
    // Build metrics
    double build_time_sec = 0.0;
    size_t memory_bytes = 0;
    
    // Search metrics
    LatencyStats search_latency;
    double throughput_qps = 0.0;
    double recall_at_k = 0.0;
    
    void print() const {
        printf("\n");
        printf("=============================================================\n");
        printf("                  BENCHMARK RESULTS\n");
        printf("=============================================================\n");
        printf("\n");
        
        printf("BUILD METRICS\n");
        printf("  Build Time:       %.2f sec\n", build_time_sec);
        printf("  Memory Usage:     %.2f MB\n", memory_bytes / 1024.0 / 1024.0);
        printf("\n");
        
        printf("SEARCH LATENCY (microseconds)\n");
        printf("  Mean:             %.2f µs\n", search_latency.mean());
        printf("  P50:              %.2f µs\n", search_latency.percentile(0.50));
        printf("  P90:              %.2f µs\n", search_latency.percentile(0.90));
        printf("  P99:              %.2f µs\n", search_latency.percentile(0.99));
        printf("  P999:             %.2f µs\n", search_latency.percentile(0.999));
        printf("\n");
        
        printf("THROUGHPUT\n");
        printf("  QPS:              %.2f queries/sec\n", throughput_qps);
        printf("\n");
        
        printf("ACCURACY\n");
        printf("  Recall@k:         %.4f (%.2f%%)\n", recall_at_k, recall_at_k * 100.0);
        printf("\n");
        printf("=============================================================\n");
    }
    
    void save_json(const std::string& path) const {
        std::ofstream ofs(path);
        if (!ofs) {
            fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
            return;
        }
        
        ofs << "{\n";
        ofs << "  \"build\": {\n";
        ofs << "    \"time_sec\": " << build_time_sec << ",\n";
        ofs << "    \"memory_mb\": " << (memory_bytes / 1024.0 / 1024.0) << "\n";
        ofs << "  },\n";
        ofs << "  \"search_latency_us\": {\n";
        ofs << "    \"mean\": " << search_latency.mean() << ",\n";
        ofs << "    \"p50\": " << search_latency.percentile(0.50) << ",\n";
        ofs << "    \"p90\": " << search_latency.percentile(0.90) << ",\n";
        ofs << "    \"p99\": " << search_latency.percentile(0.99) << ",\n";
        ofs << "    \"p999\": " << search_latency.percentile(0.999) << "\n";
        ofs << "  },\n";
        ofs << "  \"throughput\": {\n";
        ofs << "    \"qps\": " << throughput_qps << "\n";
        ofs << "  },\n";
        ofs << "  \"accuracy\": {\n";
        ofs << "    \"recall_at_k\": " << recall_at_k << "\n";
        ofs << "  }\n";
        ofs << "}\n";
        
        printf("Results saved to %s\n", path.c_str());
    }
};

// ============================================================================
// Benchmark Harness
// ============================================================================

class Benchmark {
public:
    Benchmark(const BenchmarkConfig& cfg) : config_(cfg) {}
    
    BenchmarkResults run() {
        BenchmarkResults results;
        
        // Generate dataset
        dataset_.generate(config_.num_vectors, config_.dim, 
                         config_.num_queries, config_.topk);
        
        // Phase 1: Build index
        printf("\n[1/3] Building index...\n");
        auto build_start = high_resolution_clock::now();
        
        pomai::DBOptions opts;
        opts.dim = config_.dim;
        opts.shard_count = std::thread::hardware_concurrency();
        opts.path = "/tmp/pomai_bench_" + config_.dataset_size;
        opts.fsync = pomai::FsyncPolicy::kNever;  // Disable fsync for benchmark
        
        // Clean up old data
        fs::remove_all(opts.path);
        
        std::unique_ptr<pomai::DB> db;
        auto st = pomai::DB::Open(opts, &db);
        if (!st.ok()) {
            fprintf(stderr, "Failed to open DB: %s\n", st.message().c_str());
            exit(1);
        }
        
        // Insert vectors
        for (uint32_t i = 0; i < dataset_.vectors.size(); ++i) {
            st = db->Put(i, dataset_.vectors[i]);
            if (!st.ok()) {
                fprintf(stderr, "Put failed: %s\n", st.message().c_str());
                exit(1);
            }
            
            if ((i + 1) % 10000 == 0) {
                printf("  Inserted: %u/%u\r", i + 1, (uint32_t)dataset_.vectors.size());
                fflush(stdout);
            }
        }
        printf("\n");
        
        // Freeze to build segments
        st = db->Freeze("__default__");
        if (!st.ok()) {
            fprintf(stderr, "Freeze failed: %s\n", st.message().c_str());
        }
        
        auto build_end = high_resolution_clock::now();
        results.build_time_sec = duration<double>(build_end - build_start).count();
        
        // Estimate memory (rough)
        results.memory_bytes = dataset_.vectors.size() * config_.dim * sizeof(float);
        
        // Phase 2: Warmup
        printf("\n[2/3] Warmup (100 queries)...\n");
        pomai::SearchResult search_result;
        for (uint32_t i = 0; i < std::min(100u, config_.num_queries); ++i) {
            db->Search(dataset_.queries[i], config_.topk, &search_result);
        }
        
        // Phase 3: Benchmark search
        printf("\n[3/3] Benchmarking search (%u queries, %d threads)...\n",
               config_.num_queries, config_.num_threads);
        
        if (config_.num_threads == 1) {
            // Single-threaded
            auto search_start = high_resolution_clock::now();
            
            for (uint32_t qi = 0; qi < config_.num_queries; ++qi) {
                auto q_start = high_resolution_clock::now();
                
                st = db->Search(dataset_.queries[qi], config_.topk, &search_result);
                if (!st.ok()) {
                    fprintf(stderr, "Search failed: %s\n", st.message().c_str());
                    continue;
                }
                
                auto q_end = high_resolution_clock::now();
                double lat_us = duration<double, std::micro>(q_end - q_start).count();
                results.search_latency.record(lat_us);
                
                // Compute recall for this query
                double recall = compute_recall(search_result.hits, dataset_.ground_truth[qi]);
                recall_sum_ += recall;
                
                if ((qi + 1) % 100 == 0) {
                    printf("  Progress: %u/%u\r", qi + 1, config_.num_queries);
                    fflush(stdout);
                }
            }
            printf("\n");
            
            auto search_end = high_resolution_clock::now();
            double total_time = duration<double>(search_end - search_start).count();
            results.throughput_qps = config_.num_queries / total_time;
            
        } else {
            // Multi-threaded throughput test
            std::atomic<uint32_t> query_idx{0};
            std::vector<std::thread> threads;
            std::vector<LatencyStats> per_thread_stats(config_.num_threads);
            std::atomic<double> total_recall{0.0};
            
            auto search_start = high_resolution_clock::now();
            
            for (int t = 0; t < config_.num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    pomai::SearchResult local_result;
                    double local_recall_sum = 0.0;
                    uint32_t queries_done = 0;
                    
                    while (true) {
                        uint32_t qi = query_idx.fetch_add(1);
                        if (qi >= config_.num_queries) break;
                        
                        auto q_start = high_resolution_clock::now();
                        auto st = db->Search(dataset_.queries[qi], config_.topk, &local_result);
                        auto q_end = high_resolution_clock::now();
                        
                        if (st.ok()) {
                            double lat_us = duration<double, std::micro>(q_end - q_start).count();
                            per_thread_stats[t].record(lat_us);
                            
                            double recall = compute_recall(local_result.hits, dataset_.ground_truth[qi]);
                            local_recall_sum += recall;
                            queries_done++;
                        }
                    }
                    
                    total_recall.fetch_add(local_recall_sum);
                });
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
            
            auto search_end = high_resolution_clock::now();
            double total_time = duration<double>(search_end - search_start).count();
            results.throughput_qps = config_.num_queries / total_time;
            
            // Merge latencies
            for (const auto& stats : per_thread_stats) {
                for (double lat : stats.latencies_us) {
                    results.search_latency.record(lat);
                }
            }
            
            recall_sum_ = total_recall.load();
        }
        
        results.recall_at_k = recall_sum_ / config_.num_queries;
        
        // Cleanup
        db->Close();
        fs::remove_all(opts.path);
        
        return results;
    }
    
private:
    double compute_recall(const std::vector<pomai::SearchHit>& results,
                         const std::vector<pomai::VectorId>& ground_truth) {
        std::unordered_set<pomai::VectorId> gt_set(ground_truth.begin(), ground_truth.end());
        
        uint32_t hits = 0;
        for (const auto& result : results) {
            if (gt_set.count(result.id)) {
                hits++;
            }
        }
        
        return static_cast<double>(hits) / std::min(results.size(), ground_truth.size());
    }
    
    BenchmarkConfig config_;
    Dataset dataset_;
    double recall_sum_ = 0.0;
};

// ============================================================================
// Main
// ============================================================================

void print_usage() {
    printf("Usage: comprehensive_bench [options]\n");
    printf("Options:\n");
    printf("  --dataset <small|medium|large>  Dataset size (default: small)\n");
    printf("  --threads <N>                   Concurrent threads (default: 1)\n");
    printf("  --output <path>                 JSON output path (optional)\n");
    printf("\n");
    printf("Dataset sizes:\n");
    printf("  small:  10K vectors @ 128 dims\n");
    printf("  medium: 100K vectors @ 256 dims\n");
    printf("  large:  1M vectors @ 768 dims\n");
}

int main(int argc, char** argv) {
    BenchmarkConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--dataset" && i + 1 < argc) {
            config.dataset_size = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage();
            return 1;
        }
    }
    
    config.configure();
    
    printf("=============================================================\n");
    printf("            PomaiDB Comprehensive Benchmark\n");
    printf("=============================================================\n");
    printf("Dataset:     %s (%u vectors @ %u dims)\n",
           config.dataset_size.c_str(), config.num_vectors, config.dim);
    printf("Queries:     %u\n", config.num_queries);
    printf("Top-k:       %u\n", config.topk);
    printf("Threads:     %d\n", config.num_threads);
    printf("=============================================================\n");
    
    Benchmark bench(config);
    auto results = bench.run();
    
    results.print();
    
    if (!config.output_path.empty()) {
        results.save_json(config.output_path);
    }
    
    return 0;
}
