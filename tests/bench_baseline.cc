#include <atomic>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <mutex>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <fstream>

#include "pomai/pomai.h"

using namespace pomai;

// Utils
std::vector<float> RandomVector(uint32_t dim) {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i) v[i] = dist(gen);
    return v;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::cout << "Starting Baseline Benchmark..." << std::endl;

    const uint32_t dim = 128; // Reduced to run fast
    const uint32_t n_shards = 4;
    const size_t initial_count = 50000;
    const size_t upsert_count = 50000;
    const std::chrono::seconds duration(5);

    DBOptions opt;
    opt.path = "bench_baseline_db";
    opt.dim = dim;
    opt.shard_count = n_shards;
    
    // Cleanup
    std::filesystem::remove_all(opt.path);

    std::unique_ptr<DB> db;
    if (!DB::Open(opt, &db).ok()) {
        std::cerr << "Open failed" << std::endl;
        return 1;
    }

    // Pre-fill
    std::cout << "Pre-filling " << initial_count << " vectors..." << std::endl;
    {
        std::vector<std::thread> loaders;
        size_t chunk = initial_count / 4;
        for (int i = 0; i < 4; ++i) {
            loaders.emplace_back([&, i]() {
                for (size_t j = 0; j < chunk; ++j) {
                    VectorId id = i * chunk + j;
                    auto v = RandomVector(dim);
                    db->Put(id, v);
                }
            });
        }
        for (auto& t : loaders) t.join();
    }
    std::cout << "Pre-fill done." << std::endl;

    std::atomic<bool> running{true};
    std::atomic<size_t> search_ops{0};
    std::atomic<size_t> write_ops{0};
    std::vector<double> latencies_ms;
    std::mutex lat_mu;

    // Writer Thread
    std::thread writer([&]() {
        size_t id_base = initial_count;
        while (running) {
            VectorId id = id_base + write_ops.load();
            auto v = RandomVector(dim);
            db->Put(id, v);
            write_ops++;
            // Small sleep to simulate realistic ingestion, but keep pressure
            // std::this_thread::sleep_for(std::chrono::microseconds(10)); 
            // Actually, we want MAX pressure to show blocking
        }
    });

    // Reader Threads
    int n_readers = 4;
    std::vector<std::thread> readers;
    for (int i = 0; i < n_readers; ++i) {
        readers.emplace_back([&]() {
            while (running) {
                auto q = RandomVector(dim);
                SearchResult res;
                auto start = std::chrono::high_resolution_clock::now();
                db->Search(q, 10, &res);
                auto end = std::chrono::high_resolution_clock::now();
                
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                {
                    std::lock_guard<std::mutex> lk(lat_mu);
                    if (latencies_ms.size() < 100000) // cap samples
                        latencies_ms.push_back(ms);
                }
                search_ops++;
            }
        });
    }

    std::this_thread::sleep_for(duration);
    running = false;
    writer.join();
    for (auto& t : readers) t.join();

    // Stats
    std::sort(latencies_ms.begin(), latencies_ms.end());
    double p50 = latencies_ms.empty() ? 0 : latencies_ms[latencies_ms.size() * 0.50];
    double p95 = latencies_ms.empty() ? 0 : latencies_ms[latencies_ms.size() * 0.95];
    double p99 = latencies_ms.empty() ? 0 : latencies_ms[latencies_ms.size() * 0.99];

    std::cout << "Results:" << std::endl;
    std::cout << "  Duration: " << duration.count() << "s" << std::endl;
    std::cout << "  Write Ops: " << write_ops << " (" << write_ops / duration.count() << " ops/s)" << std::endl;
    std::cout << "  Search Ops: " << search_ops << " (" << search_ops / duration.count() << " ops/s)" << std::endl;
    std::cout << "  Search Latency P50: " << p50 << " ms" << std::endl;
    std::cout << "  Search Latency P95: " << p95 << " ms" << std::endl;
    std::cout << "  Search Latency P99: " << p99 << " ms" << std::endl;

    // Output to markdown file
    {
        std::ofstream out("bench_baseline.md");
        out << "# Baseline Benchmark\n";
        out << "| Metric | Value |\n";
        out << "|---|---|\n";
        out << "| Writes | " << write_ops << " |\n";
        out << "| Searches | " << search_ops << " |\n";
        out << "| P99 Latency | " << p99 << " ms |\n";
    }

    return 0;
}
