#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <fstream>

#include "pomai/pomai.h"

using namespace pomai;

std::vector<float> RandomVector(uint32_t dim, std::mt19937& gen) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (size_t i = 0; i < dim; ++i) v[i] = dist(gen);
    return v;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::cout << "Starting Baseline Benchmark (single-threaded)..." << std::endl;

    const uint32_t dim = 128;
    const size_t initial_count = 50000;
    const std::chrono::seconds duration(5);

    std::mt19937 gen(12345);

    DBOptions opt;
    opt.path = "bench_baseline_db";
    opt.dim = dim;
    // Legacy field; monolithic runtime uses a single logical instance.
    opt.shard_count = 1;

    std::filesystem::remove_all(opt.path);

    std::unique_ptr<DB> db;
    if (!DB::Open(opt, &db).ok()) {
        std::cerr << "Open failed" << std::endl;
        return 1;
    }

    std::cout << "Pre-filling " << initial_count << " vectors..." << std::endl;
    for (size_t j = 0; j < initial_count; ++j) {
        VectorId id = static_cast<VectorId>(j);
        auto v = RandomVector(dim, gen);
        db->Put(id, v);
    }
    std::cout << "Pre-fill done." << std::endl;

    size_t write_ops = 0;
    size_t search_ops = 0;
    std::vector<double> latencies_ms;
    latencies_ms.reserve(100000);

    const auto deadline = std::chrono::steady_clock::now() + duration;
    size_t id_base = initial_count;

    while (std::chrono::steady_clock::now() < deadline) {
        auto v = RandomVector(dim, gen);
        db->Put(static_cast<VectorId>(id_base + write_ops), v);
        ++write_ops;

        auto q = RandomVector(dim, gen);
        SearchResult res;
        auto start = std::chrono::high_resolution_clock::now();
        db->Search(q, 10, &res);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (latencies_ms.size() < 100000)
            latencies_ms.push_back(ms);
        ++search_ops;
    }

    std::sort(latencies_ms.begin(), latencies_ms.end());
    double p50 = latencies_ms.empty() ? 0 : latencies_ms[latencies_ms.size() * 50 / 100];
    double p95 = latencies_ms.empty() ? 0 : latencies_ms[latencies_ms.size() * 95 / 100];
    double p99 = latencies_ms.empty() ? 0 : latencies_ms[latencies_ms.size() * 99 / 100];

    std::cout << "Results:" << std::endl;
    std::cout << "  Duration: " << duration.count() << "s" << std::endl;
    std::cout << "  Write Ops: " << write_ops << " (" << write_ops / duration.count() << " ops/s)" << std::endl;
    std::cout << "  Search Ops: " << search_ops << " (" << search_ops / duration.count() << " ops/s)" << std::endl;
    std::cout << "  Search Latency P50: " << p50 << " ms" << std::endl;
    std::cout << "  Search Latency P95: " << p95 << " ms" << std::endl;
    std::cout << "  Search Latency P99: " << p99 << " ms" << std::endl;

    {
        std::ofstream out("bench_baseline.md");
        out << "# Baseline Benchmark (single-threaded)\n";
        out << "| Metric | Value |\n";
        out << "|---|---|\n";
        out << "| Writes | " << write_ops << " |\n";
        out << "| Searches | " << search_ops << " |\n";
        out << "| P99 Latency | " << p99 << " ms |\n";
    }

    return 0;
}
