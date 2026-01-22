#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <iomanip>
#include <filesystem>

#include "src/core/pomai_db.h"
#include "src/core/config.h"
#include "src/core/metrics.h"

using namespace std::chrono;

// --- Thống kê chuẩn BigTech (Lock-free in hot path) ---
struct ThreadStats
{
    uint64_t total_ops = 0;
    uint64_t total_vectors = 0;
    std::vector<uint64_t> latencies_ns;

    void record(uint64_t ns, size_t num_vectors)
    {
        total_ops++;
        total_vectors += num_vectors;
        latencies_ns.push_back(ns);
    }
};

// --- Worker Thread: Gọi trực tiếp API nội bộ ---
void core_benchmark_worker(pomai::core::PomaiDB *db,
                           std::string membrance_name,
                           int iterations,
                           int batch_size,
                           int dim,
                           ThreadStats &stats)
{

    // 1. PRE-BAKING: Chuẩn bị dữ liệu mẫu trong RAM để triệt tiêu overhead allocation
    // Reuse một batch duy nhất để đo raw throughput của engine
    std::vector<std::pair<uint64_t, std::vector<float>>> batch;
    batch.reserve(batch_size);
    for (int j = 0; j < batch_size; ++j)
    {
        std::vector<float> vec(dim, 0.1f * j);
        batch.emplace_back(static_cast<uint64_t>(1000 + j), std::move(vec));
    }

    stats.latencies_ns.reserve(iterations);

    // 2. WARMUP: Làm nóng CPU Cache và nạp các trang nhớ (Page Faults)
    for (int i = 0; i < 50; ++i)
    {
        db->insert_batch(membrance_name, batch);
    }

    // 3. MEASUREMENT LOOP: Đo trực tiếp hàm insert_batch nội bộ
    for (int i = 0; i < iterations; ++i)
    {
        auto start = high_resolution_clock::now();

        // GỌI TRỰC TIẾP ENGINE API
        bool ok = db->insert_batch(membrance_name, batch);

        auto end = high_resolution_clock::now();

        if (ok)
        {
            auto ns = duration_cast<nanoseconds>(end - start).count();
            stats.record(ns, batch_size);
        }
    }
}

void print_report(const std::vector<ThreadStats> &all_stats, double duration_s, int batch_size)
{
    uint64_t grand_total_vectors = 0;
    std::vector<uint64_t> combined_latencies;

    for (const auto &s : all_stats)
    {
        grand_total_vectors += s.total_vectors;
        combined_latencies.insert(combined_latencies.end(), s.latencies_ns.begin(), s.latencies_ns.end());
    }

    std::sort(combined_latencies.begin(), combined_latencies.end());
    size_t n = combined_latencies.size();
    auto p = [&](double pct)
    { return n > 0 ? combined_latencies[static_cast<size_t>(n * pct / 100.0)] / 1000.0 : 0; };

    std::cout << "\n"
              << std::string(60, '=') << "\n";
    std::cout << " POMAI CORE DIRECT BENCHMARK (NO NETWORK / NO SQL)\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Throughput:  " << std::fixed << std::setprecision(2)
              << (grand_total_vectors / duration_s) << " vectors/sec\n";
    std::cout << "Batch Rate:  " << (combined_latencies.size() / duration_s) << " batches/sec\n";
    std::cout << "Total Vecs:  " << grand_total_vectors << "\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Latency per BATCH (batch_size=" << batch_size << ") in microseconds:\n";
    std::cout << "  P50:    " << std::setw(10) << p(50) << " us\n";
    std::cout << "  P99:    " << std::setw(10) << p(99) << " us\n";
    std::cout << "  Max:    " << std::setw(10) << (n > 0 ? combined_latencies.back() / 1000.0 : 0) << " us\n";
    std::cout << std::string(60, '=') << "\n";
}

int main()
{
    // 1. Cấu hình Core (Sử dụng Async WAL đã tối ưu)
    pomai::config::PomaiConfig config;
    config.res.data_root = "./data/core_bench";
    config.wal.sync_on_append = false;        // Tắt sync mỗi bản ghi để đạt max throughput
    config.wal.batch_commit_size = 64 * 1024; // 64KB group commit

    // Dọn dẹp data cũ để benchmark sạch
    std::filesystem::remove_all(config.res.data_root);

    // 2. Khởi tạo DB Core
    auto db = std::make_unique<pomai::core::PomaiDB>(config);

    // 3. Tạo membrance benchmark
    std::string m_name = "bench_core";
    pomai::core::MembranceConfig m_cfg;
    m_cfg.dim = 128;
    db->create_membrance(m_name, m_cfg);

    int threads = std::thread::hardware_concurrency();
    int iterations_per_thread = 5000;
    int batch_size = 100; // Tổng 500,000 vectors mỗi thread
    int dim = 128;

    std::cout << "[Init] Starting Core Benchmark with " << threads << " threads...\n";

    std::vector<ThreadStats> all_stats(threads);
    std::vector<std::thread> workers;

    auto start_time = high_resolution_clock::now();

    for (int i = 0; i < threads; ++i)
    {
        workers.emplace_back(core_benchmark_worker, db.get(), m_name,
                             iterations_per_thread, batch_size, dim, std::ref(all_stats[i]));
    }

    for (auto &t : workers)
        t.join();

    auto end_time = high_resolution_clock::now();
    double duration = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;

    print_report(all_stats, duration, batch_size);

    return 0;
}