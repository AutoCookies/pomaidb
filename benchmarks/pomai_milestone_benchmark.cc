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

using namespace std::chrono;

// --- Cấu trúc dữ liệu thống kê ---
struct LatencyStats
{
    std::vector<uint64_t> ns_list;
    void record(uint64_t ns) { ns_list.push_back(ns); }
    void clear() { ns_list.clear(); }

    double p(double pct) const
    {
        if (ns_list.empty())
            return 0;
        size_t idx = static_cast<size_t>(ns_list.size() * pct / 100.0);
        return ns_list[std::min(idx, ns_list.size() - 1)] / 1000.0;
    }
};

struct MilestoneThreadStats
{
    LatencyStats insert_stats;
    LatencyStats search_stats;
    uint64_t vectors_inserted = 0;
    uint64_t search_ops = 0;

    void clear()
    {
        insert_stats.clear();
        search_stats.clear();
        vectors_inserted = 0;
        search_ops = 0;
    }
};

// --- Sync & Controls ---
std::atomic<bool> is_searching{false};
std::atomic<bool> stop_all{false};
std::atomic<uint64_t> global_vectors_inserted{0};

// --- Worker Loop ---
void worker_loop(pomai::core::PomaiDB *db, std::string m_name, int dim, MilestoneThreadStats &stats)
{
    const int batch_sz = 100;
    const int top_k = 10;

    std::vector<std::pair<uint64_t, std::vector<float>>> insert_batch;
    for (int j = 0; j < batch_sz; ++j)
        insert_batch.emplace_back(100 + j, std::vector<float>(dim, 0.5f));
    std::vector<float> query_vec(dim, 0.5f);

    while (!stop_all.load(std::memory_order_relaxed))
    {
        if (is_searching.load(std::memory_order_relaxed))
        {
            auto start = high_resolution_clock::now();
            auto res = db->search(m_name, query_vec.data(), top_k);
            auto ns = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();

            if (!res.empty())
            {
                stats.search_stats.record(ns);
                stats.search_ops++;
            }
            std::this_thread::yield();
        }
        else
        {
            auto start = high_resolution_clock::now();
            if (db->insert_batch(m_name, insert_batch))
            {
                auto ns = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
                stats.insert_stats.record(ns);
                stats.vectors_inserted += batch_sz;
                global_vectors_inserted.fetch_add(batch_sz, std::memory_order_relaxed);
            }
        }
    }
}

void print_header()
{
    std::cout << "\n"
              << std::setfill('=') << std::setw(115) << "" << std::setfill(' ') << "\n";
    std::cout << std::left << std::setw(12) << "Milestone"
              << std::setw(18) << "Ins Thrpt(v/s)" << std::setw(12) << "Ins P99(us)"
              << " | "
              << std::setw(18) << "Sea Thrpt(q/s)" << std::setw(12) << "Sea P50(us)" << std::setw(12) << "Sea P99(us)" << "\n";
    std::cout << std::setfill('-') << std::setw(115) << "" << std::setfill(' ') << "\n";
}

int main()
{
    const std::vector<uint64_t> milestones = {1'000'000, 5'000'000, 10'000'000, 20'000'000};
    const int threads = std::thread::hardware_concurrency();
    const int dim = 128;
    const std::string m_name = "scaling_stress";

    pomai::config::PomaiConfig cfg;
    cfg.res.data_root = "./data/milestone_bench";
    cfg.wal.sync_on_append = false;
    cfg.hot_tier.initial_capacity = 1'000'000;

    std::filesystem::remove_all(cfg.res.data_root);
    auto db = std::make_unique<pomai::core::PomaiDB>(cfg);
    db->create_membrance(m_name, {dim, 2048, "orbit", true});

    std::cout << "[Init] Starting Milestone Insert & Search Benchmark (" << threads << " threads)\n";
    print_header();

    std::vector<MilestoneThreadStats> all_stats(threads);
    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i)
        workers.emplace_back(worker_loop, db.get(), m_name, dim, std::ref(all_stats[i]));

    for (uint64_t target : milestones)
    {
        // Phase 1: Ingest
        is_searching = false;
        auto start_time = high_resolution_clock::now();
        while (global_vectors_inserted.load() < target)
            std::this_thread::sleep_for(milliseconds(100));
        auto end_time = high_resolution_clock::now();
        double ins_dur = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;

        // Phase 2: Search (Stress for 5 seconds)
        is_searching = true;
        std::this_thread::sleep_for(seconds(1)); // Allow index to settle
        auto s_start_time = high_resolution_clock::now();
        std::this_thread::sleep_for(seconds(5));
        auto s_end_time = high_resolution_clock::now();
        double sea_dur = duration_cast<milliseconds>(s_end_time - s_start_time).count() / 1000.0;

        // Summary and aggregation
        uint64_t m_ins_vecs = 0;
        uint64_t m_sea_ops = 0;
        LatencyStats combined_ins, combined_sea;

        for (auto &s : all_stats)
        {
            m_ins_vecs += s.vectors_inserted;
            m_sea_ops += s.search_ops;

            // [FIXED] Sửa lỗi truy cập thành viên ns_list đúng cách
            combined_ins.ns_list.insert(combined_ins.ns_list.end(), s.insert_stats.ns_list.begin(), s.insert_stats.ns_list.end());
            combined_sea.ns_list.insert(combined_sea.ns_list.end(), s.search_stats.ns_list.begin(), s.search_stats.ns_list.end());
            s.clear();
        }
        std::sort(combined_ins.ns_list.begin(), combined_ins.ns_list.end());
        std::sort(combined_sea.ns_list.begin(), combined_sea.ns_list.end());

        std::cout << std::left << std::setw(12) << target
                  << std::setw(18) << std::fixed << std::setprecision(2) << (m_ins_vecs / ins_dur)
                  << std::setw(12) << combined_ins.p(99)
                  << " | "
                  << std::setw(18) << (m_sea_ops / sea_dur)
                  << std::setw(12) << combined_sea.p(50)
                  << std::setw(12) << combined_sea.p(99) << std::endl;
    }

    stop_all = true;
    for (auto &t : workers)
        t.join();
    std::cout << std::setfill('=') << std::setw(115) << "" << "\n[Done]\n";
    return 0;
}