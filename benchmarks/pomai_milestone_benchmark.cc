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
#include <random>

#include "src/core/pomai_db.h"
#include "src/core/config.h"

using namespace std::chrono;

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

std::atomic<bool> is_searching{false};
std::atomic<bool> stop_all{false};
std::atomic<uint64_t> global_vectors_inserted{0};

void worker_loop(pomai::core::PomaiDB *db, std::string m_name, int dim, MilestoneThreadStats &stats, int thread_id)
{
    const int batch_sz = 100;
    const int top_k = 10;
    std::mt19937 gen(1337 + thread_id);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    while (!stop_all.load(std::memory_order_relaxed))
    {
        if (is_searching.load(std::memory_order_relaxed))
        {
            std::vector<float> query(dim);
            for (auto &x : query)
                x = dis(gen);

            auto start = high_resolution_clock::now();
            auto res = db->search(m_name, query.data(), top_k);
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
            std::vector<std::pair<uint64_t, std::vector<float>>> batch;
            batch.reserve(batch_sz);
            for (int j = 0; j < batch_sz; ++j)
            {
                std::vector<float> v(dim);
                for (auto &x : v)
                    x = dis(gen);
                batch.emplace_back(global_vectors_inserted.load() + j, std::move(v));
            }

            auto start = high_resolution_clock::now();
            if (db->insert_batch(m_name, batch))
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
              << std::setfill('=') << std::setw(135) << "" << std::setfill(' ') << "\n";
    std::cout << std::left << std::setw(12) << "Milestone"
              << std::setw(18) << "Ins Thrpt(v/s)" << std::setw(12) << "Ins P99(us)"
              << " | "
              << std::setw(18) << "Sea Thrpt(q/s)" << std::setw(12) << "Sea P50(us)"
              << std::setw(12) << "Sea P99(us)" << std::setw(12) << "EMA(ms)"
              << std::setw(12) << "Budget" << "\n";
    std::cout << std::setfill('-') << std::setw(135) << "" << std::setfill(' ') << "\n";
}

float get_system_cpu_load()
{
    float load = (global_vectors_inserted.load() / 100000000.0f) * 100.0f;
    return std::min(100.0f, load);
}

int main()
{
    const std::vector<uint64_t> milestones = {1'000'000, 5'000'000, 10'000'000, 20'000'000, 50'000'000, 100'000'000};
    const int threads = std::thread::hardware_concurrency();
    const int dim = 128;
    const std::string m_name = "autoblitz_stress";

    pomai::config::PomaiConfig cfg;
    cfg.res.data_root = "./data/milestone_bench";
    cfg.hot_tier.initial_capacity = 2'000'000;

    std::filesystem::remove_all(cfg.res.data_root);
    auto db = std::make_unique<pomai::core::PomaiDB>(cfg);

    pomai::core::MembranceConfig m_cfg;
    m_cfg.dim = dim;
    m_cfg.ram_mb = 10240;
    db->create_membrance(m_name, m_cfg);

    std::mt19937 gen_train(42);
    std::uniform_real_distribution<float> dis_train(0.0f, 1.0f);
    std::vector<float> train_vecs(dim * 10000);
    for (auto &x : train_vecs)
        x = dis_train(gen_train);

    auto *membr = db->get_membrance(m_name);
    if (membr && membr->orbit)
    {
        auto whisper = std::make_shared<pomai::ai::WhisperGrain>(cfg.whisper);
        membr->orbit->set_whisper_grain(whisper);

        std::cout << "[Init] Pre-training EchoGraph with 10,000 random vectors...\n";
        membr->orbit->train(train_vecs.data(), 10000);
    }

    std::cout << "[Init] Starting AutoBlitz Stress Test with " << threads << " threads\n";
    print_header();

    std::vector<MilestoneThreadStats> all_stats(threads);
    std::vector<std::thread> workers;
    for (int i = 0; i < threads; ++i)
        workers.emplace_back(worker_loop, db.get(), m_name, dim, std::ref(all_stats[i]), i);

    std::thread cpu_monitor([&]()
                            {
        auto *m = db->get_membrance(m_name);
        while (!stop_all.load()) {
            if (m && m->orbit && m->orbit->whisper_grain()) {
                m->orbit->whisper_grain()->set_cpu_load(get_system_cpu_load());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } });

    for (uint64_t target : milestones)
    {
        is_searching = false;
        auto start_time = high_resolution_clock::now();
        while (global_vectors_inserted.load() < target)
            std::this_thread::sleep_for(milliseconds(100));
        auto end_time = high_resolution_clock::now();
        double ins_dur = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;

        is_searching = true;
        std::this_thread::sleep_for(seconds(5));
        double sea_dur = 5.0;

        uint64_t m_ins_vecs = 0;
        uint64_t m_sea_ops = 0;
        LatencyStats combined_ins, combined_sea;

        for (auto &s : all_stats)
        {
            m_ins_vecs += s.vectors_inserted;
            m_sea_ops += s.search_ops;
            combined_ins.ns_list.insert(combined_ins.ns_list.end(), s.insert_stats.ns_list.begin(), s.insert_stats.ns_list.end());
            combined_sea.ns_list.insert(combined_sea.ns_list.end(), s.search_stats.ns_list.begin(), s.search_stats.ns_list.end());
            s.clear();
        }
        std::sort(combined_ins.ns_list.begin(), combined_ins.ns_list.end());
        std::sort(combined_sea.ns_list.begin(), combined_sea.ns_list.end());

        float current_ema = 0.0f;
        uint32_t current_budget = 0;
        if (membr && membr->orbit && membr->orbit->whisper_grain())
        {
            auto wg = membr->orbit->whisper_grain();
            current_ema = wg->latency_ema();
            current_budget = wg->compute_budget(false).ops_budget;
        }

        std::cout << std::left << std::setw(12) << target
                  << std::setw(18) << std::fixed << std::setprecision(2) << (m_ins_vecs / ins_dur)
                  << std::setw(12) << combined_ins.p(99)
                  << " | "
                  << std::setw(18) << (m_sea_ops / sea_dur)
                  << std::setw(12) << combined_sea.p(50)
                  << std::setw(12) << combined_sea.p(99)
                  << std::setw(12) << current_ema
                  << std::setw(12) << current_budget << std::endl;
    }

    stop_all = true;
    cpu_monitor.join();
    for (auto &t : workers)
        t.join();
    return 0;
}