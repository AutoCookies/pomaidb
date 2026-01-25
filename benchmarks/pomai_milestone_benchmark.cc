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
#include <numeric>

#include "src/core/pomai_db.h"
#include "src/core/config.h"

using namespace std::chrono;

struct LatencyStats
{
    std::vector<uint64_t> ns_list;
    void record(uint64_t ns) { ns_list.push_back(ns); }
    void clear() { ns_list.clear(); }
    // percentile in microseconds
    double p(double pct) const
    {
        if (ns_list.empty())
            return 0;
        size_t idx = static_cast<size_t>(ns_list.size() * pct / 100.0);
        idx = std::min(idx, ns_list.size() - 1);
        return static_cast<double>(ns_list[idx]) / 1000.0;
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
static std::atomic<uint64_t> next_label{0};

static std::mutex sampled_lock;
static std::vector<std::pair<uint64_t, std::vector<float>>> sampled_vectors;
static const size_t sample_per_batch = 1000;
static const size_t max_samples = 10000;

// Worker: produces inserts when is_searching==false, otherwise runs searches
void worker_loop(pomai::core::PomaiDB *db, const std::string &m_name, int dim, MilestoneThreadStats &stats, int thread_id)
{
    const int batch_sz = 1000;
    const int top_k = 10;
    std::mt19937 gen(1337 + static_cast<unsigned>(thread_id));
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Local buffers to reduce allocation in loop
    std::vector<std::pair<uint64_t, std::vector<float>>> batch;
    batch.reserve(batch_sz);

    std::vector<float> query(dim);

    while (!stop_all.load(std::memory_order_relaxed))
    {
        if (is_searching.load(std::memory_order_relaxed))
        {
            for (auto &x : query)
                x = dis(gen);

            auto start = high_resolution_clock::now();
            auto res = db->search(m_name, query.data(), top_k);
            auto ns = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();

            if (!res.empty())
            {
                stats.search_stats.record(static_cast<uint64_t>(ns));
                stats.search_ops++;
            }
            // Yield to avoid tight-loop burning CPU
            std::this_thread::yield();
        }
        else
        {
            uint64_t base_label = next_label.fetch_add(static_cast<uint64_t>(batch_sz), std::memory_order_relaxed);

            batch.clear();
            for (int j = 0; j < batch_sz; ++j)
            {
                std::vector<float> v(dim);
                for (auto &x : v)
                    x = dis(gen);
                batch.emplace_back(base_label + j, std::move(v));
            }

            auto start = high_resolution_clock::now();
            if (db->insert_batch(m_name, batch))
            {
                auto ns = duration_cast<nanoseconds>(high_resolution_clock::now() - start).count();
                stats.insert_stats.record(static_cast<uint64_t>(ns));
                stats.vectors_inserted += static_cast<uint64_t>(batch_sz);
                global_vectors_inserted.fetch_add(static_cast<uint64_t>(batch_sz), std::memory_order_relaxed);

                // sample a few for recall checks (keep under max_samples)
                std::lock_guard<std::mutex> lg(sampled_lock);
                if (sampled_vectors.size() < max_samples)
                {
                    for (size_t s = 0; s < sample_per_batch && s < batch.size() && sampled_vectors.size() < max_samples; ++s)
                    {
                        // copy only what's necessary (we keep vector<float>)
                        sampled_vectors.emplace_back(batch[s].first, batch[s].second);
                    }
                }
            }
            else
            {
                // Insert failed -> backoff slightly
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
}

void print_header()
{
    std::cout << "\n"
              << std::setfill('=') << std::setw(160) << "" << std::setfill(' ') << "\n";
    std::cout << std::left << std::setw(12) << "Milestone"
              << std::setw(18) << "Ins Thrpt(v/s)" << std::setw(12) << "Ins P99(us)"
              << " | "
              << std::setw(18) << "Sea Thrpt(q/s)" << std::setw(12) << "Sea P50(us)"
              << std::setw(12) << "Sea P99(us)" << std::setw(12) << "EMA(ms)"
              << std::setw(10) << "Budget" << std::setw(10) << "R@1" << std::setw(10) << "R@10"
              << std::setw(12) << "MemVecs" << "\n";
    std::cout << std::setfill('-') << std::setw(160) << "" << std::setfill(' ') << "\n";
}

// crude system CPU proxy used by the benchmark controller
float get_system_cpu_load()
{
    // Keep same scaling as original for comparability
    float load = (global_vectors_inserted.load(std::memory_order_relaxed) / 100000000.0f) * 100.0f;
    return std::min(100.0f, load);
}

int main()
{
    const std::vector<uint64_t> milestones = {1'000'000, 5'000'000, 10'000'000, 20'000'000, 50'000'000, 100'000'000};

    unsigned hc = std::thread::hardware_concurrency();
    size_t threads = (hc == 0) ? 1u : static_cast<size_t>(hc); // MUST be >= 1

    const int dim = 128;
    const std::string m_name = "autoblitz_stress";

    pomai::config::PomaiConfig cfg;
    cfg.res.data_root = "./data/milestone_bench";

    // ensure clean start
    try
    {
        std::filesystem::remove_all(cfg.res.data_root);
    }
    catch (...)
    {
    }

    // Reserve sample vector to avoid reallocation in hot path
    {
        std::lock_guard<std::mutex> lg(sampled_lock);
        sampled_vectors.reserve(max_samples);
    }

    auto db = std::make_unique<pomai::core::PomaiDB>(cfg);

    pomai::core::MembranceConfig m_cfg;
    m_cfg.dim = dim;
    m_cfg.ram_mb = 10240;
    db->create_membrance(m_name, m_cfg);

    // lightweight pre-train data
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
    workers.reserve(threads);

    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back(worker_loop, db.get(), m_name, dim, std::ref(all_stats[i]), static_cast<int>(i));

    std::thread cpu_monitor([&]()
                            {
        auto *m = db->get_membrance(m_name);
        while (!stop_all.load(std::memory_order_relaxed))
        {
            if (m && m->orbit && m->orbit->whisper_grain())
            {
                m->orbit->whisper_grain()->set_cpu_load(get_system_cpu_load());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        } });

    for (uint64_t target : milestones)
    {
        // Insert phase until target is reached
        is_searching.store(false, std::memory_order_relaxed);
        auto start_time = high_resolution_clock::now();
        while (global_vectors_inserted.load(std::memory_order_relaxed) < target)
            std::this_thread::sleep_for(milliseconds(100));
        auto end_time = high_resolution_clock::now();
        double ins_dur = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;

        // Short search phase to measure query latency under load
        is_searching.store(true, std::memory_order_relaxed);
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
        size_t membrance_total_vectors = 0;
        if (membr && membr->orbit && membr->orbit->whisper_grain())
        {
            auto wg = membr->orbit->whisper_grain();
            current_ema = wg->latency_ema();
            current_budget = wg->compute_budget(false).ops_budget;
        }
        if (membr && membr->orbit)
        {
            try
            {
                auto info = membr->orbit->get_info();
                membrance_total_vectors = info.num_vectors;
            }
            catch (...)
            {
                membrance_total_vectors = 0;
            }
        }

        // recall checks on sampled vectors (bounded)
        double recall_at_1 = 0.0;
        double recall_at_10 = 0.0;
        size_t recall_checked = 0;
        {
            std::vector<std::pair<uint64_t, std::vector<float>>> local_samples;
            {
                std::lock_guard<std::mutex> lg(sampled_lock);
                local_samples = sampled_vectors; // small bounded copy
                sampled_vectors.clear();
            }
            size_t max_check = std::min<size_t>(1000, local_samples.size());
            if (max_check > 0)
            {
                std::mt19937 rng(12345);
                std::uniform_int_distribution<size_t> dist(0, local_samples.size() - 1);
                size_t hits1 = 0, hits10 = 0;
                for (size_t i = 0; i < max_check; ++i)
                {
                    size_t idx = (local_samples.size() <= max_check) ? i : dist(rng);
                    auto &p = local_samples[idx];
                    uint64_t lbl = p.first;
                    const float *vec = p.second.data();
                    int k = 10;
                    auto res = db->search(m_name, vec, k);
                    if (!res.empty())
                    {
                        if (res.front().first == lbl)
                            hits1++;
                        bool found = false;
                        for (auto &r : res)
                            if (r.first == lbl)
                            {
                                found = true;
                                break;
                            }
                        if (found)
                            hits10++;
                    }
                }
                recall_checked = max_check;
                recall_at_1 = static_cast<double>(hits1) / static_cast<double>(recall_checked);
                recall_at_10 = static_cast<double>(hits10) / static_cast<double>(recall_checked);
            }
        }

        std::cout << std::left << std::setw(12) << target
                  << std::setw(18) << std::fixed << std::setprecision(2) << (m_ins_vecs / ins_dur)
                  << std::setw(12) << combined_ins.p(99)
                  << " | "
                  << std::setw(18) << (m_sea_ops / sea_dur)
                  << std::setw(12) << combined_sea.p(50)
                  << std::setw(12) << combined_sea.p(99)
                  << std::setw(12) << current_ema
                  << std::setw(10) << current_budget
                  << std::setw(10) << std::fixed << std::setprecision(3) << recall_at_1
                  << std::setw(10) << std::fixed << std::setprecision(3) << recall_at_10
                  << std::setw(12) << membrance_total_vectors << std::endl;

        if (membr)
        {
            std::cout << "[GET MEMBRANCE INFO] name=" << membr->name << "\n";
            try
            {
                auto info = membr->orbit ? membr->orbit->get_info() : pomai::ai::orbit::MembranceInfo{};
                std::cout << " feature_dim: " << info.dim << "\n";
                std::cout << " total_vectors: " << info.num_vectors << "\n";
                std::cout << " disk_bytes: " << (info.disk_bytes) << "\n";
                std::cout << " ram_mb_configured: " << membr->ram_mb << "\n";
            }
            catch (...)
            {
                std::cout << " ERR reading membrance info\n";
            }
        }
    }

    stop_all.store(true);
    if (cpu_monitor.joinable())
        cpu_monitor.join();
    for (auto &t : workers)
        if (t.joinable())
            t.join();
    return 0;
}