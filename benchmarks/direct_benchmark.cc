/*
 * src/tools/direct_benchmark.cc
 *
 * PomaiDB Direct API Benchmark - HIGH PERFORMANCE EDITION
 * [UPDATED] Synced with Centralized Config Architecture.
 *
 * Optimizations:
 * 1. Multi-threaded Inserts (Auto-detect CPU cores).
 * 2. Pre-generated Random Pool (Zero RNG overhead during measurement).
 * 3. Atomic workload distribution.
 */

#include "src/core/pomai_db.h"
#include "src/core/cpu_kernels.h"
#include "src/core/config.h" // [ADDED] Include config

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <atomic>
#include <algorithm>
#include <iomanip>
#include <sys/resource.h>
#include <numeric>
#include <mutex>
#include <filesystem>

using namespace pomai::core;
using std::chrono::high_resolution_clock;

// ---- CONFIGURATION ----
const size_t DIM = 512;
const std::vector<uint64_t> MILESTONES = {
    100'000, 
    500'000, 
    1'000'000, 
    5'000'000, 
    10'000'000, 
    50'000'000, 
    100'000'000,
    500'000'000,
    1'000'000'000
};

// [AUTO-DETECT THREADS]
// Lấy số core thực tế của máy. Nếu không lấy được thì fallback về 4.
const size_t INSERT_THREADS = [](){
    unsigned int n = std::thread::hardware_concurrency();
    return n > 0 ? n : 4;
}();

const size_t INSERT_BATCH_SIZE = 5000; 
const size_t POOL_SIZE = 1'000'000; // Tăng pool lên 1M để giảm trùng lặp pattern
const size_t SEARCH_THREADS = 16;
const size_t SEARCH_TRIALS = 2000;

const std::string DB_PATH = "./data_direct_bench";
const std::string MEMBR_NAME = "titan_core";
const size_t RAM_LIMIT_MB = 10240; // 10GB

// ---- GLOBAL RESOURCES ----
std::vector<float> DATA_POOL; 

// ---- HELPERS ----

long get_ram_usage_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024;
}

void init_data_pool() {
    std::cout << "[Init] Pre-generating " << POOL_SIZE << " vectors to RAM... ";
    std::flush(std::cout);
    DATA_POOL.resize(POOL_SIZE * DIM);
    
    size_t num_th = INSERT_THREADS;
    std::vector<std::thread> workers;
    size_t chunk = POOL_SIZE / num_th;
    
    for(size_t t=0; t<num_th; ++t) {
        workers.emplace_back([t, chunk, num_th](){
            std::mt19937 rng(42 + t);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            size_t start = t * chunk * DIM;
            size_t end = (t == num_th - 1) ? POOL_SIZE * DIM : (t+1) * chunk * DIM;
            for(size_t i=start; i<end; ++i) {
                DATA_POOL[i] = dist(rng);
            }
        });
    }
    for(auto& w : workers) w.join();
    std::cout << "Done.\n";
}

struct LatencyStats { double p50, p99; };
LatencyStats calc_stats(std::vector<double>& lats) {
    if (lats.empty()) return {0, 0};
    std::sort(lats.begin(), lats.end());
    return { lats[lats.size() * 0.50], lats[lats.size() * 0.99] };
}

// ---- MAIN ----

int main() {
    // 0. Kernel Init
    pomai_init_cpu_kernels();
    
    std::cout << "========================================================\n";
    std::cout << "🔥 POMAI DIRECT BENCHMARK (AUTO-THREADING) 🔥\n";
    std::cout << "   CPU Cores: " << INSERT_THREADS << " | Dim: " << DIM << "\n";
    std::cout << "   RAM Limit: " << RAM_LIMIT_MB << " MB\n";
    std::cout << "========================================================\n\n";

    // 1. Setup Data
    init_data_pool();

    // 2. Setup DB with Config [FIXED]
    try { std::filesystem::remove_all(DB_PATH); } catch(...) {}
    
    // [FIXED] Initialize Config
    pomai::config::PomaiConfig config;
    config.res.data_root = DB_PATH;
    
    // Optimize for Benchmark: Disable WAL sync for raw insert speed testing if applicable
    // (Though PomaiDB WAL mainly tracks metadata, it's good practice)
    config.wal.sync_on_append = false; 
    
    // Disable metrics printing during benchmark to avoid clutter
    config.metrics.enabled = false;

    // Initialize DB with Config
    PomaiDB db(config);

    MembranceConfig m_cfg;
    m_cfg.dim = DIM;
    m_cfg.ram_mb = RAM_LIMIT_MB;
    m_cfg.engine = "orbit"; // Explicitly set engine
    
    if (!db.create_membrance(MEMBR_NAME, m_cfg)) {
        std::cerr << "Failed to create membrance!\n";
        return 1;
    }

    // 3. Execution State
    uint64_t current_vectors = 0;
    
    std::cout << "\n|   Vectors   | RAM (MB) | Ins Speed (v/s) | Search P50 (ms) | Search P99 (ms) |\n";
    std::cout << "|-------------|----------|-----------------|-----------------|-----------------|\n";

    for (uint64_t target : MILESTONES) {
        // --- INSERT PHASE (MULTI-THREADED) ---
        uint64_t needed = target - current_vectors;
        
        if (needed > 0) {
            std::atomic<uint64_t> atomic_ptr{0};
            auto t_start = high_resolution_clock::now();
            
            std::vector<std::thread> workers;
            // Bung số thread bằng số core máy
            for(size_t i=0; i<INSERT_THREADS; ++i) {
                workers.emplace_back([&, i]() {
                    std::vector<std::pair<uint64_t, std::vector<float>>> batch;
                    batch.reserve(INSERT_BATCH_SIZE);
                    
                    while(true) {
                        uint64_t my_base = atomic_ptr.fetch_add(INSERT_BATCH_SIZE);
                        if (my_base >= needed) break;
                        
                        uint64_t count = std::min((uint64_t)INSERT_BATCH_SIZE, needed - my_base);
                        
                        batch.clear();
                        // Lấy data từ pool, offset theo thread ID để tránh cache thrashing
                        size_t pool_offset = (my_base + i * 12345) % (POOL_SIZE - count); 
                        const float* src = DATA_POOL.data() + pool_offset * DIM;
                        
                        for(size_t k=0; k<count; ++k) {
                            std::vector<float> vec(src + k*DIM, src + (k+1)*DIM);
                            batch.push_back({current_vectors + my_base + k, std::move(vec)});
                        }
                        
                        db.insert_batch(MEMBR_NAME, batch);
                    }
                });
            }
            
            for(auto& w : workers) w.join();
            
            auto t_end = high_resolution_clock::now();
            double duration = std::chrono::duration<double>(t_end - t_start).count();
            double ips = needed / duration;
            current_vectors = target;

            // --- SEARCH PHASE (MEASURE) ---
            std::vector<double> latencies;
            std::mutex lat_mu;
            std::atomic<size_t> q_counter{0};
            std::vector<std::thread> s_workers;

            // Search concurrency giữ ở mức vừa phải hoặc bằng số core
            size_t search_concurrency = std::min((size_t)32, INSERT_THREADS * 2); 

            auto s_start = high_resolution_clock::now();
            for(size_t i=0; i<search_concurrency; ++i) {
                s_workers.emplace_back([&, i](){
                    std::vector<double> local_lats;
                    while(true) {
                        size_t q_idx = q_counter.fetch_add(1);
                        if (q_idx >= SEARCH_TRIALS) break;
                        
                        size_t pool_idx = (q_idx * 7919 + i) % POOL_SIZE;
                        const float* q = DATA_POOL.data() + pool_idx * DIM;
                        
                        auto t1 = high_resolution_clock::now();
                        auto res = db.search(MEMBR_NAME, q, 10);
                        auto t2 = high_resolution_clock::now();
                        
                        local_lats.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
                    }
                    std::lock_guard<std::mutex> lk(lat_mu);
                    latencies.insert(latencies.end(), local_lats.begin(), local_lats.end());
                });
            }
            for(auto& w : s_workers) w.join();

            LatencyStats stats = calc_stats(latencies);
            
            std::cout << "| " << std::setw(11) << current_vectors 
                      << " | " << std::setw(8) << get_ram_usage_mb()
                      << " | " << std::setw(15) << (int)ips 
                      << " | " << std::setw(15) << std::fixed << std::setprecision(2) << stats.p50 
                      << " | " << std::setw(15) << stats.p99 << " |\n";
        }
    }
    
    std::cout << "\n[Done] Benchmark finished.\n";
    return 0;
}