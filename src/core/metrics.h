#pragma once
#include <atomic>
#include <cstdint>
#include <iomanip>
#include <iostream>

struct PomaiMetrics
{
    static std::atomic<uint64_t> hits;
    static std::atomic<uint64_t> misses;
    static std::atomic<uint64_t> puts;
    static std::atomic<uint64_t> evictions;
    static std::atomic<uint64_t> harvests;
    static std::atomic<uint64_t> arena_alloc_fails;
    static std::atomic<uint64_t> seed_allocs;
    static std::atomic<uint64_t> seed_frees;
    static std::atomic<uint64_t> blob_allocs;
    static std::atomic<uint64_t> blob_frees;
    static std::atomic<uint64_t> batch_encode_ns_total;
    static std::atomic<uint64_t> batch_sort_ns_total;
    static std::atomic<uint64_t> batch_write_ns_total;
    static std::atomic<uint64_t> batch_subbatches_processed;
    static std::atomic<uint64_t> total_searches;
    static std::atomic<uint64_t> searches_empty;
    static std::atomic<uint64_t> searches_fast_miss;
    static inline uint64_t last_total_ops = 0;
    static void reset()
    {
        hits.store(0);
        misses.store(0);
        puts.store(0);
        evictions.store(0);
        harvests.store(0);
        arena_alloc_fails.store(0);
        seed_allocs.store(0);
        seed_frees.store(0);
        blob_allocs.store(0);
        blob_frees.store(0);
        batch_encode_ns_total.store(0);
        batch_sort_ns_total.store(0);
        batch_write_ns_total.store(0);
        batch_subbatches_processed.store(0);
        total_searches.store(0);
        searches_empty.store(0);
        searches_fast_miss.store(0);
    }
    static void print_summary()
    {
        uint64_t h = hits.load();
        uint64_t m = misses.load();
        uint64_t p = puts.load();
        uint64_t e = evictions.load();
        uint64_t current_total = h + m + p + e;
        if (current_total == last_total_ops)
            return;
        last_total_ops = current_total;
        uint64_t total_queries = h + m;
        double hit_rate = (total_queries > 0) ? (double)h / total_queries * 100.0 : 0.0;
        std::clog << "[Metrics] "
                  << "Puts: " << p
                  << " | Hits: " << h
                  << " | Misses: " << m
                  << " | HitRate: " << std::fixed << std::setprecision(2) << hit_rate << "%"
                  << " | Evictions: " << e << "\n";
        if (arena_alloc_fails.load() > 0)
        {
            std::clog << "[Metrics] WARNING: Arena allocation failures: " << arena_alloc_fails.load() << "\n";
        }
        std::clog << "[Metrics] Searches=" << total_searches.load()
                  << ", Empty=" << searches_empty.load()
                  << ", FastMiss=" << searches_fast_miss.load() << "\n";
    }
};

// Only define the inline static atomic variables ONCE below
inline std::atomic<uint64_t> PomaiMetrics::hits{0};
inline std::atomic<uint64_t> PomaiMetrics::misses{0};
inline std::atomic<uint64_t> PomaiMetrics::puts{0};
inline std::atomic<uint64_t> PomaiMetrics::evictions{0};
inline std::atomic<uint64_t> PomaiMetrics::harvests{0};
inline std::atomic<uint64_t> PomaiMetrics::arena_alloc_fails{0};
inline std::atomic<uint64_t> PomaiMetrics::seed_allocs{0};
inline std::atomic<uint64_t> PomaiMetrics::seed_frees{0};
inline std::atomic<uint64_t> PomaiMetrics::blob_allocs{0};
inline std::atomic<uint64_t> PomaiMetrics::blob_frees{0};
inline std::atomic<uint64_t> PomaiMetrics::batch_encode_ns_total{0};
inline std::atomic<uint64_t> PomaiMetrics::batch_sort_ns_total{0};
inline std::atomic<uint64_t> PomaiMetrics::batch_write_ns_total{0};
inline std::atomic<uint64_t> PomaiMetrics::batch_subbatches_processed{0};
inline std::atomic<uint64_t> PomaiMetrics::total_searches{0};
inline std::atomic<uint64_t> PomaiMetrics::searches_empty{0};
inline std::atomic<uint64_t> PomaiMetrics::searches_fast_miss{0};