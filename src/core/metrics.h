// core/metrics.h
#pragma once
#include <atomic>
#include <cstdint>

/*
 * Very small metrics aggregator for core events.
 * Uses atomics so metrics are safe to update without locks.
 * Added a few arena-related counters for observability.
 */

struct PomaiMetrics
{
    static std::atomic<uint64_t> hits;
    static std::atomic<uint64_t> misses;
    static std::atomic<uint64_t> puts;
    static std::atomic<uint64_t> evictions;
    static std::atomic<uint64_t> harvests;
    static std::atomic<uint64_t> arena_alloc_fails;

    // New arena metrics
    static std::atomic<uint64_t> seed_allocs;
    static std::atomic<uint64_t> seed_frees;
    static std::atomic<uint64_t> blob_allocs;
    static std::atomic<uint64_t> blob_frees;

    // Batch-insert performance metrics (ns)
    // total nanoseconds spent in encode phase across all batches
    static std::atomic<uint64_t> batch_encode_ns_total;
    // total nanoseconds spent in sort phase across all batches
    static std::atomic<uint64_t> batch_sort_ns_total;
    // total nanoseconds spent in write phase across all batches
    static std::atomic<uint64_t> batch_write_ns_total;
    // number of logical sub-batches processed
    static std::atomic<uint64_t> batch_subbatches_processed;

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
    }
};

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