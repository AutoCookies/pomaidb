/*
 * src/ai/pppq.h
 *
 * PPPQ: prototype Product Quantizer with optional async demotion and basic metrics.
 *
 * This header was extended to expose lightweight demotion metrics:
 *  - pending demotes (queue length)
 *  - demote tasks completed
 *  - total bytes written by demotes
 *  - cumulative demote latency (ns) to compute average latency
 *
 * The metrics are atomic counters / simple aggregates intended for monitoring
 * and lightweight instrumentation during benchmarks.
 */

#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <future>

#include "ppe_predictor.h"

namespace pomai::ai
{

    class PPPQ
    {
    public:
        // d: dimensionality, m: number of subquantizers, k: codebook size per sub (e.g., 256)
        // max_elems: maximum number of elements this instance will manage
        // mmap_file: file used for packed-4 backing store
        PPPQ(size_t d, size_t m, size_t k, size_t max_elems, const std::string &mmap_file = "pppq_codes.mmap");
        ~PPPQ();

        // Train codebooks from samples (n_samples x dim)
        void train(const float *samples, size_t n_samples, size_t max_iters = 10);

        // Encode and add vector for id (0..max_elems-1). Predictor used to choose bits.
        void addVec(const float *vec, size_t id);

        // Approximate distance between two vectors by their PQ codes (L2). Will read demoted codes on demand.
        float approxDist(size_t id_a, size_t id_b);

        // Periodic purge: demote predicted-cold vectors to file-backed store
        void purgeCold(uint64_t cold_thresh_ms);

        // For integration / debugging
        size_t dim() const { return dim_; }
        size_t m() const { return m_; }
        size_t subdim() const { return subdim_; }
        size_t k() const { return k_; }

        // ---------------- Demote metrics (simple monitoring helpers) --------------
        // Current number of pending demote tasks (async queue length)
        size_t get_pending_demotes() const noexcept { return pending_demotes_.load(std::memory_order_acquire); }

        // Total number of completed demote tasks
        uint64_t get_demote_tasks_completed() const noexcept { return demote_tasks_completed_.load(std::memory_order_acquire); }

        // Total bytes written by demote operations (cumulative)
        uint64_t get_demote_bytes_written() const noexcept { return demote_bytes_written_.load(std::memory_order_acquire); }

        // Average demote latency in milliseconds (0.0 if no completed tasks)
        double get_demote_avg_latency_ms() const noexcept
        {
            uint64_t done = demote_tasks_completed_.load(std::memory_order_acquire);
            if (done == 0)
                return 0.0;
            uint64_t tot_ns = demote_total_latency_ns_.load(std::memory_order_acquire);
            return static_cast<double>(tot_ns) / static_cast<double>(done) / 1e6;
        }

        // Reset metrics to zero (not thread-safe with concurrent demotes; intended for test harness use)
        void reset_demote_metrics()
        {
            pending_demotes_.store(0, std::memory_order_release);
            demote_tasks_completed_.store(0, std::memory_order_release);
            demote_bytes_written_.store(0, std::memory_order_release);
            demote_total_latency_ns_.store(0, std::memory_order_release);
        }

    private:
        // Private helpers (implementation in .cc)
        int findNearestCode(const float *subvec, float *codebook) const;
        void computePrecompDists();
        void writePacked4ToFile(size_t id, const uint8_t *nibble_bytes, size_t nibble_bytes_len);
        void readPacked4FromFile(size_t id, uint8_t *nibble_bytes, size_t nibble_bytes_len);
        void pack4From8(const uint8_t *src8, uint8_t *dst_nibbles);
        void unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8);
        size_t packed4BytesPerVec() const { return (m_ + 1) / 2; }

        // Async demote scheduling helper
        void schedule_async_demote(size_t id, const std::vector<uint8_t> &nibble_buf);

        // disable copy
        PPPQ(const PPPQ &) = delete;
        PPPQ &operator=(const PPPQ &) = delete;

        // ---------------- internal state ----------------
        size_t dim_, m_, k_, subdim_, max_elems_;
        std::vector<float> codebooks_; // size m * k * subdim
        // precomputed pairwise distances for each sub: m blocks of k * k floats
        std::vector<std::vector<float>> precomp_dists_;

        // In-RAM per-vector storage:
        // - codes8_: each element id stores m bytes (one 0..k-1 per sub)
        // - when demoted to 4-bit we pack nibbles to mmap file; code_nbits_ stores current bitness per id
        std::vector<uint8_t> codes8_;     // size max_elems_ * m_
        std::vector<uint8_t> code_nbits_; // 8 or 4, size max_elems_
        std::vector<uint8_t> in_mmap_;    // 0/1 flag size max_elems_

        // mmap/file-backed store file name and mutex
        std::string mmap_filename_;
        std::mutex mmap_file_mtx_;

        // PPE predictors per element
        std::vector<PPEPredictor> ppe_vecs_;

        // ---------------- async demote bookkeeping & metrics ----------------
        // current number of pending async demotes (incremented before task scheduled)
        std::atomic<size_t> pending_demotes_{0};

        // Completed demote counters / aggregates
        std::atomic<uint64_t> demote_tasks_completed_{0};
        std::atomic<uint64_t> demote_bytes_written_{0};
        std::atomic<uint64_t> demote_total_latency_ns_{0};

        // Keep futures of outstanding demote tasks so destructor can join them
        std::mutex demote_futures_mu_;
        std::vector<std::future<void>> demote_futures_;
    };

} // namespace pomai::ai