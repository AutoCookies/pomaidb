/*
 * src/ai/pppq.h
 *
 * PPPQ: prototype Product Quantizer with optional async demotion and basic metrics.
 *
 * This header exposes the PPPQ class (train, addVec, approxDist, purgeCold)
 * and lightweight demotion metrics used for instrumentation.
 *
 * Implementation notes:
 *  - Per-id metadata (code_nbits_, in_mmap_) are stored as atomic arrays.
 *  - A per-id sequence counter seq_ is used to provide a stable snapshot
 *    protocol for readers observing demote-related metadata and payload.
 *  - Async demote tasks are scheduled via std::async and tracked with futures.
 */

#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <future>
#include <memory>

#include "ppe_predictor.h"

namespace pomai::ai
{

    class PPPQ
    {
    public:
        // ctor/dtor
        PPPQ(size_t d, size_t m, size_t k, size_t max_elems, const std::string &mmap_file = "pppq_codes.mmap");
        ~PPPQ();

        // Return current stored bitness for given id (4 or 8). Safe to call concurrently.
        uint8_t get_code_nbits(size_t id) const noexcept;

        // train and insert
        void train(const float *samples, size_t n_samples, size_t max_iters = 10);
        void addVec(const float *vec, size_t id);

        // approximate distance between two ids (uses packed codes on-demand)
        float approxDist(size_t id_a, size_t id_b);

        // demotion sweep
        void purgeCold(uint64_t cold_thresh_ms);

        // basic accessors
        size_t dim() const { return dim_; }
        size_t m() const { return m_; }
        size_t subdim() const { return subdim_; }
        size_t k() const { return k_; }

        // demote metrics
        size_t get_pending_demotes() const noexcept { return pending_demotes_.load(std::memory_order_acquire); }
        uint64_t get_demote_tasks_completed() const noexcept { return demote_tasks_completed_.load(std::memory_order_acquire); }
        uint64_t get_demote_bytes_written() const noexcept { return demote_bytes_written_.load(std::memory_order_acquire); }
        double get_demote_avg_latency_ms() const noexcept;

        void reset_demote_metrics();

    private:
        // helpers
        int findNearestCode(const float *subvec, float *codebook) const;
        void computePrecompDists();
        void writePacked4ToFile(size_t id, const uint8_t *nibble_bytes, size_t nibble_bytes_len);
        void readPacked4FromFile(size_t id, uint8_t *nibble_bytes, size_t nibble_bytes_len);
        void pack4From8(const uint8_t *src8, uint8_t *dst_nibbles);
        void unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8);
        size_t packed4BytesPerVec() const { return (m_ + 1) / 2; }

        // async demote scheduling
        void schedule_async_demote(size_t id, const std::vector<uint8_t> &nibble_buf);

        // stable snapshot reader (member so it can access private state)
        // Returns true if a consistent snapshot was observed within max_attempts.
        // on success out_buf contains packed bytes when in_mmap==1, otherwise out_buf may be empty.
        bool stable_snapshot_read(size_t id,
                                  uint8_t &out_in_mmap,
                                  uint8_t &out_nbits,
                                  std::vector<uint8_t> &out_buf,
                                  size_t max_attempts = 100);

        // disable copy
        PPPQ(const PPPQ &) = delete;
        PPPQ &operator=(const PPPQ &) = delete;

        // internal state
        size_t dim_, m_, k_, subdim_, max_elems_;
        std::vector<float> codebooks_;
        std::vector<std::vector<float>> precomp_dists_;

        // in-ram codes (8-bit per sub)
        std::vector<uint8_t> codes8_; // size max_elems_ * m_

        // per-id metadata (atomics)
        std::unique_ptr<std::atomic<uint8_t>[]> code_nbits_; // 8 or 4
        std::unique_ptr<std::atomic<uint8_t>[]> in_mmap_;    // 0/1
        std::unique_ptr<std::atomic<uint32_t>[]> seq_;       // sequence counter per id

        // file-backed store
        std::string mmap_filename_;
        std::mutex mmap_file_mtx_;

        // PPE predictors
        std::vector<PPEPredictor> ppe_vecs_;

        // async demote bookkeeping & metrics
        std::atomic<size_t> pending_demotes_{0};
        std::atomic<uint64_t> demote_tasks_completed_{0};
        std::atomic<uint64_t> demote_bytes_written_{0};
        std::atomic<uint64_t> demote_total_latency_ns_{0};

        std::mutex demote_futures_mu_;
        std::vector<std::future<void>> demote_futures_;
    };

} // namespace pomai::ai