#pragma once
/*
 * src/core/hot_tier.h
 *
 * The "Shock Absorber" for PomaiDB.
 * A dense, write-optimized buffer for incoming vectors.
 *
 * Philosophy:
 * - Insert is O(1) memory copy (amortized).
 * - Search is Brute-force (fast for small N < 10k).
 * - Structure of Arrays (SoA) layout for cache locality.
 * - Spinlock protection (expected hold time < 500us).
 */

#include <vector>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <queue>
#include <cstring> // memcpy
#include <utility> // std::pair

namespace pomai::core
{

    // Data carrier for the background merger
    struct HotBatch
    {
        std::vector<uint64_t> labels;
        std::vector<float> data; // Flattened [v1_d1...dn, v2...]
        size_t dim;

        bool empty() const { return labels.empty(); }
        size_t count() const { return labels.size(); }
    };

    class HotTier
    {
    public:
        explicit HotTier(size_t dim, size_t capacity_hint = 4096)
            : dim_(dim), capacity_hint_(capacity_hint)
        {
            // Reserve to avoid immediate reallocations on first inserts
            labels_.reserve(capacity_hint_);
            data_.reserve(capacity_hint_ * dim_);
        }

        HotTier(const HotTier &) = delete;
        HotTier &operator=(const HotTier &) = delete;

        // ----------------------------------------------------------------
        // Insert Path: Critical Hot Path
        // ----------------------------------------------------------------
        // Thread-safe O(1) push.
        void push(uint64_t label, const float *vec)
        {
            LockGuard g(flag_);
            labels_.push_back(label);

            // Append vector data
            size_t current_size = data_.size();
            data_.resize(current_size + dim_);
            std::memcpy(data_.data() + current_size, vec, dim_ * sizeof(float));
        }

        // ----------------------------------------------------------------
        // Drain Path: Background Worker
        // ----------------------------------------------------------------
        // Atomically swaps the current buffer out to return to caller.
        // This is a "destructive read" - the HotTier is empty after this.
        HotBatch swap_and_flush()
        {
            HotBatch batch;
            batch.dim = dim_;

            LockGuard g(flag_);
            if (labels_.empty())
            {
                return batch;
            }

            // Move internals to batch (zero-copy handoff)
            batch.labels = std::move(labels_);
            batch.data = std::move(data_);

            // Reset internals (clean slate)
            // Re-reserve to keep subsequent inserts fast
            labels_ = std::vector<uint64_t>();
            data_ = std::vector<float>();
            labels_.reserve(capacity_hint_);
            data_.reserve(capacity_hint_ * dim_);

            return batch;
        }

        // ----------------------------------------------------------------
        // Search Path: Read
        // ----------------------------------------------------------------
        // Brute-force L2 search.
        // Blocks inserts for the duration of the scan.
        // [Perf Note]: Keep HotTier small (< 10k items) to keep this < 1ms.
        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k) const
        {
            std::vector<std::pair<uint64_t, float>> results;

            LockGuard g(flag_);
            size_t n = labels_.size();
            if (n == 0)
                return results;

            // Maintain top-k smallest distances.
            // std::priority_queue is a max-heap, so popping removes the largest distance.
            // This is exactly what we want (keep k smallest).
            using Entry = std::pair<float, uint64_t>; // <dist, label>
            std::priority_queue<Entry> pq;

            const float *dptr = data_.data();

            // Simple loop - easy for compiler to auto-vectorize
            for (size_t i = 0; i < n; ++i)
            {
                float dist = compute_l2_sqr(query, dptr + (i * dim_), dim_);

                if (pq.size() < k)
                {
                    pq.push({dist, labels_[i]});
                }
                else if (dist < pq.top().first)
                {
                    pq.pop();
                    pq.push({dist, labels_[i]});
                }
            }

            // Drain PQ into vector
            results.reserve(pq.size());
            while (!pq.empty())
            {
                results.emplace_back(pq.top().second, pq.top().first);
                pq.pop();
            }

            // Sort ascending by distance (PQ pop order is descending)
            std::reverse(results.begin(), results.end());

            return results;
        }

        size_t size() const
        {
            LockGuard g(flag_);
            return labels_.size();
        }

    private:
        // Local helper for L2 Squared
        static inline float compute_l2_sqr(const float *a, const float *b, size_t d)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < d; ++i)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return sum;
        }

        // Members
        size_t dim_;
        size_t capacity_hint_;

        // SoA Data Layout
        std::vector<uint64_t> labels_;
        std::vector<float> data_;

        // Spinlock
        mutable std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

        // RAII wrapper for atomic_flag
        struct LockGuard
        {
            std::atomic_flag &f;
            LockGuard(std::atomic_flag &flag) : f(flag)
            {
                while (f.test_and_set(std::memory_order_acquire))
                {
                    // Busy wait (spin)
                    // In C++20 we would use f.wait(true), but strict C++17 compat:
                    // just spin.
#if defined(__cpp_lib_atomic_wait)
                    f.wait(true, std::memory_order_relaxed);
#endif
                }
            }
            ~LockGuard()
            {
                f.clear(std::memory_order_release);
#if defined(__cpp_lib_atomic_wait)
                f.notify_one();
#endif
            }
        };
    };

} // namespace pomai::core