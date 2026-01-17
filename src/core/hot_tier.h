#pragma once
/*
 * src/core/hot_tier.h
 *
 * The "Shock Absorber" for PomaiDB.
 * A dense, write-optimized buffer for incoming vectors.
 *
 * Performance Tuning:
 * - Adaptive Hybrid Spinlock: Spins on CPU for short waits, yields to OS for long waits.
 * - Structure of Arrays (SoA): Perfect for cache locality & SIMD auto-vectorization.
 * - Pre-allocation: Minimizes resizing overhead.
 */

#include <vector>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <queue>
#include <cstring>
#include <utility>
#include <thread>

// Intrinsics for CPU pause
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#endif

namespace pomai::core
{
    // --- Helper: Adaptive Spinlock ---
    // Đây là "vũ khí bí mật" cho high-concurrency.
    // Nó không busy-wait đần độn mà biết backoff thông minh.
    class AdaptiveSpinLock
    {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;

    public:
        void lock()
        {
            // Fast path: Thử lấy lock ngay lập tức
            if (!flag.test_and_set(std::memory_order_acquire))
            {
                return;
            }

            // Slow path: Contention detected
            int spin_count = 0;
            while (flag.test_and_set(std::memory_order_acquire))
            {
                if (spin_count < 16)
                {
// Phase 1: Micro-sleep (CPU hint).
// Giúp CPU pipeline không bị flush, tiết kiệm điện và giảm latency bus RAM.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
                    _mm_pause();
#elif defined(__aarch64__)
                    asm volatile("isb"); // ARM barrier equivalent hint
#endif
                    spin_count++;
                }
                else
                {
                    // Phase 2: Yield to OS.
                    // Nếu chờ quá lâu, nhường CPU cho thread khác (tránh 100% CPU deadlock).
                    std::this_thread::yield();
                    spin_count = 0; // Reset backoff check
                }
            }
        }

        void unlock()
        {
            flag.clear(std::memory_order_release);
        }
    };

    // RAII Wrapper
    struct ScopedSpinLock
    {
        AdaptiveSpinLock &sl;
        ScopedSpinLock(AdaptiveSpinLock &l) : sl(l) { sl.lock(); }
        ~ScopedSpinLock() { sl.unlock(); }
    };

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
        HotTier(size_t dim, const pomai::config::HotTierConfig &cfg)
            : dim_(dim),
              capacity_hint_(cfg.initial_capacity)
        {
            labels_.reserve(capacity_hint_);
            data_.reserve(capacity_hint_ * dim_);
        }

        HotTier(const HotTier &) = delete;
        HotTier &operator=(const HotTier &) = delete;

        // ----------------------------------------------------------------
        // Insert Path: Write (Thread-Safe & Fast)
        // ----------------------------------------------------------------
        void push(uint64_t label, const float *vec)
        {
            ScopedSpinLock g(lock_);

            // Check capacity guard (optional simple protection)
            // Nếu vector grow quá lớn sẽ gây delay lock, nên giữ capacity hợp lý.
            labels_.push_back(label);

            // Manual copy is often faster/safer than std::copy for simple float arrays
            size_t current_size = data_.size();
            data_.resize(current_size + dim_);
            std::memcpy(data_.data() + current_size, vec, dim_ * sizeof(float));
        }

        // ----------------------------------------------------------------
        // Drain Path: Background Worker (Zero-Copy Handoff)
        // ----------------------------------------------------------------
        HotBatch swap_and_flush()
        {
            HotBatch batch;
            batch.dim = dim_;

            ScopedSpinLock g(lock_);
            if (labels_.empty())
            {
                return batch;
            }

            // Move semantics: Chuyển quyền sở hữu data sang batch cực nhanh (pointer swap)
            batch.labels = std::move(labels_);
            batch.data = std::move(data_);

            // Re-initialize buffers for next batch
            // Không dùng .clear() vì vector đã bị moved (trạng thái rỗng).
            // Ta tạo vector mới và reserve lại để đảm bảo hiệu năng insert tiếp theo.
            labels_ = std::vector<uint64_t>();
            data_ = std::vector<float>();
            labels_.reserve(capacity_hint_);
            data_.reserve(capacity_hint_ * dim_);

            return batch;
        }

        // ----------------------------------------------------------------
        // Search Path: Brute-Force (Vectorized)
        // ----------------------------------------------------------------
        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k) const
        {
            std::vector<std::pair<uint64_t, float>> results;

            ScopedSpinLock g(lock_);
            size_t n = labels_.size();
            if (n == 0)
                return results;

            // Use min-heap to keep top-k smallest distances
            // Pair: <distance, label>
            using Entry = std::pair<float, uint64_t>;
            std::priority_queue<Entry> pq;

            const float *dptr = data_.data();
            const size_t d = dim_;

            // Linear Scan
            for (size_t i = 0; i < n; ++i)
            {
                // Compute L2 Squared inline
                // Compiler (GCC/Clang -O3) will auto-vectorize (AVX2/AVX512) this loop heavily
                // because data is contiguous (SoA layout).
                float dist = 0.0f;
                const float *vec_i = dptr + (i * d);

                for (size_t j = 0; j < d; ++j)
                {
                    float diff = query[j] - vec_i[j];
                    dist += diff * diff;
                }

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

            // Finalize results
            results.reserve(pq.size());
            while (!pq.empty())
            {
                results.emplace_back(pq.top().second, pq.top().first);
                pq.pop();
            }
            // Reverse to return sorted by distance (ASC)
            std::reverse(results.begin(), results.end());

            return results;
        }

        size_t size() const
        {
            ScopedSpinLock g(lock_);
            return labels_.size();
        }

    private:
        size_t dim_;
        size_t capacity_hint_;

        // SoA Data Layout
        std::vector<uint64_t> labels_;
        std::vector<float> data_;

        // Optimized Lock
        mutable AdaptiveSpinLock lock_;
    };

} // namespace pomai::core