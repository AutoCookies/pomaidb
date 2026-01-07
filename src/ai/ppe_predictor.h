#pragma once
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>

namespace pomai::ai
{

    struct PPEPredictor
    {
        // Simple hot/cold predictor based on last access time and access count.
        std::atomic<uint64_t> last_access_ms{0};
        std::atomic<uint32_t> hits{0};
        std::mutex m;

        PPEPredictor() = default;

        // Make the type movable but not copyable. We intentionally do not move the mutex;
        // the mutex is default-constructed for the destination which is safe.
        PPEPredictor(PPEPredictor &&other) noexcept
        {
            last_access_ms.store(other.last_access_ms.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hits.store(other.hits.load(std::memory_order_relaxed), std::memory_order_relaxed);
            // mutex left default-constructed
        }
        PPEPredictor &operator=(PPEPredictor &&other) noexcept
        {
            if (this != &other)
            {
                last_access_ms.store(other.last_access_ms.load(std::memory_order_relaxed), std::memory_order_relaxed);
                hits.store(other.hits.load(std::memory_order_relaxed), std::memory_order_relaxed);
                // mutex left default-constructed
            }
            return *this;
        }

        PPEPredictor(const PPEPredictor &) = delete;
        PPEPredictor &operator=(const PPEPredictor &) = delete;

        static inline uint64_t now_ms()
        {
            return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }

        // Call on access
        void touch()
        {
            hits.fetch_add(1, std::memory_order_relaxed);
            last_access_ms.store(now_ms(), std::memory_order_relaxed);
        }

        // Return predicted next-access timestamp (ms). For prototype: if hits>threshold -> far future.
        uint64_t predictNext() const
        {
            uint64_t last = last_access_ms.load(std::memory_order_relaxed);
            uint32_t h = hits.load(std::memory_order_relaxed);
            if (h > 20)
                return last + 24ULL * 3600 * 1000; // very hot -> predict far future
            if (h > 5)
                return last + 60 * 1000; // warm -> next minute
            return last + 1000;          // cold-ish -> next second
        }

        // Decide nbits: default 8, allow 4 for cold vectors if predicted next is soon
        uint8_t predictBits(uint8_t default_bits, uint64_t thresh_ms) const
        {
            uint64_t pred = predictNext();
            uint64_t now = now_ms();
            if (pred < now + thresh_ms)
                return 4;
            return default_bits;
        }
    };

} // namespace pomai::ai