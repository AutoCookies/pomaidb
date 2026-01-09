#pragma once
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>

namespace pomai::ai
{

    // PPEPredictor: lightweight per-element predictor used by demote/promote logic.
    // - last_access_ms and hits are atomic.
    // - touch() performs a relaxed increment of hits and a release-store of timestamp.
    // - predictNext() and other readers use acquire loads to observe recent updates.
    //
    // Memory-ordering rationale:
    //  - We don't need full seq_cst for this heuristic. Using release on the timestamp
    //    and acquire on reads provides a sensible synchronization point so that a
    //    reader that sees the new timestamp also observes a recent hits update.
    //  - All operations are wait-free/lock-free (mutex only kept for callers that
    //    need external serialization for non-atomic sequences).
    struct PPEPredictor
    {
        // last observed access time (milliseconds, steady clock). 0 means never touched.
        std::atomic<uint64_t> last_access_ms{0};

        // simple hit counter (relaxed is ok; we use acquire when reading for heuristic)
        std::atomic<uint32_t> hits{0};

        // Optional mutex for callers that want to perform compound operations (not used by touch/predict).
        std::mutex m;

        PPEPredictor() = default;

        // Movable but not copyable; mutex is default-constructed for destination.
        PPEPredictor(PPEPredictor &&other) noexcept
        {
            last_access_ms.store(other.last_access_ms.load(std::memory_order_relaxed), std::memory_order_relaxed);
            hits.store(other.hits.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        PPEPredictor &operator=(PPEPredictor &&other) noexcept
        {
            if (this != &other)
            {
                last_access_ms.store(other.last_access_ms.load(std::memory_order_relaxed), std::memory_order_relaxed);
                hits.store(other.hits.load(std::memory_order_relaxed), std::memory_order_relaxed);
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

        // Record an access.
        // - increment hits (relaxed)
        // - store timestamp with release semantics so readers using acquire see a consistent view.
        inline void touch() noexcept
        {
            hits.fetch_add(1u, std::memory_order_relaxed);
            uint64_t now = now_ms();
            last_access_ms.store(now, std::memory_order_release);
        }

        // Explicit touch with provided timestamp (useful in tests / restore)
        inline void touch_at(uint64_t ts_ms) noexcept
        {
            hits.fetch_add(1u, std::memory_order_relaxed);
            last_access_ms.store(ts_ms, std::memory_order_release);
        }

        // Set explicit values (used during restore); uses release ordering for timestamp and relaxed for hits.
        inline void set_last_ms(uint64_t v) noexcept { last_access_ms.store(v, std::memory_order_release); }
        inline void set_hits(uint32_t v) noexcept { hits.store(v, std::memory_order_relaxed); }

        // Return predicted next-access timestamp (ms).
        // Uses acquire loads so that if a caller sees a recent timestamp it also sees recent hits updates.
        uint64_t predictNext() const noexcept
        {
            uint64_t last = last_access_ms.load(std::memory_order_acquire);
            uint32_t h = hits.load(std::memory_order_acquire);

            if (h > 20)
                return last + 24ULL * 3600ULL * 1000ULL; // very hot -> +1 day
            if (h > 5)
                return last + 60ULL * 1000ULL; // warm -> +1 minute
            return last + 1000ULL;             // cold-ish -> +1 second
        }

        // Decide nbits: default 8, allow 4 when predicted next-access is within thresh_ms.
        uint8_t predictBits(uint8_t default_bits, uint64_t thresh_ms) const noexcept
        {
            uint64_t pred = predictNext();
            uint64_t now = now_ms();
            if (pred < now + thresh_ms)
                return 4;
            return default_bits;
        }

        // Convenience: check if this entry is considered cold relative to threshold_ms.
        inline bool is_cold_ms(uint64_t threshold_ms) const noexcept
        {
            uint64_t last = last_access_ms.load(std::memory_order_acquire);
            if (last == 0)
                return true;
            uint64_t now = now_ms();
            return (now - last) > threshold_ms;
        }
    };

} // namespace pomai::ai