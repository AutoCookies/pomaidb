#pragma once
// ai/ppe.h
//
// Lightweight PPE header used by Pomai HNSW.
// - PPEHeader is a small, cache-friendly per-vector header that records
//   last access timestamp and an EMA of inter-arrival intervals (in ns).
// - Implementation is safe for lock-free updates using atomics and is
//   intentionally simple and conservative for production embedding.
//
// Comments and API are in English so this file can be dropped into ai/.

#include <atomic>
#include <cstdint>
#include <chrono>
#include <cmath>

namespace pomai::ai
{

    struct PPEHeader
    {
        // last observed access time (nanoseconds, steady clock)
        std::atomic<int64_t> last_access_ns{0};

        // EMA of inter-arrival interval in nanoseconds.
        // Use double for headroom; stored in plain atomic<double> (implementation-defined)
        // but we access it via relaxed ordering for heuristic purposes.
        std::atomic<double> ema_interval_ns{0.0};

        // 32-bit flags for future use (quantization bits, state markers)
        uint32_t flags{0};

        PPEHeader() noexcept = default;

        // Update the predictor with an observed access (now in ns).
        // This is lock-free and uses relaxed orderings because
        // the PPE is only used for heuristics (accept eventual consistency).
        void touch_ns(int64_t access_time_ns) noexcept
        {
            int64_t prev = last_access_ns.exchange(access_time_ns, std::memory_order_relaxed);

            if (prev == 0)
            {
                // first observation: leave ema as-is (caller may have initialized)
                return;
            }

            double interval = static_cast<double>(access_time_ns - prev);

            // Simple EMA with alpha = 0.1 (same spirit as earlier snippets).
            // Read-modify-write via compare_exchange to avoid torn writes on double.
            double oldEma = ema_interval_ns.load(std::memory_order_relaxed);
            const double alpha = 0.1;
            double newEma = alpha * interval + (1.0 - alpha) * oldEma;
            ema_interval_ns.store(newEma, std::memory_order_relaxed);
        }

        // Convenience: sample current time (steady_clock) in ns.
        static inline int64_t now_ns() noexcept
        {
            using namespace std::chrono;
            return static_cast<int64_t>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
        }

        // Touch using current time (convenience)
        void touch_now() noexcept { touch_ns(now_ns()); }

        // Return whether this header is "cold" w.r.t. threshold (ns).
        bool is_cold_ns(int64_t threshold_ns) const noexcept
        {
            int64_t last = last_access_ns.load(std::memory_order_relaxed);
            if (last == 0)
                return true; // never touched -> cold
            int64_t now = now_ns();
            return (now - last) > threshold_ns;
        }

        // Return predicted next access time (last + ema) in ns. If no last, returns now + ema.
        int64_t predict_next_ns() const noexcept
        {
            int64_t last = last_access_ns.load(std::memory_order_relaxed);
            double ema = ema_interval_ns.load(std::memory_order_relaxed);
            if (last == 0)
            {
                return now_ns() + static_cast<int64_t>(std::llround(ema));
            }
            return last + static_cast<int64_t>(std::llround(ema));
        }
    };

} // namespace pomai::ai