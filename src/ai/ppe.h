#pragma once
#include <atomic>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace pomai::ai
{

    // Flag bits for PPEHeader.flags:
    static constexpr uint32_t PPE_FLAG_INDIRECT = 0x1; // payload contains arena-local offset (uint64_t)
    static constexpr uint32_t PPE_FLAG_REMOTE = 0x2;   // payload contains remote id (demoted blob)

    static constexpr uint32_t PPE_PRECISION_SHIFT = 8;
    static constexpr uint32_t PPE_PRECISION_MASK = 0xFFu << PPE_PRECISION_SHIFT;

    struct PPEHeader
    {
        // last observed access time (nanoseconds, steady clock). 0 means never touched.
        std::atomic<int64_t> last_access_ns{0};

        // EMA of inter-arrival interval in nanoseconds.
        std::atomic<double> ema_interval_ns{0.0};

        // 32-bit flags for storage and precision hints.
        uint32_t flags{0};

        // Per-node adaptive hints (M and ef).
        std::atomic<uint16_t> hint_M{0};
        std::atomic<uint16_t> hint_ef{0};

        // Stored label for the element. This field is written once at insert time
        // and may be read concurrently by distance functions (non-atomic is OK for
        // our use because label is written before the element becomes visible).
        // Using atomic here to be safe under concurrent readers/writers.
        std::atomic<uint64_t> label{0};

        PPEHeader() noexcept = default;

        // Set/get precision (number of bits used for quantized payloads).
        void set_precision(uint32_t bits) noexcept
        {
            uint32_t b = (bits & 0xFFu);
            flags = (flags & ~PPE_PRECISION_MASK) | (b << PPE_PRECISION_SHIFT);
        }

        uint32_t get_precision() const noexcept
        {
            return (flags & PPE_PRECISION_MASK) >> PPE_PRECISION_SHIFT;
        }

        // Label helpers
        void set_label(uint64_t v) noexcept { label.store(v, std::memory_order_relaxed); }
        uint64_t get_label() const noexcept { return label.load(std::memory_order_relaxed); }

        // Initialize adaptive hints (called when a header is constructed/restored).
        void init_hints(uint16_t m, uint16_t ef) noexcept
        {
            hint_M.store(m, std::memory_order_relaxed);
            hint_ef.store(ef, std::memory_order_relaxed);
        }

        uint16_t get_hint_M() const noexcept { return hint_M.load(std::memory_order_relaxed); }
        uint16_t get_hint_ef() const noexcept { return hint_ef.load(std::memory_order_relaxed); }

        // Bump helpers
        void bump_hint_M(uint16_t v) noexcept
        {
            uint16_t oldv = hint_M.load(std::memory_order_relaxed);
            while (oldv < v && !hint_M.compare_exchange_weak(oldv, v, std::memory_order_relaxed))
            {
            }
        }
        void bump_hint_ef(uint16_t v) noexcept
        {
            uint16_t oldv = hint_ef.load(std::memory_order_relaxed);
            while (oldv < v && !hint_ef.compare_exchange_weak(oldv, v, std::memory_order_relaxed))
            {
            }
        }

        // Update the predictor with an observed access (timestamp in ns).
        void touch_ns(int64_t access_time_ns) noexcept
        {
            int64_t prev = last_access_ns.exchange(access_time_ns, std::memory_order_relaxed);
            if (prev == 0)
                return;
            double interval = static_cast<double>(access_time_ns - prev);
            double oldEma = ema_interval_ns.load(std::memory_order_relaxed);
            const double alpha = 0.1;
            double newEma = alpha * interval + (1.0 - alpha) * oldEma;
            ema_interval_ns.store(newEma, std::memory_order_relaxed);
        }

        static inline int64_t now_ns() noexcept
        {
            using namespace std::chrono;
            return static_cast<int64_t>(duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
        }

        void touch_now() noexcept { touch_ns(now_ns()); }

        bool is_cold_ns(int64_t threshold_ns) const noexcept
        {
            int64_t last = last_access_ns.load(std::memory_order_relaxed);
            if (last == 0)
                return true;
            int64_t now = now_ns();
            return (now - last) > threshold_ns;
        }

        int64_t predict_next_ns() const noexcept
        {
            int64_t last = last_access_ns.load(std::memory_order_relaxed);
            double ema = ema_interval_ns.load(std::memory_order_relaxed);
            if (last == 0)
                return now_ns() + static_cast<int64_t>(std::llround(ema));
            return last + static_cast<int64_t>(std::llround(ema));
        }
    };

} // namespace pomai::ai