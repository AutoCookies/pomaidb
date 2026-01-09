/*
 * src/ai/ppe.h
 *
 * PPEHeader: per-element header containing adaptive hints and storage flags.
 *
 * This version adds a small sequence counter (seqlock-style) and helper APIs
 * so readers can take a consistent snapshot of (flags, payload) and writers
 * can publish/unpublish payloads safely across threads/tests that expect the
 * invariant "if flags has INDIRECT then payload != 0".
 *
 * Note: PPEHeader remains POD-like for placement-new into mmap'd memory.
 */

#pragma once
#include <atomic>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <thread> // <<-- added for std::this_thread::yield()

#include "src/ai/atomic_utils.h"

namespace pomai::ai
{

    // Flag bits for PPEHeader.flags:
    static constexpr uint32_t PPE_FLAG_INDIRECT = 0x1; // payload contains arena-local offset (uint64_t)
    static constexpr uint32_t PPE_FLAG_REMOTE = 0x2;   // payload contains remote id (demoted blob)

    static constexpr uint32_t PPE_PRECISION_SHIFT = 8;
    static constexpr uint32_t PPE_PRECISION_MASK = 0xFFu << PPE_PRECISION_SHIFT;

    // Ensure PPEHeader is 8-byte aligned so a uint64_t payload placed immediately
    // after the header will be 8-aligned.
    struct alignas(8) PPEHeader
    {
        // Small seqlock-like counter. Even => stable; odd => writer in progress.
        // Placed first to keep it well-aligned and to minimize false-sharing with other fields.
        std::atomic<uint32_t> seq{0};

        // last observed access time (nanoseconds, steady clock). 0 means never touched.
        std::atomic<int64_t> last_access_ns{0};

        // EMA of inter-arrival interval in nanoseconds.
        std::atomic<double> ema_interval_ns{0.0};

        // 32-bit flags for storage and precision hints.
        // Access/modification must use provided helpers when concurrency is possible.
        uint32_t flags{0};

        // Per-node adaptive hints (M and ef).
        std::atomic<uint16_t> hint_M{0};
        std::atomic<uint16_t> hint_ef{0};

        // Stored label for the element.
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

        // ----------------------------
        // Atomic flag helpers (existing)
        // ----------------------------
        inline void atomic_set_flags(uint32_t bits) noexcept
        {
            pomai::ai::atomic_utils::atomic_fetch_or_u32(&flags, bits);
        }

        inline void atomic_clear_flags(uint32_t bits) noexcept
        {
            pomai::ai::atomic_utils::atomic_fetch_and_u32(&flags, ~bits);
        }

        inline uint32_t atomic_load_flags() const noexcept
        {
            return pomai::ai::atomic_utils::atomic_load_u32(&flags);
        }

        // --------------------------------------------------------------------
        // New: seqlock-style helpers to publish/unpublish payloads and to read a
        // consistent snapshot of (flags, payload).
        //
        // Usage conventions:
        //  - payload_ptr points to uint64_t location immediately after this header
        //    (i.e. reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(this) + sizeof(PPEHeader)))
        //  - Writers should call atomic_publish_payload(...) to write payload and
        //    set the given flag bits atomically with respect to readers using snapshot.
        //  - Writers should call atomic_unpublish_payload(...) to clear flag bits
        //    and reset payload (if desired).
        //  - Readers that need to observe flags+payload consistently should call
        //    atomic_snapshot_payload_and_flags(...).
        // --------------------------------------------------------------------

        inline void atomic_publish_payload(uint64_t *payload_ptr, uint64_t value, uint32_t flagbits) noexcept
        {
            // Enter writer critical section (make seq odd)
            seq.fetch_add(1, std::memory_order_acq_rel); // now odd

            // Store payload (release)
            pomai::ai::atomic_utils::atomic_store_u64(payload_ptr, value);

            // Publish flags (fetch_or seq_cst to avoid races with other flag modifiers)
            pomai::ai::atomic_utils::atomic_fetch_or_u32(&flags, flagbits);

            // Leave critical section (make seq even)
            seq.fetch_add(1, std::memory_order_acq_rel);
        }

        inline void atomic_unpublish_payload(uint64_t *payload_ptr, uint32_t flagbits) noexcept
        {
            // Enter writer critical section (make seq odd)
            seq.fetch_add(1, std::memory_order_acq_rel); // odd

            // Clear flag bits first (so there is small window where flag cleared but payload still present).
            // Use seq_cst RMW for flags.
            pomai::ai::atomic_utils::atomic_fetch_and_u32(&flags, ~flagbits);

            // Then reset payload to zero (release)
            pomai::ai::atomic_utils::atomic_store_u64(payload_ptr, 0);

            // Leave critical section (make seq even)
            seq.fetch_add(1, std::memory_order_acq_rel);
        }

        // Reader snapshot: attempts to read flags and payload as a consistent snapshot.
        // Returns true on success (flags_out and payload_out are set).
        // This will retry if a writer is in progress (seq odd) or if seq changes during read.
        inline bool atomic_snapshot_payload_and_flags(const uint64_t *payload_ptr, uint32_t &flags_out, uint64_t &payload_out) const noexcept
        {
            for (;;)
            {
                uint32_t s1 = seq.load(std::memory_order_acquire);
                // If writer in progress, retry
                if (s1 & 1)
                {
                    std::this_thread::yield();
                    continue;
                }

                // read flags and payload
                uint32_t f = pomai::ai::atomic_utils::atomic_load_u32(&flags);
                uint64_t v = pomai::ai::atomic_utils::atomic_load_u64(payload_ptr);

                uint32_t s2 = seq.load(std::memory_order_acquire);
                if (s1 == s2 && !(s2 & 1))
                {
                    flags_out = f;
                    payload_out = v;
                    return true;
                }
                // else retry
                std::this_thread::yield();
            }
            // unreachable
            return false;
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

    static_assert(alignof(PPEHeader) >= alignof(uint64_t), "PPEHeader must be at least 8-byte aligned");
    static_assert(sizeof(PPEHeader) % alignof(uint64_t) == 0, "sizeof(PPEHeader) must be multiple of 8 so following payload is 8-byte aligned");

} // namespace pomai::ai