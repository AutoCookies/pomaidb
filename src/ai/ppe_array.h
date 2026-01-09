/*
 * src/ai/ppe_array.h
 *
 * Compact PPE predictor array for per-vector hot/cold predictions.
 *
 * Purpose
 * -------
 * Provide a small, mmap-friendly per-vector predictor used by the Thaut65/PomaiLight
 * pipeline and by PPPQ demotion logic. Each entry is tiny (16 bytes) and contains:
 *   - atomic<uint64_t> last_access_ms : last observed access timestamp (ms)
 *   - atomic<uint32_t> hits           : access count (relaxed)
 *   - uint32_t reserved               : padding / future use
 *
 * Design notes
 * ------------
 * - The struct is intentionally minimal and uses only atomic primitives that are
 *   safe to placement-new into mmap'd memory. To initialize an array in mapped
 *   memory call PPEArray::initialize(mapping_ptr, count).
 *
 * - Methods are noexcept and use relaxed atomics where strict ordering is not required.
 *   The predictor is heuristic; absolute precision is not required and relaxed/order-
 *   relaxed policies improve performance.
 *
 * - The API exposes:
 *     * PPEEntry: per-entry helpers (touch, predictNextMs, predictBits, is_cold_ms)
 *     * PPEArray: thin wrapper over a contiguous PPEEntry buffer with helpers to
 *                initialize, touch, and query entries by index.
 *
 * Thread-safety
 * -------------
 * - Concurrent readers/writers are supported: touch() and predictNextMs() use atomics.
 * - If the array resides in a file mapping, ensure the mapping is writable when
 *   initialize() is called (placement-new writes must be allowed).
 *
 * Usage
 * -----
 *   // allocate or mmap region of size PPEArray::required_bytes(count)
 *   void *mem = mmap(...);
 *   PPEArray::initialize(mem, count);           // placement-new each entry
 *   PPEArray ppe(mem, count);                   // create wrapper (no ownership)
 *   ppe.touch(idx);                             // record access
 *   uint64_t pred = ppe.predictNextMs(idx);     // predicted next-access in ms
 *   uint8_t bits = ppe.predictBits(idx, 8, 5000); // decide 4/8 bits preference
 *
 * Tested: portable, small, easy to instrument.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <chrono>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace pomai::ai
{

    // Small POD-like predictor entry sized to 16 bytes (aligned).
    // Layout: [8 bytes last_access_ms][4 bytes hits][4 bytes reserved]
    struct alignas(8) PPEEntry
    {
        std::atomic<uint64_t> last_access_ms; // milliseconds since epoch (steady clock)
        std::atomic<uint32_t> hits;           // simple hit counter (relaxed)
        uint32_t reserved;                    // padding / future flags

        // No-throw default constructor leaves fields zeroed for safety when placement-new used.
        PPEEntry() noexcept : last_access_ms(0), hits(0), reserved(0) {}

        // Record an access (touch). Fast, relaxed updates for hit counter and timestamp.
        inline void touch() noexcept
        {
            hits.fetch_add(1u, std::memory_order_relaxed);
            // store timestamp with relaxed order; exact ordering between producers/consumers is not required
            uint64_t now = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
            last_access_ms.store(now, std::memory_order_relaxed);
        }

        // Return predicted next-access timestamp (ms). Heuristic:
        //  - if hits > 20 => far future (very hot)
        //  - if hits > 5 => next minute
        //  - otherwise next second
        // This mirrors PPEPredictor::predictNext semantics but uses ms units.
        inline uint64_t predictNextMs() const noexcept
        {
            uint64_t last = last_access_ms.load(std::memory_order_relaxed);
            uint32_t h = hits.load(std::memory_order_relaxed);

            if (h > 20)
                return last + 24ULL * 3600ULL * 1000ULL; // far future: +1 day
            if (h > 5)
                return last + 60ULL * 1000ULL; // warm: +1 minute
            return last + 1000ULL;             // cold-ish: +1 second
        }

        // Decide preferred nbits for PQ encoding based on prediction.
        // Returns either default_bits or 4 (for aggressive compression) when predicted next
        // access is soon (within thresh_ms).
        inline uint8_t predictBits(uint8_t default_bits, uint64_t thresh_ms) const noexcept
        {
            uint64_t pred = predictNextMs();
            uint64_t now = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
            if (pred < now + thresh_ms)
                return 4;
            return default_bits;
        }

        // Returns true if entry is considered cold relative to threshold_ms (since last touch).
        inline bool is_cold_ms(uint64_t threshold_ms) const noexcept
        {
            uint64_t last = last_access_ms.load(std::memory_order_relaxed);
            if (last == 0)
                return true;
            uint64_t now = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
            return (now - last) > threshold_ms;
        }

        // Convenience to set explicit values (e.g., during restore). Uses relaxed ordering.
        inline void set_last_ms(uint64_t v) noexcept { last_access_ms.store(v, std::memory_order_relaxed); }
        inline void set_hits(uint32_t v) noexcept { hits.store(v, std::memory_order_relaxed); }
    };

    // Static assertions: size and alignment expectations
    static_assert(sizeof(PPEEntry) == 16, "PPEEntry must be 16 bytes");
    static_assert(std::is_trivially_copyable<PPEEntry>::value, "PPEEntry must be trivially copyable");

    // Thin wrapper over a contiguous PPEEntry buffer (non-owning).
    class PPEArray
    {
    public:
        // Compute required bytes for count entries (useful for mmap sizing)
        static inline size_t required_bytes(size_t count) noexcept { return count * sizeof(PPEEntry); }

        // Initialize memory region with PPEEntry objects using placement-new.
        // - mem must point to writable memory of at least required_bytes(count).
        // - This performs placement-new for each entry to ensure atomics are constructed.
        // - Throws std::bad_alloc on failure (rare).
        static inline void initialize(void *mem, size_t count)
        {
            if (!mem && count > 0)
                throw std::bad_alloc();
            char *ptr = static_cast<char *>(mem);
            for (size_t i = 0; i < count; ++i)
            {
                void *slot = ptr + i * sizeof(PPEEntry);
                // placement-new default ctor
                new (slot) PPEEntry();
            }
        }

        // Create wrapper over existing buffer (non-owning). mem may be mmap'd memory.
        PPEArray(void *mem, size_t count) noexcept : data_(reinterpret_cast<PPEEntry *>(mem)), count_(count) {}

        // Accessors
        size_t size() const noexcept { return count_; }
        PPEEntry *data() const noexcept { return data_; }

        // Safe index check in debug builds
        inline void check_index(size_t idx) const
        {
#ifndef NDEBUG
            if (idx >= count_)
                throw std::out_of_range("PPEArray: index out of range");
#endif
        }

        // Touch entry at idx
        inline void touch(size_t idx) noexcept
        {
            check_index(idx);
            data_[idx].touch();
        }

        // Predict next access ms for idx
        inline uint64_t predictNextMs(size_t idx) const noexcept
        {
            check_index(idx);
            return data_[idx].predictNextMs();
        }

        // Predict bits for idx
        inline uint8_t predictBits(size_t idx, uint8_t default_bits, uint64_t thresh_ms) const noexcept
        {
            check_index(idx);
            return data_[idx].predictBits(default_bits, thresh_ms);
        }

        // Check coldness
        inline bool is_cold_ms(size_t idx, uint64_t threshold_ms) const noexcept
        {
            check_index(idx);
            return data_[idx].is_cold_ms(threshold_ms);
        }

        // Low-level access to the raw PPEEntry (for advanced ops)
        inline PPEEntry &operator[](size_t idx) noexcept
        {
            check_index(idx);
            return data_[idx];
        }
        inline const PPEEntry &operator[](size_t idx) const noexcept
        {
            check_index(idx);
            return data_[idx];
        }

    private:
        PPEEntry *data_;
        size_t count_;
    };

} // namespace pomai::ai