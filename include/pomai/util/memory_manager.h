#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace pomai
{

    class MemoryManager
    {
    public:
        enum class Pool : std::size_t
        {
            Wal = 0,
            Search = 1,
            Indexing = 2,
            Memtable = 3,
            Count = 4
        };

        static MemoryManager &Instance();

        void AddUsage(Pool pool, std::size_t bytes);
        void ReleaseUsage(Pool pool, std::size_t bytes);

        std::size_t Usage(Pool pool) const;
        std::size_t TotalUsage() const;

        bool CanAllocate(std::size_t bytes) const;
        bool AtOrAboveSoftWatermark() const;
        std::size_t SoftWatermarkBytes() const;
        std::size_t HardWatermarkBytes() const;

        // Testing helpers (no-op for production callers).
        void SetTotalMemoryBytesForTesting(std::size_t bytes);
        void ResetUsageForTesting();

    private:
        MemoryManager();

        std::atomic<std::int64_t> total_usage_{0};
        std::array<std::atomic<std::int64_t>, static_cast<std::size_t>(Pool::Count)> usage_{};
        std::atomic<std::int64_t> total_bytes_{0};
    };

} // namespace pomai
