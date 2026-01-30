#include <pomai/util/memory_manager.h>

#if defined(__has_include)
#if __has_include(<unistd.h>)
#include <unistd.h>
#define POMAI_HAS_UNISTD 1
#endif
#endif

#if defined(_WIN32)
#undef POMAI_HAS_UNISTD
#endif

namespace pomai
{
    namespace
    {
        constexpr double kSoftWatermarkRatio = 0.70;
        constexpr double kHardWatermarkRatio = 0.90;

        std::size_t DetectTotalMemoryBytes()
        {
#if defined(POMAI_HAS_UNISTD) && defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
            long pages = ::sysconf(_SC_PHYS_PAGES);
            long page_size = ::sysconf(_SC_PAGE_SIZE);
            if (pages <= 0 || page_size <= 0)
                return 8ULL * 1024ULL * 1024ULL * 1024ULL;
            return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
#else
            return 8ULL * 1024ULL * 1024ULL * 1024ULL;
#endif
        }

        std::size_t ClampToSize(std::int64_t value)
        {
            if (value < 0)
                return 0;
            return static_cast<std::size_t>(value);
        }
    } // namespace

    MemoryManager &MemoryManager::Instance()
    {
        static MemoryManager instance;
        return instance;
    }

    MemoryManager::MemoryManager()
    {
        total_bytes_.store(static_cast<std::int64_t>(DetectTotalMemoryBytes()), std::memory_order_relaxed);
        for (auto &u : usage_)
            u.store(0, std::memory_order_relaxed);
    }

    void MemoryManager::AddUsage(Pool pool, std::size_t bytes)
    {
        if (bytes == 0)
            return;
        usage_[static_cast<std::size_t>(pool)].fetch_add(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
        total_usage_.fetch_add(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
    }

    void MemoryManager::ReleaseUsage(Pool pool, std::size_t bytes)
    {
        if (bytes == 0)
            return;
        usage_[static_cast<std::size_t>(pool)].fetch_sub(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
        total_usage_.fetch_sub(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
    }

    std::size_t MemoryManager::Usage(Pool pool) const
    {
        return ClampToSize(usage_[static_cast<std::size_t>(pool)].load(std::memory_order_relaxed));
    }

    std::size_t MemoryManager::TotalUsage() const
    {
        return ClampToSize(total_usage_.load(std::memory_order_relaxed));
    }

    std::size_t MemoryManager::SoftWatermarkBytes() const
    {
        auto total = ClampToSize(total_bytes_.load(std::memory_order_relaxed));
        return static_cast<std::size_t>(static_cast<double>(total) * kSoftWatermarkRatio);
    }

    std::size_t MemoryManager::HardWatermarkBytes() const
    {
        auto total = ClampToSize(total_bytes_.load(std::memory_order_relaxed));
        return static_cast<std::size_t>(static_cast<double>(total) * kHardWatermarkRatio);
    }

    bool MemoryManager::CanAllocate(std::size_t bytes) const
    {
        if (bytes == 0)
            return true;
        const std::size_t total = TotalUsage();
        return total + bytes <= HardWatermarkBytes();
    }

    bool MemoryManager::AtOrAboveSoftWatermark() const
    {
        return TotalUsage() >= SoftWatermarkBytes();
    }

    void MemoryManager::SetTotalMemoryBytesForTesting(std::size_t bytes)
    {
        total_bytes_.store(static_cast<std::int64_t>(bytes), std::memory_order_relaxed);
    }

    void MemoryManager::ResetUsageForTesting()
    {
        total_usage_.store(0, std::memory_order_relaxed);
        for (auto &u : usage_)
            u.store(0, std::memory_order_relaxed);
    }

} // namespace pomai
