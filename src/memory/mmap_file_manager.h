/*
 * src/memory/mmap_file_manager.h
 *
 * MmapFileManager
 *
 * Small utility to manage a file-backed memory mapping with:
 *  - preallocation of file bytes (posix_fallocate or ftruncate fallback)
 *  - page-aligned mmap of the requested region
 *  - thread-safe read/write/append wrappers (coarse-grained mutex)
 *  - optional msync/flush and madvise helpers
 *  - optional mlock/munlock to pin/unpin hot pages
 *  - atomic helpers for 64-bit loads/stores into the mapping (uses atomic_utils)
 *
 * See mmap_file_manager.cc for implementation details.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <mutex>
#include <atomic>
#include "src/core/config.h"

namespace pomai::memory
{

    class MmapFileManager
    {
    public:
        MmapFileManager() noexcept;
        MmapFileManager(const std::string &path, size_t size_bytes, 
                        const pomai::config::PomaiConfig& cfg, bool create = true);
        ~MmapFileManager();

        MmapFileManager(const MmapFileManager &) = delete;
        MmapFileManager &operator=(const MmapFileManager &) = delete;

        MmapFileManager(MmapFileManager &&other) noexcept;
        MmapFileManager &operator=(MmapFileManager &&other) noexcept;

        bool open(const std::string &path, size_t size_bytes, 
                  const pomai::config::PomaiConfig& cfg, bool create = true);
        void close();
        bool is_valid() const noexcept;
        int fd() const noexcept;
        size_t mapped_size() const noexcept;

        bool read_at(size_t offset, void *dst, size_t len) const;
        bool write_at(size_t offset, const void *src, size_t len);
        size_t append(const void *src, size_t len);
        bool flush(size_t offset = 0, size_t len = 0, bool sync = false);
        bool advise_willneed(size_t offset, size_t len);

        enum class AdviseMode
        {
            NORMAL,
            SEQUENTIAL,
            RANDOM
        };
        bool advise_mode(size_t offset, size_t len, AdviseMode mode);

        bool mlock_range(size_t offset, size_t len);
        bool munlock_range(size_t offset, size_t len);

        const char *base_ptr() const noexcept;
        char *writable_base_ptr() noexcept;

        // ATOMIC helpers -------------------------------------------------------
        // Atomically load a uint64_t value located at byte 'offset' within the mapping.
        // Returns true on success and writes loaded value to 'out'.
        bool atomic_load_u64_at(size_t offset, uint64_t &out) const;

        // Atomically store a uint64_t value located at byte 'offset' within the mapping.
        // Returns true on success.
        bool atomic_store_u64_at(size_t offset, uint64_t value);

    private:
        bool ensure_mapped(size_t size_bytes);
        void unmap_and_close();

        mutable std::mutex io_mu_;
        int fd_;
        char *base_addr_;
        size_t mapped_size_;
        std::string path_;
        std::atomic<size_t> append_offset_;
        pomai::config::PomaiConfig cfg_;
    };

} // namespace pomai::memory