#pragma once
// src/memory/append_only_arena.h
//
// Thread-safe append-only arena backed by a file + mmap.
// (unchanged API)

#include <string>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include "src/core/config.h"

namespace pomai::memory
{

    class AppendOnlyArena
    {
    public:
        static AppendOnlyArena *OpenOrCreate(const std::string &path,
                                             size_t initial_size_bytes,
                                             const pomai::config::StorageConfig &cfg);

        ~AppendOnlyArena();

        AppendOnlyArena(const AppendOnlyArena &) = delete;
        AppendOnlyArena &operator=(const AppendOnlyArena &) = delete;

        void *alloc_blob(uint32_t payload_size);
        uint64_t offset_from_blob_ptr(const void *blob_hdr) const noexcept;
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const noexcept;
        bool persist_range(uint64_t offset, size_t len, bool synchronous = false) const noexcept;
        size_t get_capacity_bytes() const noexcept { return mapped_size_.load(std::memory_order_acquire); }
        bool grow_to(size_t new_size);

        std::string path() const noexcept { return path_; }

    private:
        AppendOnlyArena(int fd, void *map_base, size_t mapped_size,
                        const std::string &path, size_t page_size,
                        const pomai::config::StorageConfig &cfg);

        bool remap_locked(size_t new_size);

        int fd_;
        // make pointer/size atomic so readers can safely see a consistent mapping
        std::atomic<void *> map_base_;
        std::atomic<size_t> mapped_size_;
        std::atomic<uint64_t> write_offset_; // next free offset inside mapping
        mutable std::mutex grow_mu_;         // protects grow/remap and alloc path when remapping
        std::string path_;
        size_t page_size_;
        pomai::config::StorageConfig cfg_;
    };

} // namespace pomai::memory