#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <shared_mutex>
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
        size_t get_capacity_bytes() const noexcept;
        bool grow_to(size_t new_size);
        std::string path() const noexcept { return path_; }

    private:
        AppendOnlyArena(int fd, void *map_base, size_t mapped_size,
                        const std::string &path, size_t page_size,
                        const pomai::config::StorageConfig &cfg);

        bool remap_locked(size_t new_size);

        int fd_;
        void* map_base_;
        size_t mapped_size_;
        std::atomic<uint64_t> write_offset_;
        mutable std::mutex grow_mu_;
        mutable std::shared_mutex map_rw_mu_;
        std::string path_;
        size_t page_size_;
        pomai::config::StorageConfig cfg_;
    };
}