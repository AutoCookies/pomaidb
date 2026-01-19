// (excerpt) src/memory/append_only_arena.cc
// Only modified parts shown — integrates atomic mapping pointer/size and msync alignment.

#include "src/memory/append_only_arena.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sys/types.h>

#ifndef MREMAP_MAYMOVE
#define MREMAP_MAYMOVE 1
#endif

namespace pomai::memory
{

    static inline size_t round_up_to_page(size_t v, size_t page)
    {
        return ((v + page - 1) / page) * page;
    }

    static inline size_t align_down(size_t v, size_t page) { return (v / page) * page; }
    static inline size_t align_up(size_t v, size_t page) { return ((v + page - 1) / page) * page; }

    AppendOnlyArena *AppendOnlyArena::OpenOrCreate(const std::string &path,
                                                   size_t initial_size_bytes,
                                                   const pomai::config::StorageConfig &cfg)
    {
        if (initial_size_bytes == 0)
            throw std::invalid_argument("initial_size_bytes must be > 0");

        int fd = ::open(path.c_str(), O_RDWR | O_CREAT, static_cast<int>(cfg.default_file_permissions));
        if (fd < 0)
            throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));

        long pg = sysconf(_SC_PAGESIZE);
        size_t page_size = (pg > 0) ? static_cast<size_t>(pg) : 4096;
        size_t target = round_up_to_page(initial_size_bytes, page_size);

#ifdef __linux__
        if (cfg.prefer_fallocate && posix_fallocate(fd, 0, static_cast<off_t>(target)) != 0)
        {
            if (ftruncate(fd, static_cast<off_t>(target)) != 0)
            {
                ::close(fd);
                throw std::runtime_error(std::string("preallocate failed: ") + std::strerror(errno));
            }
        }
        else if (!cfg.prefer_fallocate)
        {
            if (ftruncate(fd, static_cast<off_t>(target)) != 0)
            {
                ::close(fd);
                throw std::runtime_error(std::string("ftruncate failed: ") + std::strerror(errno));
            }
        }
#else
        if (ftruncate(fd, static_cast<off_t>(target)) != 0)
        {
            ::close(fd);
            throw std::runtime_error(std::string("ftruncate failed: ") + std::strerror(errno));
        }
#endif

        void *map = mmap(nullptr, target, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED)
        {
            ::close(fd);
            throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
        }

        return new AppendOnlyArena(fd, map, target, path, page_size, cfg);
    }

    AppendOnlyArena::AppendOnlyArena(int fd, void *map_base, size_t mapped_size,
                                     const std::string &path, size_t page_size,
                                     const pomai::config::StorageConfig &cfg)
        : fd_(fd),
          map_base_(map_base),
          mapped_size_(mapped_size),
          write_offset_(static_cast<uint64_t>(page_size)),
          path_(path),
          page_size_(page_size),
          cfg_(cfg)
    {
        void *mb = map_base_.load(std::memory_order_relaxed);
        size_t ms = mapped_size_.load(std::memory_order_relaxed);
        if (mb && ms >= page_size_)
            std::memset(reinterpret_cast<char *>(mb), 0, page_size_);
    }

    AppendOnlyArena::~AppendOnlyArena()
    {
        void *mb = map_base_.load(std::memory_order_relaxed);
        size_t ms = mapped_size_.load(std::memory_order_relaxed);
        if (mb && ms > 0)
            ::munmap(mb, ms);
        if (fd_ >= 0)
            ::close(fd_);
    }

    void *AppendOnlyArena::alloc_blob(uint32_t payload_size)
    {
        const uint32_t hdr_len = sizeof(uint32_t);
        size_t total = static_cast<size_t>(hdr_len) + static_cast<size_t>(payload_size);

        const size_t ALIGN = cfg_.alignment;
        size_t total_aligned = ((total + ALIGN - 1) / ALIGN) * ALIGN;

        std::lock_guard<std::mutex> lk(grow_mu_);

        uint64_t cur_offset = static_cast<uint64_t>(write_offset_.load(std::memory_order_relaxed));
        size_t mapped = mapped_size_.load(std::memory_order_acquire);
        if (cur_offset + total_aligned > mapped)
        {
            size_t desired_by_factor = static_cast<size_t>(static_cast<double>(mapped) * static_cast<double>(cfg_.growth_factor));
            size_t desired_by_need = round_up_to_page(cur_offset + total_aligned, page_size_);
            size_t new_size = std::max(desired_by_factor, desired_by_need);

            if (!remap_locked(new_size))
            {
                size_t min_needed = round_up_to_page(cur_offset + total_aligned + page_size_, page_size_);
                if (!remap_locked(min_needed))
                {
                    throw std::runtime_error("AppendOnlyArena::alloc_blob: disk full (grow failed)");
                }
            }
            mapped = mapped_size_.load(std::memory_order_acquire);
        }

        uint64_t my_off = static_cast<uint64_t>(write_offset_.fetch_add(static_cast<uint64_t>(total_aligned), std::memory_order_acq_rel));
        if (my_off + total_aligned > mapped)
        {
            throw std::runtime_error("AppendOnlyArena::alloc_blob: allocation overflow after grow");
        }

        void *mb = map_base_.load(std::memory_order_acquire);
        char *ptr = reinterpret_cast<char *>(mb) + static_cast<size_t>(my_off);
        std::memset(ptr, 0, total_aligned);

        uint32_t *lenp = reinterpret_cast<uint32_t *>(ptr);
        *lenp = payload_size;

        return static_cast<void *>(ptr);
    }

    uint64_t AppendOnlyArena::offset_from_blob_ptr(const void *blob_hdr) const noexcept
    {
        if (!blob_hdr)
            return UINT64_MAX;
        void *mb = map_base_.load(std::memory_order_acquire);
        if (!mb)
            return UINT64_MAX;
        intptr_t diff = reinterpret_cast<const char *>(blob_hdr) - reinterpret_cast<const char *>(mb);
        size_t mapped = mapped_size_.load(std::memory_order_acquire);
        if (diff < 0 || static_cast<size_t>(diff) >= mapped)
            return UINT64_MAX;
        return static_cast<uint64_t>(diff);
    }

    const char *AppendOnlyArena::blob_ptr_from_offset_for_map(uint64_t offset) const noexcept
    {
        void *mb = map_base_.load(std::memory_order_acquire);
        if (!mb)
            return nullptr;
        size_t mapped = mapped_size_.load(std::memory_order_acquire);
        if (offset + sizeof(uint32_t) > mapped)
            return nullptr;
        uint64_t wo = write_offset_.load(std::memory_order_acquire);
        if (offset >= wo)
            return nullptr;
        return reinterpret_cast<const char *>(mb) + static_cast<size_t>(offset);
    }

    bool AppendOnlyArena::persist_range(uint64_t offset, size_t len, bool synchronous) const noexcept
    {
        void *mb = map_base_.load(std::memory_order_acquire);
        if (!mb)
            return false;
        size_t mapped = mapped_size_.load(std::memory_order_acquire);
        if (offset + len > mapped)
            return false;

        // align to page boundaries
        size_t page = page_size_;
        size_t start = align_down(offset, page);
        size_t end = align_up(offset + len, page);
        size_t ms = end - start;
        int flags = synchronous ? MS_SYNC : MS_ASYNC;
        int rc = msync(reinterpret_cast<char *>(mb) + start, ms, flags);
        return rc == 0;
    }

    bool AppendOnlyArena::remap_locked(size_t new_size)
    {
        size_t cur_mapped = mapped_size_.load(std::memory_order_acquire);
        if (new_size <= cur_mapped)
            return true;

#ifdef __linux__
        if (cfg_.prefer_fallocate && posix_fallocate(fd_, 0, static_cast<off_t>(new_size)) != 0)
        {
            if (ftruncate(fd_, static_cast<off_t>(new_size)) != 0)
                return false;
        }
        else if (!cfg_.prefer_fallocate)
        {
            if (ftruncate(fd_, static_cast<off_t>(new_size)) != 0)
                return false;
        }
#else
        if (ftruncate(fd_, static_cast<off_t>(new_size)) != 0)
            return false;
#endif

#ifdef __linux__
        void *old_base = map_base_.load(std::memory_order_acquire);
        size_t old_size = mapped_size_.load(std::memory_order_acquire);

        // Try mremap in-place first (preferred)
        if (old_base)
        {
            void *maybe = mremap(old_base, old_size, new_size, MREMAP_MAYMOVE);
            if (maybe != MAP_FAILED)
            {
                map_base_.store(maybe, std::memory_order_release);
                mapped_size_.store(new_size, std::memory_order_release);
                return true;
            }
        }
#endif

        // Fallback: mmap new mapping (may return different address)
        void *newmap = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (newmap == MAP_FAILED)
        {
            // try to rollback file size
            if (ftruncate(fd_, static_cast<off_t>(cur_mapped)) != 0)
            {
                // ignore
            }
            return false;
        }

        // Atomically publish the new mapping pointer/size so readers see consistent mapping
        void *oldmap = map_base_.exchange(newmap, std::memory_order_acq_rel);
        mapped_size_.store(new_size, std::memory_order_release);

        // Now it's safe to unmap the old mapping (if any)
        if (oldmap && oldmap != newmap && cur_mapped > 0)
        {
            ::munmap(oldmap, cur_mapped);
        }

        return true;
    }

    bool AppendOnlyArena::grow_to(size_t new_size)
    {
        std::lock_guard<std::mutex> lk(grow_mu_);
        return remap_locked(new_size);
    }

} // namespace pomai::memory