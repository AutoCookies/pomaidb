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

    AppendOnlyArena *AppendOnlyArena::OpenOrCreate(const std::string &path, size_t initial_size_bytes)
    {
        if (initial_size_bytes == 0)
            throw std::invalid_argument("initial_size_bytes must be > 0");

        int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd < 0)
            throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));

        long pg = sysconf(_SC_PAGESIZE);
        size_t page_size = (pg > 0) ? static_cast<size_t>(pg) : 4096;
        size_t target = round_up_to_page(initial_size_bytes, page_size);

        // Try to ensure file is at least target bytes
#ifdef __linux__
        // Prefer posix_fallocate when available
        if (posix_fallocate(fd, 0, static_cast<off_t>(target)) != 0)
        {
            if (ftruncate(fd, static_cast<off_t>(target)) != 0)
            {
                ::close(fd);
                throw std::runtime_error(std::string("preallocate failed: ") + std::strerror(errno));
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

        // Construct AppendOnlyArena and ensure its write_offset_ is initialized inside ctor.
        AppendOnlyArena *a = new AppendOnlyArena(fd, map, target, path, page_size);

        // NOTE: constructor already set write_offset_ to a non-zero page-aligned value.
        // No further store here to avoid any potential race window.

        return a;
    }

    AppendOnlyArena::AppendOnlyArena(int fd, void *map_base, size_t mapped_size, const std::string &path, size_t page_size)
        : fd_(fd),
          map_base_(map_base),
          mapped_size_(mapped_size),
          // Initialize write_offset_ to a sane non-zero page-aligned value immediately.
          // This ensures alloc_blob() will never return offset==0 and avoids the race window.
          write_offset_(static_cast<uint64_t>(page_size)),
          path_(path),
          page_size_(page_size)
    {
        // Zero the initial page so that first blob allocations start from a deterministic, non-zero area.
        if (map_base_ && mapped_size_ >= page_size_)
            std::memset(reinterpret_cast<char *>(map_base_), 0, page_size_);
    }

    AppendOnlyArena::~AppendOnlyArena()
    {
        if (map_base_ && mapped_size_ > 0)
            ::munmap(map_base_, mapped_size_);
        if (fd_ >= 0)
            ::close(fd_);
        map_base_ = nullptr;
        fd_ = -1;
        mapped_size_ = 0;
    }

    void *AppendOnlyArena::alloc_blob(uint32_t payload_size)
    {
        // layout per-blob: uint32_t len + payload bytes
        const uint32_t hdr_len = sizeof(uint32_t);
        size_t total = static_cast<size_t>(hdr_len) + static_cast<size_t>(payload_size);

        // Align allocations to 8 bytes to make subsequent atomic uint64_t stores safe on payload location
        const size_t ALIGN = 8;
        size_t total_aligned = ((total + ALIGN - 1) / ALIGN) * ALIGN;

        std::lock_guard<std::mutex> lk(grow_mu_);

        // ensure capacity
        uint64_t cur_offset = static_cast<uint64_t>(write_offset_.load(std::memory_order_relaxed));
        if (cur_offset + total_aligned > mapped_size_)
        {
            // grow at least double or to fit requested
            size_t new_size = std::max(mapped_size_ * 2, round_up_to_page(cur_offset + total_aligned, page_size_));
            if (new_size < mapped_size_ + page_size_)
                new_size = mapped_size_ + page_size_;
                
            // [FIXED] Retry logic if disk full
            if (!remap_locked(new_size)) {
                // First retry: try smaller increment (just enough for this alloc)
                size_t min_needed = round_up_to_page(cur_offset + total_aligned + page_size_, page_size_);
                if (min_needed < new_size) {
                    if (!remap_locked(min_needed)) {
                        throw std::runtime_error("AppendOnlyArena::alloc_blob: disk full (grow failed)");
                    }
                } else {
                    throw std::runtime_error("AppendOnlyArena::alloc_blob: disk full (grow failed)");
                }
            }
        }

        // allocate
        uint64_t my_off = static_cast<uint64_t>(write_offset_.fetch_add(static_cast<uint64_t>(total_aligned), std::memory_order_acq_rel));
        // sanity
        if (my_off + total_aligned > mapped_size_)
        {
            // rare: race or remap; this shouldn't happen with grow_mu_ held
            throw std::runtime_error("AppendOnlyArena::alloc_blob: allocation overflow after grow");
        }

        char *ptr = reinterpret_cast<char *>(map_base_) + my_off;
        // zero header (len) and payload area to give deterministic contents (helps recovery/reads)
        std::memset(ptr, 0, total_aligned);

        // write payload length (little-endian)
        uint32_t *lenp = reinterpret_cast<uint32_t *>(ptr);
        *lenp = payload_size; // caller will write payload bytes to ptr + hdr_len

        return static_cast<void *>(ptr);
    }

    uint64_t AppendOnlyArena::offset_from_blob_ptr(const void *blob_hdr) const noexcept
    {
        if (!blob_hdr || !map_base_)
            return UINT64_MAX;
        intptr_t diff = reinterpret_cast<const char *>(blob_hdr) - reinterpret_cast<const char *>(map_base_);
        if (diff < 0 || static_cast<size_t>(diff) >= mapped_size_)
            return UINT64_MAX;
        return static_cast<uint64_t>(diff);
    }

    const char *AppendOnlyArena::blob_ptr_from_offset_for_map(uint64_t offset) const noexcept
    {
        if (!map_base_)
            return nullptr;
        if (offset + sizeof(uint32_t) > mapped_size_)
            return nullptr;
        // we can check whether offset < write_offset_, roughly
        uint64_t wo = write_offset_.load(std::memory_order_acquire);
        if (offset >= wo)
            return nullptr; // not yet written/published
        return reinterpret_cast<const char *>(map_base_) + static_cast<size_t>(offset);
    }

    bool AppendOnlyArena::persist_range(uint64_t offset, size_t len, bool synchronous) const noexcept
    {
        if (!map_base_ || offset + len > mapped_size_)
            return false;
        int flags = MS_SYNC;
        if (!synchronous)
            flags = MS_ASYNC;
        if (msync(reinterpret_cast<char *>(map_base_) + offset, len, flags) != 0)
        {
            return false;
        }
        return true;
    }

    bool AppendOnlyArena::remap_locked(size_t new_size)
    {
        // grow file to new_size and remap map_base_. Caller must hold grow_mu_.
        if (new_size <= mapped_size_)
            return true;

#ifdef __linux__
        if (posix_fallocate(fd_, 0, static_cast<off_t>(new_size)) != 0)
        {
            if (ftruncate(fd_, static_cast<off_t>(new_size)) != 0)
                return false; // [FIX] Graceful fail without unmap
        }
#else
        if (ftruncate(fd_, static_cast<off_t>(new_size)) != 0)
            return false; // [FIX] Graceful fail without unmap
#endif

#ifdef __linux__
        // [FIX] Try MREMAP first (fastest & safest)
        void *new_addr = mremap(map_base_, mapped_size_, new_size, MREMAP_MAYMOVE);
        if (new_addr != MAP_FAILED) {
            map_base_ = new_addr;
            mapped_size_ = new_size;
            return true;
        }
        // If mremap fails, fall back to mmap
#endif

        // [FIX] Traditional mmap fallback: MAP NEW before UNMAP OLD
        // This prevents data loss if new mmap fails (e.g. out of V-Memory)
        void *newmap = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (newmap == MAP_FAILED)
        {
            // Revert file size if possible (best effort)
            ftruncate(fd_, static_cast<off_t>(mapped_size_));
            return false;
        }

        // Only unmap old after new is successful
        if (map_base_ && mapped_size_ > 0)
        {
            ::munmap(map_base_, mapped_size_);
        }

        map_base_ = newmap;
        mapped_size_ = new_size;
        return true;
    }

    bool AppendOnlyArena::grow_to(size_t new_size)
    {
        std::lock_guard<std::mutex> lk(grow_mu_);
        return remap_locked(new_size);
    }

} // namespace pomai::memory