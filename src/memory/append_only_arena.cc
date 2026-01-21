#include "src/memory/append_only_arena.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace pomai::memory
{
    static inline size_t round_up_to_page(size_t v, size_t page) { return ((v + page - 1) / page) * page; }
    static inline size_t align_down(size_t v, size_t page) { return (v / page) * page; }
    static inline size_t align_up(size_t v, size_t page) { return ((v + page - 1) / page) * page; }

    AppendOnlyArena *AppendOnlyArena::OpenOrCreate(const std::string &path,
                                                   size_t initial_size_bytes,
                                                   const pomai::config::StorageConfig &cfg)
    {
        int fd = ::open(path.c_str(), O_RDWR | O_CREAT, static_cast<int>(cfg.default_file_permissions));
        if (fd < 0) throw std::runtime_error("open failed");

        long pg = sysconf(_SC_PAGESIZE);
        size_t page_size = (pg > 0) ? static_cast<size_t>(pg) : 4096;
        size_t target = round_up_to_page(initial_size_bytes, page_size);

        if (ftruncate(fd, static_cast<off_t>(target)) != 0) {
            ::close(fd);
            throw std::runtime_error("ftruncate failed");
        }

        void *map = mmap(nullptr, target, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("mmap failed");
        }
        
        madvise(map, target, MADV_SEQUENTIAL);

        return new AppendOnlyArena(fd, map, target, path, page_size, cfg);
    }

    AppendOnlyArena::AppendOnlyArena(int fd, void *map_base, size_t mapped_size,
                                     const std::string &path, size_t page_size,
                                     const pomai::config::StorageConfig &cfg)
        : fd_(fd), map_base_(map_base), mapped_size_(mapped_size),
          write_offset_(0), path_(path), page_size_(page_size), cfg_(cfg) {}

    AppendOnlyArena::~AppendOnlyArena()
    {
        std::unique_lock<std::shared_mutex> lk(map_rw_mu_);
        if (map_base_) munmap(map_base_, mapped_size_);
        if (fd_ >= 0) close(fd_);
    }

    void *AppendOnlyArena::alloc_blob(uint32_t payload_size)
    {
        size_t total = sizeof(uint32_t) + payload_size;
        size_t total_aligned = (total + cfg_.alignment - 1) & ~(cfg_.alignment - 1);

        std::lock_guard<std::mutex> lk(grow_mu_);

        uint64_t cur_offset = write_offset_.load(std::memory_order_relaxed);
        if (cur_offset + total_aligned > mapped_size_) {
            size_t new_size = std::max(static_cast<size_t>(mapped_size_ * cfg_.growth_factor),
                                       round_up_to_page(cur_offset + total_aligned, page_size_));
            if (!remap_locked(new_size)) throw std::runtime_error("grow failed");
        }

        uint64_t my_off = write_offset_.fetch_add(total_aligned, std::memory_order_acq_rel);
        
        std::shared_lock<std::shared_mutex> read_lk(map_rw_mu_);
        char *ptr = reinterpret_cast<char *>(map_base_) + my_off;
        std::memset(ptr, 0, total_aligned);
        *reinterpret_cast<uint32_t *>(ptr) = payload_size;

        return ptr;
    }

    const char *AppendOnlyArena::blob_ptr_from_offset_for_map(uint64_t offset) const noexcept
    {
        std::shared_lock<std::shared_mutex> lk(map_rw_mu_);
        if (!map_base_ || offset + sizeof(uint32_t) > mapped_size_ || offset >= write_offset_.load(std::memory_order_acquire))
            return nullptr;
        return reinterpret_cast<const char *>(map_base_) + offset;
    }

    uint64_t AppendOnlyArena::offset_from_blob_ptr(const void *blob_hdr) const noexcept
    {
        std::shared_lock<std::shared_mutex> lk(map_rw_mu_);
        if (!blob_hdr || !map_base_) return UINT64_MAX;
        intptr_t diff = reinterpret_cast<const char *>(blob_hdr) - reinterpret_cast<const char *>(map_base_);
        if (diff < 0 || static_cast<size_t>(diff) >= mapped_size_) return UINT64_MAX;
        return static_cast<uint64_t>(diff);
    }

    bool AppendOnlyArena::remap_locked(size_t new_size)
    {
        if (new_size <= mapped_size_) return true;

        if (ftruncate(fd_, static_cast<off_t>(new_size)) != 0) return false;

        std::unique_lock<std::shared_mutex> lk(map_rw_mu_);
#ifdef __linux__
        void *new_map = mremap(map_base_, mapped_size_, new_size, MREMAP_MAYMOVE);
        if (new_map == MAP_FAILED) return false;
        map_base_ = new_map;
        mapped_size_ = new_size;
#else
        void *new_map = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (new_map == MAP_FAILED) return false;
        if (map_base_) munmap(map_base_, mapped_size_);
        map_base_ = new_map;
        mapped_size_ = new_size;
#endif
        madvise(map_base_, mapped_size_, MADV_SEQUENTIAL);
        return true;
    }

    size_t AppendOnlyArena::get_capacity_bytes() const noexcept
    {
        std::shared_lock<std::shared_mutex> lk(map_rw_mu_);
        return mapped_size_;
    }

    bool AppendOnlyArena::grow_to(size_t new_size)
    {
        std::lock_guard<std::mutex> lk(grow_mu_);
        return remap_locked(new_size);
    }

    bool AppendOnlyArena::persist_range(uint64_t offset, size_t len, bool synchronous) const noexcept
    {
        std::shared_lock<std::shared_mutex> lk(map_rw_mu_);
        if (!map_base_ || offset + len > mapped_size_) return false;
        size_t start = align_down(offset, page_size_);
        size_t end = align_up(offset + len, page_size_);
        return msync(reinterpret_cast<char *>(map_base_) + start, end - start, synchronous ? MS_SYNC : MS_ASYNC) == 0;
    }
}