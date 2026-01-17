// src/memory/shard_arena.cc
//
// Implementation of ShardArena (see header).
// - Uses mmap(MAP_ANONYMOUS) and attempts MAP_HUGETLB if available.
// - Atomic bump allocator with 64-byte alignment for cache friendliness.
// - Lazy mmap and caching for remote files; remote ids encoded with MSB set.
// - [FIXED] Added Cache Eviction (Cap 100 files) to prevent OOM.
//
// Build notes: compile with -std=c++20 -O3 -march=native for best performance.

#include "src/memory/shard_arena.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <cstring>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <unistd.h>

namespace pomai::memory
{

    static inline uint64_t make_encoded_remote_id(uint64_t id) noexcept
    {
        return id | (1ULL << 63);
    }

    static inline bool is_encoded_remote_id(uint64_t v) noexcept
    {
        return (v >> 63) != 0;
    }

    ShardArena::ShardArena(uint32_t shard_id, size_t capacity_bytes, const pomai::config::PomaiConfig& cfg)
        : id_(shard_id),
          capacity_(capacity_bytes),
          base_addr_(nullptr),
          write_head_(0),
          remote_dir_(cfg.arena.remote_dir.empty() ? "/tmp" : cfg.arena.remote_dir),
          max_remote_mmaps_(cfg.arena.max_remote_mmaps), 
          next_remote_id_(1),
          page_size_(static_cast<size_t>(sysconf(_SC_PAGESIZE)))
    {
        if (capacity_ == 0)
            throw std::invalid_argument("ShardArena: capacity must be > 0");

        // Round capacity to page size
        size_t map_size = ((capacity_ + page_size_ - 1) / page_size_) * page_size_;

        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_HUGETLB
        // Try huge pages first (may fail gracefully)
        void *p = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, flags | MAP_HUGETLB, -1, 0);
        if (p == MAP_FAILED)
        {
            // Fallback to plain mmap without huge pages
            p = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, flags, -1, 0);
        }
#else
        void *p = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, flags, -1, 0);
#endif

        if (p == MAP_FAILED)
        {
            int err = errno;
            std::ostringstream ss;
            ss << "ShardArena: mmap failed for shard " << id_ << " : " << std::strerror(err);
            throw std::runtime_error(ss.str());
        }

        base_addr_ = reinterpret_cast<char *>(p);

        // Initialize write head to a small non-zero reserved offset (avoid returning 0)
        const uint64_t reserved = 64; // reserve first 64 bytes
        write_head_.store(reserved, std::memory_order_relaxed);

        // Ensure remote dir exists
        try
        {
            std::filesystem::create_directories(remote_dir_);
        }
        catch (...)
        {
            // ignore: remote ops will fail later if dir missing
        }
    }

    ShardArena::~ShardArena()
    {
        // Unmap cached remote mmaps
        {
            std::lock_guard<std::mutex> lk(remote_mu_);
            for (auto &ent : remote_mmaps_)
            {
                const char *addr = ent.second.first;
                size_t sz = ent.second.second;
                if (addr && sz > 0)
                {
                    munmap(const_cast<char *>(addr), sz);
                }
            }
            remote_mmaps_.clear();
        }

        // Unmap base region
        if (base_addr_ && capacity_ > 0)
        {
            munmap(base_addr_, capacity_);
            base_addr_ = nullptr;
        }
    }

    char *ShardArena::alloc_blob(uint32_t len)
    {
        if (!base_addr_)
            return nullptr;

        const uint32_t hdr = sizeof(uint32_t);
        size_t total = static_cast<size_t>(hdr) + static_cast<size_t>(len) + 1; // include terminating zero
        // Align to 64 bytes for cache efficiency
        const size_t ALIGN = 64;
        size_t aligned = align_up(total, ALIGN);

        uint64_t my_offset = write_head_.fetch_add(static_cast<uint64_t>(aligned), std::memory_order_acq_rel);

        // bounds check
        if (my_offset + aligned > capacity_)
        {
            // undo? can't easily subtract from atomic; we simply return nullptr to indicate OOM.
            return nullptr;
        }

        char *ptr = base_addr_ + static_cast<size_t>(my_offset);
        // store length (little-endian native)
        *reinterpret_cast<uint32_t *>(ptr) = len;
        // zero terminator (safe to write)
        ptr[hdr + len] = '\0';
        // It's the caller's responsibility to write payload bytes to ptr + hdr
        return ptr;
    }

    uint64_t ShardArena::offset_from_blob_ptr(const char *ptr) const noexcept
    {
        if (!ptr || !base_addr_)
            return UINT64_MAX;
        intptr_t diff = reinterpret_cast<const char *>(ptr) - base_addr_;
        if (diff <= 0 || static_cast<size_t>(diff) >= capacity_)
            return UINT64_MAX;
        return static_cast<uint64_t>(diff);
    }

    const char *ShardArena::blob_ptr_from_offset_for_map(uint64_t offset) const noexcept
    {
        if (!base_addr_)
            return nullptr;

        // Remote encoded id?
        if (is_encoded_remote_id(offset))
        {
            // Lazy mmap and cache lookup
            std::lock_guard<std::mutex> lk(remote_mu_);
            auto it = remote_mmaps_.find(offset);
            if (it != remote_mmaps_.end())
                return it->second.first;

            // [FIX HIGH] Cap Open Files: Prevent OOM by evicting old maps
            // Giới hạn 100 file mmap đồng thời. Nếu đầy, đóng bớt 1 cái.
            // (Dùng chiến thuật Random Eviction đơn giản: xóa phần tử đầu tiên của map)
            if (remote_mmaps_.size() >= max_remote_mmaps_)
            {
                // Eviction logic
                auto victim = remote_mmaps_.begin();
                const char *v_addr = victim->second.first;
                size_t v_sz = victim->second.second;
                if (v_addr && v_sz > 0) munmap(const_cast<char *>(v_addr), v_sz);
                remote_mmaps_.erase(victim);
            }

            // Map file corresponding to offset
            std::string fname = generate_remote_filename(offset);
            int fd = open(fname.c_str(), O_RDONLY);
            if (fd < 0)
                return nullptr;
            struct stat st;
            if (fstat(fd, &st) != 0)
            {
                close(fd);
                return nullptr;
            }
            size_t sz = static_cast<size_t>(st.st_size);
            if (sz == 0)
            {
                close(fd);
                return nullptr;
            }

            void *mp = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (mp == MAP_FAILED)
                return nullptr;
            const char *addr = reinterpret_cast<const char *>(mp);
            remote_mmaps_.emplace(offset, std::make_pair(addr, sz));
            return addr;
        }

        // Local offset path
        if (offset == 0)
            return nullptr;
        if (offset >= capacity_)
            return nullptr;

        // Ensure offset is within published region: compare to write_head_.
        uint64_t wh = write_head_.load(std::memory_order_acquire);
        if (offset >= wh)
            return nullptr; // not yet published

        return base_addr_ + static_cast<size_t>(offset);
    }

    uint64_t ShardArena::demote_blob(uint64_t local_offset)
    {
        if (!base_addr_)
            return 0;
        if (local_offset == 0 || local_offset >= capacity_)
            return 0;

        const char *p = base_addr_ + static_cast<size_t>(local_offset);
        uint32_t len = *reinterpret_cast<const uint32_t *>(p);
        size_t total = sizeof(uint32_t) + static_cast<size_t>(len) + 1;

        // Reserve a remote id
        uint64_t id = next_remote_id_.fetch_add(1, std::memory_order_relaxed);
        uint64_t encoded = make_encoded_remote_id(id);
        std::string fname = generate_remote_filename(encoded);
        std::string tmp = fname + ".tmp";

        // Write to temp file
        int fd = open(tmp.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (fd < 0)
            return 0;
        const char *src = p;
        size_t left = total;
        while (left > 0)
        {
            ssize_t w = write(fd, src, left);
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                close(fd);
                unlink(tmp.c_str());
                return 0;
            }
            left -= static_cast<size_t>(w);
            src += w;
        }
        // durable write
        fsync(fd);
        close(fd);

        // Atomic rename to final name
        if (rename(tmp.c_str(), fname.c_str()) != 0)
        {
            unlink(tmp.c_str());
            return 0;
        }

        // We don't mmap file now; let blob_ptr_from_offset_for_map lazily map on read.
        return encoded;
    }

    std::vector<char> ShardArena::read_remote_blob(uint64_t remote_id) const
    {
        if (!is_encoded_remote_id(remote_id))
            return {};
        std::string fname = generate_remote_filename(remote_id);
        std::ifstream ifs(fname, std::ios::binary | std::ios::ate);
        if (!ifs)
            return {};
        std::streamsize sz = ifs.tellg();
        if (sz <= 0)
            return {};
        ifs.seekg(0, std::ios::beg);
        std::vector<char> buf(static_cast<size_t>(sz));
        if (!ifs.read(buf.data(), sz))
            return {};
        return buf;
    }

    std::string ShardArena::generate_remote_filename(uint64_t encoded_remote_id) const
    {
        // Strip MSB for filename stability
        uint64_t id = encoded_remote_id & (~(1ULL << 63));
        std::ostringstream ss;
        ss << remote_dir_ << "/shard_" << id_ << "_blob_" << id << ".bin";
        return ss.str();
    }

    void ShardArena::reset() noexcept
    {
        // Danger: this invalidates existing offsets/pointers
        write_head_.store(64, std::memory_order_relaxed);

        // Unmap and clear remote mmaps
        std::lock_guard<std::mutex> lk(remote_mu_);
        for (auto &ent : remote_mmaps_)
        {
            const char *addr = ent.second.first;
            size_t sz = ent.second.second;
            if (addr && sz > 0)
                munmap(const_cast<char *>(addr), sz);
        }
        remote_mmaps_.clear();
        next_remote_id_.store(1);
    }

    // [FIX CRITICAL #1] Madvise Alignment Safe
    void ShardArena::demote_range(uint64_t offset, size_t len)
    {
        const char *ptr = blob_ptr_from_offset_for_map(offset);
        if (!ptr)
            return;

        // 1. Lấy Page Size thực tế
        static const long page_size = sysconf(_SC_PAGESIZE);
        const uintptr_t page_mask = ~(static_cast<uintptr_t>(page_size - 1));

        // 2. Tính toán Alignment an toàn
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned_addr = addr & page_mask; // Round down

        // Tính offset dư ra để cộng vào length
        size_t diff = addr - aligned_addr;
        size_t aligned_len = len + diff;

        // 3. Gọi madvise
        int ret = madvise(reinterpret_cast<void *>(aligned_addr), aligned_len, MADV_DONTNEED);

        if (ret != 0)
        {
            // log error if needed
        }
    }

} // namespace pomai::memory