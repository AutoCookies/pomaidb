// src/memory/shard_arena.cc
//
// File-backed ShardArena implementation using mmap(MAP_SHARED).
// See header for behavior notes.

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
#include <algorithm>

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

    static inline size_t page_round_up(size_t v, size_t page)
    {
        return ((v + page - 1) / page) * page;
    }

    ShardArena::ShardArena(uint32_t shard_id, size_t capacity_bytes, const pomai::config::PomaiConfig& cfg)
        : id_(shard_id),
          capacity_(0),
          base_addr_(nullptr),
          fd_(-1),
          write_head_(0),
          remote_dir_(cfg.arena.remote_dir.empty() ? "/tmp" : cfg.arena.remote_dir),
          max_remote_mmaps_(cfg.arena.max_remote_mmaps),
          next_remote_id_(1),
          page_size_(static_cast<size_t>(sysconf(_SC_PAGESIZE)))
    {
        if (capacity_bytes == 0)
            throw std::invalid_argument("ShardArena: capacity must be > 0");

        // Ensure remote dir exists
        try { std::filesystem::create_directories(remote_dir_); } catch (...) {}

        // Determine backing file path: prefer config.res.data_root if present else remote_dir_
        std::string bfile = backing_filename();

        // Open or create file
        fd_ = ::open(bfile.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0)
        {
            int err = errno;
            std::ostringstream ss;
            ss << "ShardArena: open backing file failed: " << bfile << " : " << std::strerror(err);
            throw std::runtime_error(ss.str());
        }

        // Round up size to page boundaries
        size_t map_size = page_round_up(capacity_bytes, page_size_);

        // Try to allocate space (posix_fallocate if supported)
#ifdef __linux__
        if (cfg.storage.prefer_fallocate)
        {
            if (posix_fallocate(fd_, 0, static_cast<off_t>(map_size)) != 0)
            {
                // fallback to ftruncate
                if (ftruncate(fd_, static_cast<off_t>(map_size)) != 0)
                {
                    ::close(fd_);
                    fd_ = -1;
                    throw std::runtime_error("ShardArena: failed to allocate backing file size");
                }
            }
        }
        else
        {
            if (ftruncate(fd_, static_cast<off_t>(map_size)) != 0)
            {
                ::close(fd_);
                fd_ = -1;
                throw std::runtime_error("ShardArena: failed to truncate backing file");
            }
        }
#else
        if (ftruncate(fd_, static_cast<off_t>(map_size)) != 0)
        {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("ShardArena: failed to truncate backing file");
        }
#endif

        // mmap with MAP_SHARED so pages are backed by file and persistable
        void *p = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED)
        {
            int err = errno;
            ::close(fd_);
            fd_ = -1;
            std::ostringstream ss;
            ss << "ShardArena: mmap failed for file " << bfile << " : " << std::strerror(err);
            throw std::runtime_error(ss.str());
        }

        base_addr_ = reinterpret_cast<char *>(p);
        capacity_ = map_size;

        // Reserve small non-zero offset
        const uint64_t reserved = 64;
        write_head_.store(reserved, std::memory_order_relaxed);
    }

    ShardArena::~ShardArena()
    {
        // Unmap remote mmaps
        {
            std::lock_guard<std::mutex> lk(remote_mu_);
            for (auto &ent : remote_mmaps_)
            {
                const char *addr = ent.second.first;
                size_t sz = ent.second.second;
                if (addr && sz > 0)
                    munmap(const_cast<char *>(addr), sz);
            }
            remote_mmaps_.clear();
        }

        // Persist any dirty pages and unmap base region
        if (base_addr_ && capacity_ > 0)
        {
            // Best-effort msync entire mapping
            msync(base_addr_, capacity_, MS_SYNC);
            munmap(base_addr_, capacity_);
            base_addr_ = nullptr;
        }

        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

    char *ShardArena::alloc_blob(uint32_t len)
    {
        if (!base_addr_ || fd_ < 0)
            return nullptr;

        const uint32_t hdr = sizeof(uint32_t);
        size_t total = static_cast<size_t>(hdr) + static_cast<size_t>(len) + 1; // include terminating zero
        const size_t ALIGN = 64;
        size_t aligned = align_up(total, ALIGN);

        uint64_t my_offset = write_head_.fetch_add(static_cast<uint64_t>(aligned), std::memory_order_acq_rel);

        // bounds check
        if (my_offset + aligned > capacity_)
        {
            // attempt to grow if allowed (best-effort)
            size_t need = static_cast<size_t>(my_offset + aligned);
            size_t new_size = std::max(need, capacity_ * 2);
            new_size = page_round_up(new_size, page_size_);
            if (!grow_to(new_size))
            {
                return nullptr;
            }
        }

        char *ptr = base_addr_ + static_cast<size_t>(my_offset);
        // store length (little-endian native)
        *reinterpret_cast<uint32_t *>(ptr) = len;
        // zero terminator (safe to write)
        ptr[hdr + len] = '\0';
        // Caller writes payload to ptr + hdr
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
            std::lock_guard<std::mutex> lk(remote_mu_);
            auto it = remote_mmaps_.find(offset);
            if (it != remote_mmaps_.end())
                return it->second.first;

            // Evict if cache full
            if (remote_mmaps_.size() >= max_remote_mmaps_)
            {
                auto victim = remote_mmaps_.begin();
                const char *v_addr = victim->second.first;
                size_t v_sz = victim->second.second;
                if (v_addr && v_sz > 0) munmap(const_cast<char *>(v_addr), v_sz);
                remote_mmaps_.erase(victim);
            }

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
        if (offset == 0 || offset >= capacity_)
            return nullptr;

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

        uint64_t id = next_remote_id_.fetch_add(1, std::memory_order_relaxed);
        uint64_t encoded = make_encoded_remote_id(id);
        std::string fname = generate_remote_filename(encoded);
        std::string tmp = fname + ".tmp";

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
        fsync(fd);
        close(fd);

        if (rename(tmp.c_str(), fname.c_str()) != 0)
        {
            unlink(tmp.c_str());
            return 0;
        }

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
        uint64_t id = encoded_remote_id & (~(1ULL << 63));
        std::ostringstream ss;
        ss << remote_dir_ << "/shard_" << id_ << "_blob_" << id << ".bin";
        return ss.str();
    }

    void ShardArena::reset() noexcept
    {
        write_head_.store(64, std::memory_order_relaxed);

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

    void ShardArena::demote_range(uint64_t offset, size_t len)
    {
        const char *ptr = blob_ptr_from_offset_for_map(offset);
        if (!ptr)
            return;

        long ps = sysconf(_SC_PAGESIZE);
        const uintptr_t page_mask = ~(static_cast<uintptr_t>(ps - 1));

        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned_addr = addr & page_mask;
        size_t diff = addr - aligned_addr;
        size_t aligned_len = len + diff;
        // round up to multiple of page
        aligned_len = page_round_up(aligned_len, ps);

        // call madvise
        int ret = madvise(reinterpret_cast<void *>(aligned_addr), aligned_len, MADV_DONTNEED);
        (void)ret;
    }

    bool ShardArena::persist_range(uint64_t offset, size_t len, bool synchronous) const noexcept
    {
        if (!base_addr_)
            return false;
        if (offset + len > capacity_)
            return false;

        void *addr = base_addr_ + static_cast<size_t>(offset);
        int flags = synchronous ? MS_SYNC : MS_ASYNC;
        return msync(addr, len, flags) == 0;
    }

    bool ShardArena::grow_to(size_t new_size)
    {
        if (new_size <= capacity_)
            return true;

        // round up
        new_size = page_round_up(new_size, page_size_);

        // extend file on disk
#ifdef __linux__
        if (posix_fallocate(fd_, capacity_, static_cast<off_t>(new_size - capacity_)) != 0)
        {
            // fallback to ftruncate
            if (ftruncate(fd_, static_cast<off_t>(new_size)) != 0)
                return false;
        }
#else
        if (ftruncate(fd_, static_cast<off_t>(new_size)) != 0)
            return false;
#endif

        // try mremap (linux) to extend mapping in-place
#ifdef __linux__
        void *new_addr = mremap(base_addr_, capacity_, new_size, MREMAP_MAYMOVE);
        if (new_addr == MAP_FAILED)
        {
            // fallback: create new mapping, copy (should be rare)
            void *mp = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            if (mp == MAP_FAILED)
                return false;
            // msync old range to ensure persistence
            msync(base_addr_, capacity_, MS_SYNC);
            munmap(base_addr_, capacity_);
            base_addr_ = reinterpret_cast<char *>(mp);
            capacity_ = new_size;
            return true;
        }
        base_addr_ = reinterpret_cast<char *>(new_addr);
        capacity_ = new_size;
        return true;
#else
        // Non-linux: remap new mapping, unmap old
        void *mp = mmap(nullptr, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mp == MAP_FAILED)
            return false;
        msync(base_addr_, capacity_, MS_SYNC);
        munmap(base_addr_, capacity_);
        base_addr_ = reinterpret_cast<char *>(mp);
        capacity_ = new_size;
        return true;
#endif
    }

    std::string ShardArena::backing_filename() const
    {
        // If data_root is present in environment (convention used elsewhere), prefer it.
        // Attempt to derive from remote_dir_ if data_root not provided.
        // Form: <remote_dir_>/shard_<id>.blob
        std::ostringstream ss;
        ss << remote_dir_ << "/shard_" << id_ << ".blob";
        return ss.str();
    }

} // namespace pomai::memory