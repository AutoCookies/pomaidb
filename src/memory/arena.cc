// src/memory/arena.cc
//
// Implementation for PomaiArena declared in arena.h
//
// This implementation favors correctness and clarity. It is intentionally conservative
// in its use of a single mutex to protect internal state and uses simple file-backed
// demotion with lazy mmap on promotion. For production, replace the remote storage
// path and improve IO reliability, atomic rename, crash-recovery, and GC.

#include "src/memory/arena.h"
#include "src/core/seed.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cassert>
#include <unistd.h>
#include <sys/types.h>
#include <algorithm>
#include <vector>

namespace pomai::memory
{

    PomaiArena::PomaiArena()
        : base_addr_(nullptr),
          capacity_bytes_(0),
          seed_base_(nullptr),
          seed_region_bytes_(0),
          seed_max_slots_(0),
          seed_next_slot_(0),
          blob_base_(nullptr),
          blob_region_bytes_(0),
          blob_next_offset_(0),
          next_remote_id_(1),
          rng_(std::random_device{}()) {}

    PomaiArena::PomaiArena(uint64_t bytes) : PomaiArena()
    {
        allocate_region(bytes);
    }

    PomaiArena::~PomaiArena()
    {
        cleanup();
    }

    PomaiArena::PomaiArena(PomaiArena &&o) noexcept
    {
        std::lock_guard<std::mutex> lk(o.mu_);
        base_addr_ = o.base_addr_;
        capacity_bytes_ = o.capacity_bytes_;
        seed_base_ = o.seed_base_;
        seed_region_bytes_ = o.seed_region_bytes_;
        seed_max_slots_ = o.seed_max_slots_;
        seed_next_slot_ = o.seed_next_slot_;
        blob_base_ = o.blob_base_;
        blob_region_bytes_ = o.blob_region_bytes_;
        blob_next_offset_ = o.blob_next_offset_;
        free_seeds_ = std::move(o.free_seeds_);
        active_seeds_ = std::move(o.active_seeds_);
        active_pos_ = std::move(o.active_pos_);
        free_lists_ = std::move(o.free_lists_);
        remote_map_ = std::move(o.remote_map_);
        remote_mmaps_ = std::move(o.remote_mmaps_);
        next_remote_id_ = o.next_remote_id_;
        rng_ = std::move(o.rng_);

        o.base_addr_ = nullptr;
        o.capacity_bytes_ = 0;
        o.seed_base_ = nullptr;
        o.blob_base_ = nullptr;
        o.seed_region_bytes_ = 0;
        o.blob_region_bytes_ = 0;
        o.seed_max_slots_ = 0;
        o.seed_next_slot_ = 0;
        o.blob_next_offset_ = 0;
        o.next_remote_id_ = 1;
    }

    PomaiArena &PomaiArena::operator=(PomaiArena &&o) noexcept
    {
        if (this == &o)
            return *this;
        cleanup();
        {
            std::lock_guard<std::mutex> lk(o.mu_);
            base_addr_ = o.base_addr_;
            capacity_bytes_ = o.capacity_bytes_;
            seed_base_ = o.seed_base_;
            seed_region_bytes_ = o.seed_region_bytes_;
            seed_max_slots_ = o.seed_max_slots_;
            seed_next_slot_ = o.seed_next_slot_;
            blob_base_ = o.blob_base_;
            blob_region_bytes_ = o.blob_region_bytes_;
            blob_next_offset_ = o.blob_next_offset_;
            free_seeds_ = std::move(o.free_seeds_);
            active_seeds_ = std::move(o.active_seeds_);
            active_pos_ = std::move(o.active_pos_);
            free_lists_ = std::move(o.free_lists_);
            remote_map_ = std::move(o.remote_map_);
            remote_mmaps_ = std::move(o.remote_mmaps_);
            next_remote_id_ = o.next_remote_id_;
            rng_ = std::move(o.rng_);
        }

        o.base_addr_ = nullptr;
        o.capacity_bytes_ = 0;
        o.seed_base_ = nullptr;
        o.blob_base_ = nullptr;
        o.seed_region_bytes_ = 0;
        o.blob_region_bytes_ = 0;
        o.seed_max_slots_ = 0;
        o.seed_next_slot_ = 0;
        o.blob_next_offset_ = 0;
        o.next_remote_id_ = 1;

        return *this;
    }

    // ---------------- Factories ----------------
    PomaiArena PomaiArena::FromMB(uint64_t mb)
    {
        PomaiArena a;
        if (mb == 0)
            return a;
        uint64_t bytes = mb * 1024ULL * 1024ULL;
        a.allocate_region(bytes);
        return a;
    }

    PomaiArena PomaiArena::FromGB(double gb)
    {
        PomaiArena a;
        double use_gb = gb;
        if (use_gb <= 0.0)
        {
            // default: 512MB per shard (compatible with earlier config)
            use_gb = static_cast<double>(512) / 1024.0;
        }
        uint64_t bytes = static_cast<uint64_t>(use_gb * 1024.0 * 1024.0 * 1024.0);
        if (bytes == 0)
            bytes = 1;
        a.allocate_region(bytes);
        return a;
    }

    bool PomaiArena::allocate_region(uint64_t bytes)
    {
        if (bytes == 0)
            return false;

        std::lock_guard<std::mutex> lk(mu_);

        if (base_addr_)
        {
            // already allocated
            return true;
        }

        // Try to mmap with/without hugepages (best-effort)
#ifdef MAP_HUGETLB
        void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p == MAP_FAILED)
        {
            p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }
#else
        void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

        if (p == MAP_FAILED)
        {
            int err = errno;
            std::cerr << "PomaiArena: mmap failed (" << err << "): " << std::strerror(err) << "\n";
            base_addr_ = nullptr;
            capacity_bytes_ = 0;
            return false;
        }

        base_addr_ = reinterpret_cast<char *>(p);
        capacity_bytes_ = bytes;

        // Partition: seed region first, then blob region
        seed_region_bytes_ = static_cast<uint64_t>(static_cast<double>(capacity_bytes_) * SEED_REGION_RATIO);
        if (seed_region_bytes_ < sizeof(Seed))
            seed_region_bytes_ = sizeof(Seed);
        // round down to multiple of Seed size to avoid partial slots
        seed_region_bytes_ = (seed_region_bytes_ / sizeof(Seed)) * sizeof(Seed);
        seed_base_ = base_addr_;
        seed_max_slots_ = seed_region_bytes_ / sizeof(Seed);
        seed_next_slot_ = 0;
        active_seeds_.clear();
        active_pos_.assign(seed_max_slots_, UINT64_MAX);
        free_seeds_.clear();

        blob_base_ = base_addr_ + seed_region_bytes_;
        blob_region_bytes_ = capacity_bytes_ - seed_region_bytes_;
        blob_next_offset_ = 0;

        // init counters
        next_remote_id_ = 1;

        return true;
    }

    bool PomaiArena::is_valid() const noexcept
    {
        std::lock_guard<std::mutex> lk(mu_);
        return base_addr_ != nullptr && capacity_bytes_ > 0;
    }

    void PomaiArena::seed_rng(uint64_t seed)
    {
        std::lock_guard<std::mutex> lk(mu_);
        rng_.seed(seed);
    }

    uint64_t PomaiArena::get_capacity_bytes() const noexcept
    {
        std::lock_guard<std::mutex> lk(mu_);
        return capacity_bytes_;
    }

    static inline uint64_t roundup_pow2(uint64_t v)
    {
        if (v <= 1)
            return 1;
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;
        return ++v;
    }

    uint64_t PomaiArena::block_size_for(uint64_t bytes)
    {
        uint64_t need = bytes;
        if (need < MIN_BLOB_BLOCK)
            need = MIN_BLOB_BLOCK;
        return roundup_pow2(need);
    }

    // ---------------- Seed allocation API ----------------
    Seed *PomaiArena::alloc_seed()
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !seed_base_)
            return nullptr;

        uint64_t idx = UINT64_MAX;
        if (!free_seeds_.empty())
        {
            idx = free_seeds_.back();
            free_seeds_.pop_back();
        }
        else if (seed_next_slot_ < seed_max_slots_)
        {
            idx = seed_next_slot_++;
        }
        else
        {
            return nullptr; // exhausted
        }

        // pointer to slot
        char *slot = seed_base_ + idx * sizeof(Seed);
        Seed *s = reinterpret_cast<Seed *>(slot);

        // initialize fields to zero / safe defaults
        s->header.store(0ULL, std::memory_order_relaxed);
        s->entropy = 0;
        s->checksum = 0;
        s->type = 0;
        s->flags = 0;
        std::memset(s->reserved, 0, sizeof(s->reserved));
        std::memset(s->payload, 0, sizeof(s->payload));

        // add to active_seeds_
        active_pos_[idx] = active_seeds_.size();
        active_seeds_.push_back(idx);

        return s;
    }

    void PomaiArena::free_seed(Seed *s)
    {
        if (!s)
            return;
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !seed_base_)
            return;
        char *p = reinterpret_cast<char *>(s);
        if (p < seed_base_ || p >= seed_base_ + seed_region_bytes_)
            return;
        uint64_t idx = static_cast<uint64_t>((p - seed_base_) / sizeof(Seed));
        if (idx >= seed_max_slots_)
            return;

        // clear logical contents
        s->header.store(0ULL, std::memory_order_release);
        s->entropy = 0;
        s->checksum = 0;
        s->type = 0;
        s->flags = 0;

        // Remove from active_seeds_ (swap-remove)
        uint64_t pos = active_pos_[idx];
        if (pos != UINT64_MAX)
        {
            uint64_t last_idx = active_seeds_.back();
            active_seeds_[pos] = last_idx;
            active_pos_[last_idx] = pos;
            active_seeds_.pop_back();
            active_pos_[idx] = UINT64_MAX;
        }

        // push into free_seeds_ for reuse
        free_seeds_.push_back(idx);
    }

    uint64_t PomaiArena::num_active_seeds() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return active_seeds_.size();
    }

    Seed *PomaiArena::get_random_seed()
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (active_seeds_.empty())
            return nullptr;
        std::uniform_int_distribution<uint64_t> dist(0, active_seeds_.size() - 1);
        uint64_t pos = dist(rng_);
        uint64_t idx = active_seeds_[pos];
        return reinterpret_cast<Seed *>(seed_base_ + idx * sizeof(Seed));
    }

    // ---------------- Blob allocator ----------------
    char *PomaiArena::alloc_blob(uint32_t len)
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !blob_base_)
            return nullptr;

        const uint64_t hdr = sizeof(uint32_t);
        const uint64_t total = hdr + static_cast<uint64_t>(len) + 1; // extra nul
        const uint64_t block = block_size_for(total);

        auto fit = free_lists_.find(block);
        if (fit != free_lists_.end() && !fit->second.empty())
        {
            uint64_t offset = fit->second.back();
            fit->second.pop_back();
            char *p = blob_base_ + offset;
            uint32_t *lenptr = reinterpret_cast<uint32_t *>(p);
            *lenptr = len;
            char *payload = p + hdr;
            payload[len] = '\0';
            return p;
        }

        // bump allocate (align to block)
        uint64_t aligned_next = (blob_next_offset_ + (block - 1)) & ~(block - 1);
        if (aligned_next + block > blob_region_bytes_)
        {
            // No space available in arena blob region
            return nullptr;
        }
        uint64_t offset = aligned_next;
        char *p = blob_base_ + offset;
        uint32_t *lenptr = reinterpret_cast<uint32_t *>(p);
        *lenptr = len;
        char *payload = p + hdr;
        payload[len] = '\0';
        blob_next_offset_ = offset + block;
        return p;
    }

    void PomaiArena::free_blob(char *header_ptr)
    {
        if (!header_ptr)
            return;
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !blob_base_)
            return;
        if (header_ptr < blob_base_ || header_ptr >= blob_base_ + blob_region_bytes_)
            return;

        uint64_t offset = static_cast<uint64_t>(header_ptr - blob_base_);
        uint32_t stored_len = *reinterpret_cast<uint32_t *>(header_ptr);
        const uint64_t total = sizeof(uint32_t) + static_cast<uint64_t>(stored_len) + 1;
        const uint64_t block = block_size_for(total);

        auto &vec = free_lists_[block];
        if (vec.size() < MAX_FREELIST_PER_BUCKET)
            vec.push_back(offset);
        // otherwise drop it; freelist capped
    }

    uint64_t PomaiArena::offset_from_blob_ptr(const char *p) const noexcept
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !blob_base_)
            return UINT64_MAX;
        if (p < blob_base_ || p >= blob_base_ + blob_region_bytes_)
            return UINT64_MAX;
        return static_cast<uint64_t>(p - blob_base_);
    }

    const char *PomaiArena::blob_ptr_from_offset_for_map(uint64_t offset) const
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !blob_base_)
            return nullptr;

        // Local
        if (offset < blob_region_bytes_)
        {
            return blob_base_ + offset;
        }

        // Remote id
        auto mit = remote_mmaps_.find(offset);
        if (mit != remote_mmaps_.end())
        {
            return mit->second.first;
        }

        auto it = remote_map_.find(offset);
        if (it == remote_map_.end())
            return nullptr;

        const std::string &fname = it->second;

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

    // ---------------- Demote / Promote ----------------
    std::string PomaiArena::generate_remote_filename(uint64_t id) const
    {
        // Build deterministic but unique filename containing pid, time and id.
        std::ostringstream ss;
        pid_t pid = getpid();
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        ss << remote_dir_ << "/pomai_blob_" << pid << "_" << now << "_" << id << ".bin";
        return ss.str();
    }

    uint64_t PomaiArena::demote_blob(uint64_t local_offset)
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !blob_base_)
            return 0;
        if (local_offset >= blob_region_bytes_)
            return 0;

        char *p = blob_base_ + local_offset;
        uint32_t blen = *reinterpret_cast<uint32_t *>(p);
        const uint64_t total = sizeof(uint32_t) + static_cast<uint64_t>(blen) + 1;
        if (total == 0)
            return 0;

        // create remote filename
        uint64_t id = next_remote_id_++;
        std::string fname = generate_remote_filename(id);

        // write atomically: write to temp and rename
        std::string tmpname = fname + ".tmp";
        int fd = open(tmpname.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (fd < 0)
        {
            std::cerr << "PomaiArena::demote_blob: open tmp failed: " << strerror(errno) << "\n";
            return 0;
        }
        ssize_t w = write(fd, p, static_cast<size_t>(total));
        if (w != static_cast<ssize_t>(total))
        {
            std::cerr << "PomaiArena::demote_blob: write failed (w=" << w << " wanted=" << total << "): " << strerror(errno) << "\n";
            close(fd);
            unlink(tmpname.c_str());
            return 0;
        }
        fsync(fd);
        close(fd);
        // atomic rename
        if (rename(tmpname.c_str(), fname.c_str()) != 0)
        {
            std::cerr << "PomaiArena::demote_blob: rename failed: " << strerror(errno) << "\n";
            unlink(tmpname.c_str());
            return 0;
        }

        // Return block to freelist
        uint64_t block = block_size_for(total);
        auto &vec = free_lists_[block];
        if (vec.size() < MAX_FREELIST_PER_BUCKET)
            vec.push_back(local_offset);

        // encode remote id
        uint64_t remote_id = blob_region_bytes_ + id;
        remote_map_[remote_id] = fname;

        std::cerr << "PomaiArena::demote_blob: demoted offset=" << local_offset << " -> file=" << fname << " remote_id=" << remote_id << " bytes=" << total << "\n";

        // Note: mapping not created now (lazy). Caller should replace stored offset with remote_id.
        return remote_id;
    }

    uint64_t PomaiArena::demote_blob_data(const char *data_with_header, uint32_t total_bytes)
    {
        if (!data_with_header || total_bytes == 0)
            return 0;

        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_)
            return 0;

        // create remote filename
        uint64_t id = next_remote_id_++;
        std::string fname = generate_remote_filename(id);

        // write atomically: write to temp and rename
        std::string tmpname = fname + ".tmp";
        int fd = open(tmpname.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (fd < 0)
        {
            std::cerr << "PomaiArena::demote_blob_data: open tmp failed: " << strerror(errno) << "\n";
            return 0;
        }

        ssize_t w = 0;
        const char *ptr = data_with_header;
        size_t left = static_cast<size_t>(total_bytes);
        while (left > 0)
        {
            ssize_t n = write(fd, ptr, left);
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                std::cerr << "PomaiArena::demote_blob_data: write error: " << strerror(errno) << "\n";
                close(fd);
                unlink(tmpname.c_str());
                return 0;
            }
            w += n;
            ptr += n;
            left -= static_cast<size_t>(n);
        }

        if (w != static_cast<ssize_t>(total_bytes))
        {
            std::cerr << "PomaiArena::demote_blob_data: short write (w=" << w << " expected=" << total_bytes << ")\n";
            close(fd);
            unlink(tmpname.c_str());
            return 0;
        }

        fsync(fd);
        close(fd);
        if (rename(tmpname.c_str(), fname.c_str()) != 0)
        {
            std::cerr << "PomaiArena::demote_blob_data: rename failed: " << strerror(errno) << "\n";
            unlink(tmpname.c_str());
            return 0;
        }

        uint64_t remote_id = blob_region_bytes_ + id;
        remote_map_[remote_id] = fname;

        std::cerr << "PomaiArena::demote_blob_data: demoted direct buffer -> file=" << fname << " remote_id=" << remote_id << " bytes=" << total_bytes << "\n";

        return remote_id;
    }

    uint64_t PomaiArena::promote_remote(uint64_t remote_id)
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !blob_base_)
            return UINT64_MAX;
        auto it = remote_map_.find(remote_id);
        if (it == remote_map_.end())
            return UINT64_MAX;
        const std::string &fname = it->second;

        // read file into vector
        int fd = open(fname.c_str(), O_RDONLY);
        if (fd < 0)
            return UINT64_MAX;
        struct stat st;
        if (fstat(fd, &st) != 0)
        {
            close(fd);
            return UINT64_MAX;
        }
        size_t filesize = static_cast<size_t>(st.st_size);
        if (filesize < sizeof(uint32_t))
        {
            close(fd);
            return UINT64_MAX;
        }

        std::vector<char> buf(filesize);
        ssize_t r = pread(fd, buf.data(), filesize, 0);
        close(fd);
        if (r != static_cast<ssize_t>(filesize))
            return UINT64_MAX;

        uint32_t blen = *reinterpret_cast<uint32_t *>(buf.data());
        const uint64_t total = sizeof(uint32_t) + static_cast<uint64_t>(blen) + 1;
        if (total != filesize)
            return UINT64_MAX;

        const uint64_t block = block_size_for(total);

        uint64_t offset = UINT64_MAX;
        auto fit = free_lists_.find(block);
        if (fit != free_lists_.end() && !fit->second.empty())
        {
            offset = fit->second.back();
            fit->second.pop_back();
            char *p = blob_base_ + offset;
            std::memcpy(p, buf.data(), filesize);
        }
        else
        {
            uint64_t aligned_next = (blob_next_offset_ + (block - 1)) & ~(block - 1);
            if (aligned_next + block > blob_region_bytes_)
                return UINT64_MAX;
            offset = aligned_next;
            char *p = blob_base_ + offset;
            std::memcpy(p, buf.data(), filesize);
            blob_next_offset_ = offset + block;
        }

        // If remote was mmapped, unmap and remove mapping
        auto mit = remote_mmaps_.find(remote_id);
        if (mit != remote_mmaps_.end())
        {
            const char *addr = mit->second.first;
            size_t sz = mit->second.second;
            if (addr && sz > 0)
            {
                munmap(const_cast<char *>(addr), sz);
            }
            remote_mmaps_.erase(mit);
        }

        // remove remote file metadata and unlink file (best-effort)
        unlink(fname.c_str());
        remote_map_.erase(remote_id);

        std::cerr << "PomaiArena::promote_remote: promoted remote_id=" << remote_id << " into offset=" << offset << "\n";

        return offset;
    }

    const char *PomaiArena::blob_base_ptr() const noexcept
    {
        std::lock_guard<std::mutex> lk(mu_);
        return blob_base_;
    }

    uint64_t PomaiArena::blob_region_size() const noexcept
    {
        std::lock_guard<std::mutex> lk(mu_);
        return blob_region_bytes_;
    }

    const char *PomaiArena::seed_base_ptr() const noexcept
    {
        std::lock_guard<std::mutex> lk(mu_);
        return seed_base_;
    }

    uint64_t PomaiArena::seed_region_size() const noexcept
    {
        std::lock_guard<std::mutex> lk(mu_);
        return seed_region_bytes_;
    }

    void PomaiArena::cleanup()
    {
        std::lock_guard<std::mutex> lk(mu_);
        // unmap remote mmaps
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
        remote_map_.clear();

        if (base_addr_ && capacity_bytes_ > 0)
        {
            munmap(base_addr_, capacity_bytes_);
        }
        base_addr_ = nullptr;
        capacity_bytes_ = 0;
        seed_base_ = nullptr;
        seed_region_bytes_ = 0;
        seed_max_slots_ = 0;
        seed_next_slot_ = 0;
        blob_base_ = nullptr;
        blob_region_bytes_ = 0;
        blob_next_offset_ = 0;

        free_seeds_.clear();
        active_seeds_.clear();
        active_pos_.clear();
        free_lists_.clear();
    }

} // namespace pomai::memory