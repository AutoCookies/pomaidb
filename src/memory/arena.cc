/*
 * src/memory/arena.cc
 *
 * Implementation of PomaiArena (blob + seed region) with support for:
 * - alloc/free seeds
 * - alloc/free blobs in mapped region (power-of-two blocks)
 * - demote_blob / demote_blob_data (sync write to disk)
 * - demote_blob_async (queue-based, background worker)
 * - promote_remote (read remote file back into arena)
 * - read_remote_blob (read remote file into a temporary RAM buffer for one-shot defrost)
 * - resolve_pending_remote (small non-blocking inspector; blocking/wait-with-timeout lives in arena_async_demote.cc)
 *
 * Notes: heavy async logic lives here (single background worker).  Remote mmap cache
 * is LRU-capped to avoid unbounded growth.
 * [FIXED] Worker thread loop wrapped in try-catch to prevent process termination.
 */

#include "src/memory/arena.h"
#include "src/core/config.h"
#include "src/core/seed.h"

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
#include <chrono>
#include <sstream>
#include <fstream>
#include <random>
#include <cstdlib> // getenv

namespace pomai::memory
{

    // Helper: round up to next power-of-two bucket but not below MIN_BLOB_BLOCK
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

    uint64_t PomaiArena::block_size_for(uint64_t bytes) {
    uint64_t need = bytes;
    if (need < cfg_.min_blob_block) need = cfg_.min_blob_block;
    return roundup_pow2(need);
}

    // ---------------- Constructors / Destructor ----------------

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
          pending_counter_(1),
          pending_base_(0)
    {
        // seed RNG
        rng_.seed(std::random_device{}());
    }

    PomaiArena::~PomaiArena()
    {
        // stop demote worker if running
        {
            std::lock_guard<std::mutex> lk(demote_mu_);
            demote_worker_running_.store(false, std::memory_order_release);
            demote_cv_.notify_all();
        }
        if (demote_worker_.joinable())
            demote_worker_.join();

        cleanup();
    }

    // ---------------- Move semantics ----------------

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
        remote_mmap_lru_ = std::move(o.remote_mmap_lru_);
        remote_mmap_iter_ = std::move(o.remote_mmap_iter_);
        next_remote_id_ = o.next_remote_id_;

        pending_counter_.store(o.pending_counter_.load(std::memory_order_relaxed), std::memory_order_relaxed);

        rng_ = std::move(o.rng_);

        demote_worker_running_.store(false);
        o.demote_worker_running_.store(false);

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
        o.pending_counter_.store(1, std::memory_order_relaxed);
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
            remote_mmap_lru_ = std::move(o.remote_mmap_lru_);
            remote_mmap_iter_ = std::move(o.remote_mmap_iter_);
            next_remote_id_ = o.next_remote_id_;

            pending_counter_.store(o.pending_counter_.load(std::memory_order_relaxed), std::memory_order_relaxed);

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
        o.pending_counter_.store(1, std::memory_order_relaxed);

        return *this;
    }

    // ---------------- Factories for convenience ----------------

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
            use_gb = static_cast<double>(512) / 1024.0;
        }
        uint64_t bytes = static_cast<uint64_t>(use_gb * 1024.0 * 1024.0 * 1024.0);
        if (bytes == 0)
            bytes = 1;
        a.allocate_region(bytes);
        return a;
    }

    PomaiArena::PomaiArena(const pomai::config::PomaiConfig& global_cfg)
    : PomaiArena() 
{
    cfg_ = global_cfg.arena; 
 
    remote_dir_ = cfg_.remote_dir;
    max_remote_mmaps_ = cfg_.max_remote_mmaps;
    demote_batch_bytes_ = cfg_.demote_batch_bytes;
    max_pending_demotes_ = global_cfg.res.demote_async_max_pending;

    // Tiến hành cấp phát
    allocate_region(global_cfg.res.arena_mb_per_shard * 1024ULL * 1024ULL);
}

    // ---------------- allocate_region / grow / remap ----------------

    static inline size_t align_to_page(size_t v)
    {
        long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0)
            ps = 4096;
        size_t page = static_cast<size_t>(ps);
        return ((v + page - 1) / page) * page;
    }

    bool PomaiArena::allocate_region(uint64_t bytes)
    {
        if (bytes == 0)
            return false;

        std::lock_guard<std::mutex> lk(mu_);

        if (base_addr_)
        {
            return true; // already allocated
        }

        size_t map_bytes = static_cast<size_t>(align_to_page(bytes));

#ifdef MAP_HUGETLB
        void *p = mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p == MAP_FAILED)
        {
            p = mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }
#else
        void *p = mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

        if (p == MAP_FAILED)
        {
            int err = errno;
            std::cerr << "PomaiArena::allocate_region: mmap failed (" << err << "): " << std::strerror(err) << "\n";
            base_addr_ = nullptr;
            capacity_bytes_ = 0;
            return false;
        }

        base_addr_ = reinterpret_cast<char *>(p);
        capacity_bytes_ = map_bytes;

        // Partition: 25% seeds, rest blobs
        seed_region_bytes_ = static_cast<uint64_t>(static_cast<double>(capacity_bytes_) * cfg_.seed_region_ratio);
        if (seed_region_bytes_ < sizeof(Seed))
            seed_region_bytes_ = sizeof(Seed);
        seed_region_bytes_ = (seed_region_bytes_ / sizeof(Seed)) * sizeof(Seed);

        seed_base_ = base_addr_;
        seed_max_slots_ = seed_region_bytes_ / sizeof(Seed);
        seed_next_slot_ = 0;
        active_seeds_.clear();
        active_pos_.assign(seed_max_slots_, UINT64_MAX);
        free_seeds_.clear();

        blob_base_ = base_addr_ + seed_region_bytes_;
        blob_region_bytes_ = capacity_bytes_ - seed_region_bytes_;

        // Reserve first block so alloc_blob never returns offset 0
        uint64_t reserve_block = block_size_for(static_cast<uint64_t>(sizeof(uint32_t) + 1));
        if (reserve_block >= blob_region_bytes_)
        {
            // pathological: not enough room; make blob_next_offset_ small non-zero sentinel
            blob_next_offset_ = 1;
        }
        else
        {
            blob_next_offset_ = reserve_block;
        }

        // counters
        next_remote_id_ = 1;
        pending_counter_.store(1, std::memory_order_relaxed);

        max_pending_demotes_ = 1000;

        // configure remote mmap cap via env or default
        {
            const char *env = std::getenv("POMAI_MAX_REMOTE_MMAPS");
            size_t cap = 256;
            if (env)
            {
                try
                {
                    size_t v = static_cast<size_t>(std::stoul(env));
                    if (v > 0)
                        cap = v;
                }
                catch (...)
                {
                }
            }
            max_remote_mmaps_ = cap;
        }

        // start demote worker lazily (single background thread processing demote_queue_)
        {
            std::lock_guard<std::mutex> lk2(demote_mu_);
            if (!demote_worker_running_.load(std::memory_order_acquire))
            {
                demote_worker_running_.store(true, std::memory_order_release);
                demote_worker_ = std::thread([this]()
                                             {
                                                 while (demote_worker_running_.load(std::memory_order_acquire))
                                                 {
                                                     try
                                                     {
                                                         std::vector<DemoteTask> batch;
                                                         batch.reserve(64);
                                                         {
                                                             std::unique_lock<std::mutex> qlk(demote_mu_);
                                                             demote_cv_.wait_for(qlk, std::chrono::milliseconds(200), [this]()
                                                                                 { return !demote_queue_.empty() || !demote_worker_running_.load(std::memory_order_acquire); });
                                                             
                                                             if (!demote_worker_running_.load(std::memory_order_acquire) && demote_queue_.empty())
                                                                 break;
                                                             
                                                             size_t bytes_acc = 0;
                                                             while (!demote_queue_.empty() && batch.size() < 256)
                                                             {
                                                                 DemoteTask t = std::move(demote_queue_.front());
                                                                 demote_queue_.pop_front();
                                                                 bytes_acc += t.payload.size();
                                                                 batch.push_back(std::move(t));
                                                                 if (bytes_acc >= demote_batch_bytes_)
                                                                     break;
                                                             }
                                                         }

                                                         for (auto &task : batch)
                                                         {
                                                             uint64_t rid = task.remote_id; 
                                                             std::string fname = generate_remote_filename(rid);
                                                             std::string tmpname = fname + ".tmp";
                                                             
                                                             int fd = open(tmpname.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
                                                             if (fd < 0)
                                                             {
                                                                 std::cerr << "PomaiArena::demote_worker: open tmp failed: " << strerror(errno) << "\n";
                                                                 std::lock_guard<std::mutex> lk(mu_);
                                                                 remote_map_.erase(rid);
                                                                 // fulfill pending with failure
                                                                 if (task.pend)
                                                                 {
                                                                     std::lock_guard<std::mutex> plk(task.pend->m);
                                                                     task.pend->final_remote_id = 0;
                                                                     task.pend->done = true;
                                                                 }
                                                                 if (task.pend) task.pend->cv.notify_all();
                                                                 continue;
                                                             }
                                                             
                                                             ssize_t left = static_cast<ssize_t>(task.payload.size());
                                                             const char *ptr = task.payload.data();
                                                             bool write_failed = false;
                                                             while (left > 0)
                                                             {
                                                                 ssize_t w = write(fd, ptr, static_cast<size_t>(left));
                                                                 if (w < 0)
                                                                 {
                                                                     if (errno == EINTR)
                                                                         continue;
                                                                     std::cerr << "PomaiArena::demote_worker: write error: " << strerror(errno) << "\n";
                                                                     write_failed = true;
                                                                     break;
                                                                 }
                                                                 left -= w;
                                                                 ptr += w;
                                                             }
                                                             
                                                             if (!write_failed)
                                                             {
                                                                 fsync(fd);
                                                                 close(fd);
                                                                 if (rename(tmpname.c_str(), fname.c_str()) != 0)
                                                                 {
                                                                     std::cerr << "PomaiArena::demote_worker: rename failed: " << strerror(errno) << "\n";
                                                                     unlink(tmpname.c_str());
                                                                     std::lock_guard<std::mutex> lk(mu_);
                                                                     remote_map_.erase(rid);
                                                                     if (task.pend)
                                                                     {
                                                                         std::lock_guard<std::mutex> plk(task.pend->m);
                                                                         task.pend->final_remote_id = 0;
                                                                         task.pend->done = true;
                                                                     }
                                                                     if (task.pend) task.pend->cv.notify_all();
                                                                 }
                                                                 else
                                                                 {
                                                                     {
                                                                         std::lock_guard<std::mutex> lk(mu_);
                                                                         remote_map_[rid] = fname;
                                                                     }
                                                                     if (task.pend)
                                                                     {
                                                                         std::lock_guard<std::mutex> plk(task.pend->m);
                                                                         task.pend->final_remote_id = rid;
                                                                         task.pend->done = true;
                                                                     }
                                                                     if (task.pend) task.pend->cv.notify_all();
                                                                     // std::clog << "[PomaiArena] demote_worker: wrote remote_id=" << rid << " -> " << fname << "\n";
                                                                 }
                                                             }
                                                             else
                                                             {
                                                                 close(fd);
                                                                 unlink(tmpname.c_str());
                                                                 std::lock_guard<std::mutex> lk(mu_);
                                                                 remote_map_.erase(rid);
                                                                 if (task.pend)
                                                                 {
                                                                     std::lock_guard<std::mutex> plk(task.pend->m);
                                                                     task.pend->final_remote_id = 0;
                                                                     task.pend->done = true;
                                                                 }
                                                                 if (task.pend) task.pend->cv.notify_all();
                                                             }
                                                         }
                                                     }
                                                     catch (const std::exception &e)
                                                     {
                                                         std::cerr << "[PomaiArena] Critical error in demote worker: " << e.what() << "\n";
                                                         // Continue loop, don't crash
                                                     }
                                                     catch (...)
                                                     {
                                                         std::cerr << "[PomaiArena] Unknown critical error in demote worker\n";
                                                         // Continue loop
                                                     }
                                                 } });
            }
        }

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

        char *slot = seed_base_ + idx * sizeof(Seed);
        Seed *s = reinterpret_cast<Seed *>(slot);

        s->header.store(0ULL, std::memory_order_relaxed);
        s->entropy = 0;
        s->checksum = 0;
        s->type = 0;
        s->flags = 0;
        std::memset(s->reserved, 0, sizeof(s->reserved));
        std::memset(s->payload, 0, sizeof(s->payload));

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

        s->header.store(0ULL, std::memory_order_release);
        s->entropy = 0;
        s->checksum = 0;
        s->type = 0;
        s->flags = 0;

        uint64_t pos = active_pos_[idx];
        if (pos != UINT64_MAX)
        {
            uint64_t last_idx = active_seeds_.back();
            active_seeds_[pos] = last_idx;
            active_pos_[last_idx] = pos;
            active_seeds_.pop_back();
            active_pos_[idx] = UINT64_MAX;
        }

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
        const uint64_t total = hdr + static_cast<uint64_t>(len) + 1;
        const uint64_t block = block_size_for(total);

        auto fit = free_lists_.find(block);
        if (fit != free_lists_.end())
        {
            // try to find a non-zero offset from freelist (0 is reserved sentinel)
            while (!fit->second.empty())
            {
                uint64_t offset = fit->second.back();
                fit->second.pop_back();
                if (offset == 0)
                    continue; // skip reserved/invalid entry
                if (offset + block <= blob_region_bytes_)
                {
                    char *p = blob_base_ + offset;
                    uint32_t *lenptr = reinterpret_cast<uint32_t *>(p);
                    *lenptr = len;
                    char *payload = p + hdr;
                    payload[len] = '\0';
                    return p;
                }
                // otherwise ignore invalid entry and continue
            }
        }

        // bump allocate (align to block)
        uint64_t aligned_next = (blob_next_offset_ + (block - 1)) & ~(block - 1);
        if (aligned_next + block > blob_region_bytes_)
        {
            return nullptr;
        }
        uint64_t offset = aligned_next;
        char *p = blob_base_ + offset;
        uint32_t *lenptr = reinterpret_cast<uint32_t *>(p);
        *lenptr = len;
        char *payload = p + hdr;
        payload[len] = '\0';
        blob_next_offset_ = offset + block;

        // Ensure we never return offset == 0 (should be guaranteed by reserve in allocate_region)
        if (offset == 0)
        {
            // This should not happen; advance to next block and return adjusted pointer.
            uint64_t next = block_size_for(hdr + 1);
            if (next + block > blob_region_bytes_)
            {
                return nullptr;
            }
            offset = next;
            p = blob_base_ + offset;
            lenptr = reinterpret_cast<uint32_t *>(p);
            *lenptr = len;
            payload = p + hdr;
            payload[len] = '\0';
            blob_next_offset_ = offset + block;
        }

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
        if (offset == 0)
            return; // never reuse reserved zero offset

        uint32_t stored_len = *reinterpret_cast<uint32_t *>(header_ptr);
        const uint64_t total = sizeof(uint32_t) + static_cast<uint64_t>(stored_len) + 1;
        const uint64_t block = block_size_for(total);

        auto &vec = free_lists_[block];
        if (vec.size() < cfg_.max_freelist_per_bucket) {
            vec.push_back(offset);
        }
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

    static void evict_one_remote_mmap(std::unordered_map<uint64_t, std::pair<const char *, size_t>> &remote_mmaps,
                                      std::list<uint64_t> &lru,
                                      std::unordered_map<uint64_t, std::list<uint64_t>::iterator> &iter_map)
    {
        if (lru.empty())
            return;
        uint64_t victim = lru.back();
        lru.pop_back();
        auto it_rm = remote_mmaps.find(victim);
        if (it_rm != remote_mmaps.end())
        {
            const char *addr = it_rm->second.first;
            size_t sz = it_rm->second.second;
            if (addr && sz > 0)
            {
                munmap(const_cast<char *>(addr), sz);
            }
            remote_mmaps.erase(it_rm);
        }
        iter_map.erase(victim);
    }

    const char *PomaiArena::blob_ptr_from_offset_for_map(uint64_t offset) const
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !blob_base_)
            return nullptr;

        // Local offset path
        if (offset < blob_region_bytes_)
        {
            // reject reserved 0 offset as "not mapped"
            if (offset == 0)
                return nullptr;
            return blob_base_ + offset;
        }

        // Remote id path: if mapped already, return mapping and update LRU
        auto mit = remote_mmaps_.find(offset);
        if (mit != remote_mmaps_.end())
        {
            // move to front of LRU
            auto itit = remote_mmap_iter_.find(offset);
            if (itit != remote_mmap_iter_.end())
            {
                remote_mmap_lru_.erase(itit->second);
            }
            remote_mmap_lru_.push_front(offset);
            remote_mmap_iter_[offset] = remote_mmap_lru_.begin();
            return mit->second.first;
        }

        auto it = remote_map_.find(offset);
        if (it == remote_map_.end())
            return nullptr;

        const std::string &fname = it->second;
        if (fname.empty())
            return nullptr; // pending

        int fd = open(fname.c_str(), O_RDONLY);
        if (fd < 0)
        {
            return nullptr;
        }
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

        // evict LRU entries if capacity exceeded
        remote_mmaps_.emplace(offset, std::make_pair(addr, sz));
        // push front to LRU
        remote_mmap_lru_.push_front(offset);
        remote_mmap_iter_[offset] = remote_mmap_lru_.begin();

        // evict until under cap
        while (remote_mmaps_.size() > max_remote_mmaps_)
        {
            evict_one_remote_mmap(remote_mmaps_, remote_mmap_lru_, remote_mmap_iter_);
        }

        return addr;
    }

    // ---------------- Remote (disk) operations ----------------

    std::string PomaiArena::generate_remote_filename(uint64_t id) const
    {
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
        if (local_offset == 0)
            return 0; // never demote reserved offset

        char *p = blob_base_ + local_offset;
        uint32_t blen = *reinterpret_cast<uint32_t *>(p);
        const uint64_t total = sizeof(uint32_t) + static_cast<uint64_t>(blen) + 1;
        if (total == 0)
            return 0;

        uint64_t id = next_remote_id_++;
        std::string fname = generate_remote_filename(id);

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
        if (rename(tmpname.c_str(), fname.c_str()) != 0)
        {
            std::cerr << "PomaiArena::demote_blob: rename failed: " << strerror(errno) << "\n";
            unlink(tmpname.c_str());
            return 0;
        }

        uint64_t block = block_size_for(total);
        auto &vec = free_lists_[block];
        if (local_offset != 0 && vec.size() < cfg_.max_freelist_per_bucket)
            vec.push_back(local_offset);

        uint64_t remote_id = blob_region_bytes_ + id;
        remote_map_[remote_id] = fname;

        return remote_id;
    }

    uint64_t PomaiArena::demote_blob_data(const char *data_with_header, uint32_t total_bytes)
    {
        if (!data_with_header || total_bytes == 0)
            return 0;

        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_)
            return 0;

        uint64_t id = next_remote_id_++;
        std::string fname = generate_remote_filename(id);

        std::string tmpname = fname + ".tmp";
        int fd = open(tmpname.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (fd < 0)
        {
            std::cerr << "PomaiArena::demote_blob_data: open tmp failed: " << strerror(errno) << "\n";
            return 0;
        }

        // ssize_t w = 0;
        const char *ptr = data_with_header;
        size_t left = static_cast<size_t>(total_bytes);
        while (left > 0)
        {
            ssize_t n = write(fd, ptr, left);
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                std::cerr << "PomaiArena::demote_blob_data: write failed: " << strerror(errno) << "\n";
                close(fd);
                unlink(tmpname.c_str());
                return 0;
            }
            left -= static_cast<size_t>(n);
            ptr += n;
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
        return remote_id;
    }

    // Light wrapper: forward char* overload to the generic void* overload.
    // The heavy implementation of demote_blob_async(const void*, uint32_t) lives in arena_async_demote.cc
    uint64_t PomaiArena::demote_blob_async(const char *data_with_header, uint32_t total_bytes)
    {
        if (!data_with_header || total_bytes == 0)
            return 0;
        return demote_blob_async(static_cast<const void *>(data_with_header), total_bytes);
    }

    // ---------------- Promote / Read remote ----------------

    uint64_t PomaiArena::promote_remote(uint64_t remote_id)
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!base_addr_ || !blob_base_)
            return UINT64_MAX;

        auto it = remote_map_.find(remote_id);
        if (it == remote_map_.end())
            return UINT64_MAX;

        const std::string &fname = it->second;
        if (fname.empty())
            return UINT64_MAX;

        int fd = open(fname.c_str(), O_RDONLY);
        if (fd < 0)
        {
            return UINT64_MAX;
        }
        struct stat st;
        if (fstat(fd, &st) != 0)
        {
            close(fd);
            return UINT64_MAX;
        }
        size_t sz = static_cast<size_t>(st.st_size);
        if (sz == 0)
        {
            close(fd);
            return UINT64_MAX;
        }
        std::vector<char> buf(sz);
        ssize_t r = read(fd, buf.data(), sz);
        close(fd);
        if (r != static_cast<ssize_t>(sz))
            return UINT64_MAX;

        uint32_t blen = *reinterpret_cast<uint32_t *>(buf.data());
        if (static_cast<size_t>(blen) + sizeof(uint32_t) + 1 != sz)
        {
            return UINT64_MAX;
        }

        char *hdr = alloc_blob(blen);
        if (!hdr)
            return UINT64_MAX;
        std::memcpy(hdr, buf.data(), sz);

        uint64_t off = offset_from_blob_ptr(hdr);
        if (off == UINT64_MAX)
            return UINT64_MAX;

        // remove remote mapping entry (we promoted into local arena)
        remote_map_.erase(remote_id);

        return off;
    }

    // Read remote blob into RAM buffer (including header). Used for one-shot defrost without
    // permanently promoting into the arena. Returns empty vector on failure.
    std::vector<char> PomaiArena::read_remote_blob(uint64_t remote_id) const
    {
        std::lock_guard<std::mutex> lk(mu_);

        auto it = remote_map_.find(remote_id);
        if (it == remote_map_.end())
            return {}; // not found

        const std::string &fname = it->second;
        if (fname.empty())
            return {}; // pending

        std::ifstream file(fname, std::ios::binary | std::ios::ate);
        if (!file.is_open())
        {
            std::cerr << "[PomaiArena] Failed to open remote blob: " << fname << "\n";
            return {};
        }

        std::streamsize size = file.tellg();
        if (size <= 0)
            return {};
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(static_cast<size_t>(size));
        if (!file.read(buffer.data(), size))
        {
            std::cerr << "[PomaiArena] Failed to read remote blob: " << fname << "\n";
            return {};
        }
        return buffer;
    }

    // Non-blocking inspector: if placeholder present and resolved to filename, return placeholder id,
    // otherwise return 0. This is a cheap check used by callers that don't want to wait.
    uint64_t PomaiArena::resolve_pending_remote(uint64_t placeholder_remote_id) const
    {
        // non-blocking quick path: if placeholder -> check pending_map_ for completion
        if (!is_placeholder_id(placeholder_remote_id))
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = remote_map_.find(placeholder_remote_id);
            if (it == remote_map_.end())
                return 0;
            const std::string &fname = it->second;
            if (fname.empty())
                return 0;
            return placeholder_remote_id;
        }

        std::shared_ptr<PendingDemote> pend;
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            auto it = pending_map_.find(placeholder_remote_id);
            if (it == pending_map_.end())
                return 0;
            pend = it->second;
        }
        if (!pend)
            return 0;
        std::lock_guard<std::mutex> plk(pend->m);
        if (pend->done)
            return pend->final_remote_id;
        return 0;
    }

    // ---------------- Misc helpers / introspection ----------------

    size_t PomaiArena::get_demote_queue_length() const noexcept
    {
        std::lock_guard<std::mutex> lk(demote_mu_);
        return demote_queue_.size();
    }

    void PomaiArena::set_demote_queue_max(size_t max_pending)
    {
        std::lock_guard<std::mutex> lk(demote_mu_);
        max_pending_demotes_ = max_pending;
    }

    size_t PomaiArena::get_demote_queue_max() const noexcept
    {
        std::lock_guard<std::mutex> lk(demote_mu_);
        return max_pending_demotes_;
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
        remote_mmap_lru_.clear();
        remote_mmap_iter_.clear();
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

        {
            std::lock_guard<std::mutex> lk2(demote_mu_);
            demote_worker_running_.store(false, std::memory_order_release);
            demote_cv_.notify_all();
        }
        if (demote_worker_.joinable())
            demote_worker_.join();
    }

} // namespace pomai::memory