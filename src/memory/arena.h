#pragma once
// src/memory/arena.h
//
// PomaiArena -- a simple mmap-backed arena that provides:
//   - segregated "seed" and "blob" regions inside a single mmap()
//   - slab-style blob allocations (power-of-two bucket freelists)
//   - simple seed-slot management and active-seed indexing for random sampling
//   - cold-demote / remote-file promote support (demote_blob/promote_remote)
//   - utilities to resolve an offset (stored in index payloads) to a pointer,
//     including lazy mmap of demoted remote files.
//
// This header declares the public API. Implementation is in arena.cc and
// arena_async_demote.cc.
//
// Notes (production-readiness):
//  - Pending async demotes are represented by placeholder ids whose MSB is set.
//    The async path publishes a placeholder immediately and resolves it later to
//    a concrete remote id once the background worker finishes writing the blob.
//  - The header provides a small PendingDemote helper used by the async path so
//    callers can wait for completion if desired.
//  - The design favors correctness and clear synchronization (mutexes + condition_variable).
//
// Threading: Most mutable members are protected by `mu_`. The async/pending maps
// use `pending_mu_`. The demote queue is protected by `demote_mu_`.

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <mutex>
#include <optional>
#include <condition_variable>
#include <deque>
#include <thread>
#include <atomic>
#include <memory>
#include <sstream>
#include <iostream>
#include <list>
#include "src/core/config.h"

struct Seed; // forward-declare; concrete definition in src/core/seed.h

namespace pomai::memory
{

    // PendingDemote: small struct used by asynchronous demote helpers.
    // Producers create a shared_ptr<PendingDemote>, publish it into pending_map_
    // keyed by placeholder id, and background worker will set final_remote_id and
    // notify via the condition variable when complete.
    struct PendingDemote
    {
        PendingDemote() : done(false), final_remote_id(0) {}
        std::mutex m;
        std::condition_variable cv;
        bool done;
        uint64_t final_remote_id;
    };

    // Placeholder encoding helpers:
    // We encode placeholders by setting the MSB (bit 63). This is simple,
    // portable and avoids depending on arena internals. Real deployments may
    // prefer a different scheme (e.g., using pending_base_ offsets).
    static inline uint64_t make_placeholder(uint64_t ctr) noexcept
    {
        return (1ULL << 63) | (ctr & 0x7FFFFFFFFFFFFFFFULL);
    }
    static inline bool is_placeholder_id(uint64_t id) noexcept
    {
        return (id & (1ULL << 63)) != 0;
    }

    class PomaiArena
    {
    public:
        // Construct an empty (invalid) arena. Must call allocate_region() or use factory.
        PomaiArena();

        // Construct and allocate `bytes` bytes immediately (mmap).
        explicit PomaiArena(const pomai::config::PomaiConfig& cfg);

        ~PomaiArena();

        // Non-copyable
        PomaiArena(const PomaiArena &) = delete;
        PomaiArena &operator=(const PomaiArena &) = delete;

        // Movable (moves ownership). Note: moving an arena is uncommon.
        PomaiArena(PomaiArena &&other) noexcept;
        PomaiArena &operator=(PomaiArena &&other) noexcept;

        // Create and map a region of `bytes`. Returns true on success.
        bool allocate_region(uint64_t bytes);

        // Is arena valid / mapped?
        bool is_valid() const noexcept;

        // Seed RNG for reproducible sampling.
        void seed_rng(uint64_t seed);

        // Total capacity (mapped bytes)
        uint64_t get_capacity_bytes() const noexcept;

        // ---------------- Seed allocation API ----------------
        // Allocate a Seed slot from seed region. Returns nullptr on OOM.
        // The returned Seed* is zero-initialized (header==0) and is considered "active"
        // until free_seed() is called for that pointer.
        Seed *alloc_seed();

        // Free a previously allocated Seed (returned by alloc_seed()). If s==nullptr no-op.
        void free_seed(Seed *s);

        // Number of active seeds (allocated and not freed)
        uint64_t num_active_seeds() const;

        // Randomly sample active seed. Returns nullptr if none available.
        Seed *get_random_seed();

        // ---------------- Blob helpers ------------------------------
        // Allocate a blob with 4-byte length header: layout [uint32_t len][data...][\0]
        // Returns pointer to the start of the header (i.e. header pointer), or nullptr on OOM.
        // The returned pointer is inside the arena's blob region.
        char *alloc_blob(uint32_t len);

        // Free a blob previously returned by alloc_blob (pass the header pointer).
        // Safe to call with nullptr (no-op).
        void free_blob(char *header_ptr);

        // Convert blob offset (relative to blob_base) to pointer. Returns nullptr on error.
        // This supports both local offsets (< blob_region_bytes) and remote ids (>= blob_region_bytes)
        // for which the function will mmap the remote file lazily and return the mapping address.
        // Returned pointer points to the start of the file/memory (i.e. beginning of uint32_t length header).
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const;

        // Convert a pointer inside the blob region to an offset relative to blob_base.
        // Returns UINT64_MAX on error.
        uint64_t offset_from_blob_ptr(const char *p) const noexcept;

        // Demote a local in-arena blob (given by local offset) to disk.
        // On success returns a remote_id >= blob_region_bytes that caller should store into payload.
        // Returns 0 on failure.
        uint64_t demote_blob(uint64_t local_offset);

        // Demote raw data (already packed: [uint32_t len][payload...][\0]) directly to a remote file.
        // This is used as a fallback when we cannot allocate a blob inside the arena.
        // Returns remote_id (> blob_region_bytes_) on success, or 0 on failure.
        uint64_t demote_blob_data(const char *data_with_header, uint32_t total_bytes);

        // Asynchronous demote: schedule data to be written to remote store by background worker.
        // Returns a placeholder id (MSB set) on success immediately (file write happens asynchronously).
        // If the internal demote queue is full the call will fall back to synchronous demote (writes inline).
        // Returns 0 on failure.
        uint64_t demote_blob_async(const char *data_with_header, uint32_t total_bytes);

        // New overload: accept void* (generic) for callers that already have packed blob bytes.
        // Returns placeholder id (MSB set) or final remote id on sync fallback / failure (0).
        uint64_t demote_blob_async(const void *data, uint32_t len);

        // Promote a remote id back into arena; returns new local offset on success,
        // UINT64_MAX on failure.
        uint64_t promote_remote(uint64_t remote_id);

        // [NEW] Read remote blob content into a temporary buffer (RAM).
        // Used for on-the-fly search of frozen buckets without permanent promotion.
        // Returns a vector containing the full blob (header + payload). Empty on failure.
        std::vector<char> read_remote_blob(uint64_t remote_id) const;

        // Resolve pending placeholder remote id to final remote id if the async demote completed.
        // Returns 0 if not yet completed or if not found.
        uint64_t resolve_pending_remote(uint64_t placeholder_remote_id) const;

        // Resolve placeholder with optional wait: if maybe_placeholder is a placeholder id,
        // wait up to timeout_ms for completion and return final remote id (or 0 on timeout/failure).
        uint64_t resolve_pending_remote(uint64_t maybe_placeholder, uint64_t timeout_ms);

        // Demote queue metrics
        size_t get_demote_queue_length() const noexcept;
        void set_demote_queue_max(size_t max_pending);
        size_t get_demote_queue_max() const noexcept;

        // Quick accessors for region pointers (debug / external use)
        const char *blob_base_ptr() const noexcept;
        uint64_t blob_region_size() const noexcept;
        const char *seed_base_ptr() const noexcept;
        uint64_t seed_region_size() const noexcept;

        // ---------------- Factories for tests / convenience ------------
        // Create arena from MB/GB; returns a PomaiArena object (invalid if allocation failed).
        static PomaiArena FromMB(uint64_t mb);
        static PomaiArena FromGB(double gb);

        // Create arena sized according to runtime configuration (POMAI_ARENA_MB).
        // Reads pomai::config::runtime.arena_mb_per_shard and returns an arena sized to that value
        // (or a sensible default if it is zero).
        static PomaiArena FromConfig(const pomai::config::PomaiConfig &cfg);

    private:
        pomai::config::ArenaConfig cfg_;

        size_t max_remote_mmaps_;
        size_t max_pending_demotes_;
        size_t demote_batch_bytes_;
        std::string remote_dir_;

        // Internal helpers
        uint64_t block_size_for(uint64_t bytes); // round up to power-of-two bucket (>= min)
        std::string generate_remote_filename(uint64_t id) const;
        void cleanup();

        // Mmap'ed base region and partition offsets
        char *base_addr_;
        uint64_t capacity_bytes_;

        // seed region [base_addr_ .. base_addr_ + seed_region_bytes_)
        char *seed_base_;
        uint64_t seed_region_bytes_;
        uint64_t seed_max_slots_;
        uint64_t seed_next_slot_; // bump allocator index

        // blob region [blob_base_ .. blob_base_ + blob_region_bytes_)
        char *blob_base_;
        uint64_t blob_region_bytes_;
        uint64_t blob_next_offset_; // bump pointer (bytes) within blob region

        // freelists and active-seed tracking
        std::vector<uint64_t> free_seeds_;   // indices of freed seed slots (LIFO)
        std::vector<uint64_t> active_seeds_; // list of active seed indices for random sampling
        std::vector<uint64_t> active_pos_;   // mapping slot idx -> position in active_seeds_ (UINT64_MAX == not active)

        // slab freelists keyed by block size
        std::unordered_map<uint64_t, std::vector<uint64_t>> free_lists_;

        // remote storage: remote_id -> filepath (empty string means pending)
        std::unordered_map<uint64_t, std::string> remote_map_;

        // cached mmap of remote files (lazy mappings) with LRU eviction
        mutable std::unordered_map<uint64_t, std::pair<const char *, size_t>> remote_mmaps_;
        mutable std::list<uint64_t> remote_mmap_lru_; // front = most recent
        mutable std::unordered_map<uint64_t, std::list<uint64_t>::iterator> remote_mmap_iter_;

        // remote id counter (starts at 1); remote_id = blob_region_bytes_ + next_remote_id_
        uint64_t next_remote_id_;

        // pending async demote id counter (for diagnostics; remote ids reserved via next_remote_id_)
        // Use atomic so async path can reserve placeholders without external locking.
        std::atomic<uint64_t> pending_counter_{1};

        // base value used for placeholder generation (if desired); kept for diagnostics/compatibility.
        // (arena.cc computes and sets this during allocate_region)
        uint64_t pending_base_{0};

        // random generator for sampling
        mutable std::mt19937_64 rng_;

        // coarse-grained mutex for all mutable state
        mutable std::mutex mu_;

        // ----------------- Async demote bookkeeping & helpers --------------
        // pending_map_ holds placeholders -> PendingDemote shared state so callers can wait.
        mutable std::mutex pending_mu_;
        std::unordered_map<uint64_t, std::shared_ptr<PendingDemote>> pending_map_;

        // DemoteTask: now carries placeholder + pending ptr so worker can fulfill promise
        struct DemoteTask
        {
            uint64_t remote_id;                  // reserved remote id assigned at enqueue time (encoded)
            uint64_t placeholder;                // placeholder id returned to caller (MSB set) or 0
            std::vector<char> payload;           // copy of [uint32_t len][data...][\0]
            std::shared_ptr<PendingDemote> pend; // optional pending handle (nullptr if sync fallback)
        };

        // ----------------- Async demote worker ------------------
        mutable std::mutex demote_mu_; // protects demote_queue_ and worker condition
        std::condition_variable demote_cv_;
        std::deque<DemoteTask> demote_queue_;
        std::thread demote_worker_;
        std::atomic<bool> demote_worker_running_{false};

        size_t demote_segment_size_{512 * 1024 * 1024}; // not used yet; reserved
    };

} // namespace pomai::memory