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
// This header declares the public API. Implementation is in arena.cc.
//
// Design notes
//  - This is a pragmatic single-process, single-machine prototype design.
//    Remote/demoted blobs are written to files in a configurable directory (defaults to /tmp).
//  - All mutable state is protected by a single mutex (mu) to keep correctness simple.
//    This is easy to reason about, though coarse-grained. Optimize later if needed.
//  - Remote ids are encoded as: remote_id = blob_region_bytes + small_counter.
//    That guarantees they never overlap with local offsets (which are < blob_region_bytes).
//
// Thread safety: all public methods that mutate or read mutable state grab `mu`.
// Methods that return raw pointers (blob_ptr_from_offset_for_map) return pointers
// that remain valid until the arena is destroyed or promote_remote removes the mapping.
// Users must not modify or munmap those pointers.

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <mutex>
#include <optional>

struct Seed; // forward-declare; concrete definition in user's code (seed.h)

namespace pomai::memory
{

    class PomaiArena
    {
    public:
        // Construct an empty (invalid) arena. Must call allocate_region() or use factory.
        PomaiArena();

        // Construct and allocate `bytes` bytes immediately (mmap).
        explicit PomaiArena(uint64_t bytes);

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

        // Promote a remote id back into arena; returns new local offset on success,
        // UINT64_MAX on failure.
        uint64_t promote_remote(uint64_t remote_id);

        // Quick accessors for region pointers (debug / external use)
        const char *blob_base_ptr() const noexcept;
        uint64_t blob_region_size() const noexcept;
        const char *seed_base_ptr() const noexcept;
        uint64_t seed_region_size() const noexcept;

        // ---------------- Factories for tests / convenience ------------
        // Create arena from MB/GB; returns a PomaiArena object (invalid if allocation failed).
        static PomaiArena FromMB(uint64_t mb);
        static PomaiArena FromGB(double gb);

    private:
        // Internal helpers
        static uint64_t block_size_for(uint64_t bytes); // round up to power-of-two bucket (>= min)
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

        // remote storage: remote_id -> filepath
        std::unordered_map<uint64_t, std::string> remote_map_;

        // cached mmap of remote files (lazy mappings)
        mutable std::unordered_map<uint64_t, std::pair<const char *, size_t>> remote_mmaps_;

        // remote id counter (starts at 1); remote_id = blob_region_bytes_ + next_remote_id_
        uint64_t next_remote_id_;

        // random generator for sampling
        mutable std::mt19937_64 rng_;

        // coarse-grained mutex for all mutable state
        mutable std::mutex mu_;

        // configuration
        static constexpr double SEED_REGION_RATIO = 0.25; // fraction of capacity reserved for seeds
        static constexpr uint64_t MIN_BLOB_BLOCK = 64;
        static constexpr size_t MAX_FREELIST_PER_BUCKET = 4096;

        // directory where demoted blobs are written (can be changed if necessary)
        std::string remote_dir_ = "/tmp";
    };

} // namespace pomai::memory

// Backwards compatibility: many files expect an unqualified PomaiArena type.
using PomaiArena = pomai::memory::PomaiArena;