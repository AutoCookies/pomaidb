#pragma once

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
        PomaiArena();
        explicit PomaiArena(const pomai::config::PomaiConfig &cfg);
        ~PomaiArena();

        PomaiArena(const PomaiArena &) = delete;
        PomaiArena &operator=(const PomaiArena &) = delete;

        PomaiArena(PomaiArena &&other) noexcept;
        PomaiArena &operator=(PomaiArena &&other) noexcept;

        // Create and map a region of `bytes`. Returns true on success.
        bool allocate_region(uint64_t bytes);

        bool is_valid() const noexcept;
        void seed_rng(uint64_t seed);
        uint64_t get_capacity_bytes() const noexcept;

        // ---------------- Seed allocation API ----------------
        Seed *alloc_seed();
        void free_seed(Seed *s);
        uint64_t num_active_seeds() const;
        Seed *get_random_seed();

        // ---------------- Blob helpers ------------------------------
        char *alloc_blob(uint32_t len);
        void free_blob(char *header_ptr);
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const;
        uint64_t offset_from_blob_ptr(const char *p) const noexcept;
        uint64_t demote_blob(uint64_t local_offset);
        uint64_t demote_blob_data(const char *data_with_header, uint32_t total_bytes);
        uint64_t demote_blob_async(const char *data_with_header, uint32_t total_bytes);
        uint64_t demote_blob_async(const void *data, uint32_t len);
        uint64_t promote_remote(uint64_t remote_id);
        std::vector<char> read_remote_blob(uint64_t remote_id) const;
        uint64_t resolve_pending_remote(uint64_t placeholder_remote_id) const;
        uint64_t resolve_pending_remote(uint64_t maybe_placeholder, uint64_t timeout_ms);

        size_t get_demote_queue_length() const noexcept;
        void set_demote_queue_max(size_t max_pending);
        size_t get_demote_queue_max() const noexcept;

        const char *blob_base_ptr() const noexcept;
        uint64_t blob_region_size() const noexcept;
        const char *seed_base_ptr() const noexcept;
        uint64_t seed_region_size() const noexcept;

        static PomaiArena FromMB(uint64_t mb);
        static PomaiArena FromGB(double gb);
        static PomaiArena FromConfig(const pomai::config::PomaiConfig &cfg);

    private:
        pomai::config::ArenaConfig cfg_;

        size_t max_remote_mmaps_;
        size_t max_pending_demotes_;
        size_t demote_batch_bytes_;
        std::string remote_dir_;

        // NEW: directory for arena backing file (if file-backed mmap enabled)
        std::string arena_backing_dir_;

        // Internal helpers
        uint64_t block_size_for(uint64_t bytes);
        std::string generate_remote_filename(uint64_t id) const;
        void cleanup();

        // Mmap'ed base region and partition offsets
        char *base_addr_;
        uint64_t capacity_bytes_;

        // Backing file fd + path (if file-backed). -1 means anonymous mapping.
        int backing_fd_;
        std::string backing_file_path_;

        // seed region [base_addr_ .. base_addr_ + seed_region_bytes_)
        char *seed_base_;
        uint64_t seed_region_bytes_;
        uint64_t seed_max_slots_;
        uint64_t seed_next_slot_;

        // blob region [blob_base_ .. blob_base_ + blob_region_bytes_)
        char *blob_base_;
        uint64_t blob_region_bytes_;
        uint64_t blob_next_offset_;

        // freelists and active-seed tracking
        std::vector<uint64_t> free_seeds_;
        std::vector<uint64_t> active_seeds_;
        std::vector<uint64_t> active_pos_;

        std::unordered_map<uint64_t, std::vector<uint64_t>> free_lists_;

        // remote storage: remote_id -> filepath (empty string means pending)
        std::unordered_map<uint64_t, std::string> remote_map_;

        // cached mmap of remote files (lazy mappings) with LRU eviction
        mutable std::unordered_map<uint64_t, std::pair<const char *, size_t>> remote_mmaps_;
        mutable std::list<uint64_t> remote_mmap_lru_;
        mutable std::unordered_map<uint64_t, std::list<uint64_t>::iterator> remote_mmap_iter_;

        uint64_t next_remote_id_;
        std::atomic<uint64_t> pending_counter_{1};
        uint64_t pending_base_{0};

        mutable std::mt19937_64 rng_;

        mutable std::mutex mu_;

        // Async demote bookkeeping
        mutable std::mutex pending_mu_;
        std::unordered_map<uint64_t, std::shared_ptr<PendingDemote>> pending_map_;

        struct DemoteTask
        {
            uint64_t remote_id;
            uint64_t placeholder;
            std::vector<char> payload;
            std::shared_ptr<PendingDemote> pend;
        };

        mutable std::mutex demote_mu_;
        std::condition_variable demote_cv_;
        std::deque<DemoteTask> demote_queue_;
        std::thread demote_worker_;
        std::atomic<bool> demote_worker_running_{false};

        size_t demote_segment_size_{512 * 1024 * 1024};
    };

} // namespace pomai::memory