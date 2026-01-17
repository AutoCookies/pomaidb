#pragma once
// src/memory/shard_arena.h

#include <cstdint>
#include <atomic>
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>
#include "src/core/config.h" // Bat buoc phai co de dung PomaiConfig

namespace pomai::memory
{
    class ShardArena
    {
    public:
        // Constructor nhan config
        ShardArena(uint32_t shard_id, size_t capacity_bytes, const pomai::config::PomaiConfig& cfg);
        ~ShardArena();

        ShardArena(const ShardArena &) = delete;
        ShardArena &operator=(const ShardArena &) = delete;
        ShardArena(ShardArena &&) = delete;
        ShardArena &operator=(ShardArena &&) = delete;

        char *alloc_blob(uint32_t len);
        uint64_t offset_from_blob_ptr(const char *ptr) const noexcept;
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const noexcept;

        uint64_t demote_blob(uint64_t local_offset);
        std::vector<char> read_remote_blob(uint64_t remote_id) const;
        void demote_range(uint64_t offset, size_t len);

        size_t used_bytes() const noexcept { return static_cast<size_t>(write_head_.load(std::memory_order_relaxed)); }
        size_t capacity() const noexcept { return capacity_; }
        uint32_t id() const noexcept { return id_; }
        void reset() noexcept;

    private:
        std::string generate_remote_filename(uint64_t encoded_remote_id) const;

        uint32_t id_;
        size_t capacity_;
        char *base_addr_;

        alignas(64) std::atomic<uint64_t> write_head_;

        mutable std::mutex remote_mu_;
        std::string remote_dir_;
        
        // [FIXED] Them cac bien bi thieu de khop voi shard_arena.cc
        size_t max_remote_mmaps_; 
        std::atomic<uint64_t> next_remote_id_;
        
        mutable std::unordered_map<uint64_t, std::pair<const char *, size_t>> remote_mmaps_;
        size_t page_size_;

        static inline size_t align_up(size_t v, size_t a) noexcept { return (v + a - 1) & ~(a - 1); }
    };

} // namespace pomai::memory