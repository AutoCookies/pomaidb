#pragma once
// src/memory/shard_arena.h
//
// ShardArena: file-backed memory arena using mmap(MAP_SHARED).
// - alloc_blob(len) reserves (4 bytes length header) + payload and returns pointer to blob header.
// - offset_from_blob_ptr(ptr) converts pointer -> offset.
// - blob_ptr_from_offset_for_map(offset) returns pointer if offset published/mapped.
// - demote_blob writes a single blob out to remote file and returns encoded remote id.
// - read_remote_blob reads remote file into RAM buffer.
// - demote_range calls madvise(..., MADV_DONTNEED) aligned to page boundaries.
// - persist_range(msync) flushes region to disk.
//
// Notes:
// - Constructor will create/open a backing blob file under remote_dir (by default data_root or /tmp).
// - The arena has a fixed initial capacity (map_size). grow_to() supports optional expansion.
#include <cstdint>
#include <atomic>
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>
#include "src/core/config.h" // must include for PomaiConfig

namespace pomai::memory
{
    class ShardArena
    {
    public:
        // Constructor: create (or open) a file-backed arena with capacity_bytes.
        // cfg.arena.remote_dir is used to place broken-out remote files; the backing file lives
        // under cfg.res.data_root/shard_<id>.blob if cfg.res.data_root is set, otherwise remote_dir.
        ShardArena(uint32_t shard_id, size_t capacity_bytes, const pomai::config::PomaiConfig& cfg);
        ~ShardArena();

        ShardArena(const ShardArena &) = delete;
        ShardArena &operator=(const ShardArena &) = delete;
        ShardArena(ShardArena &&) = delete;
        ShardArena &operator=(ShardArena &&) = delete;

        // Allocate a blob payload of `len` bytes. Returns pointer to blob header (4-byte len), or nullptr on OOM.
        char *alloc_blob(uint32_t len);

        // Convert blob header pointer to offset (relative to mapping base). Returns UINT64_MAX on error.
        uint64_t offset_from_blob_ptr(const char *ptr) const noexcept;

        // Given an offset (local offset or encoded remote id), return pointer for read (or nullptr).
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const noexcept;

        // Demote a local blob (at local_offset) into an external remote file; return encoded remote id (MSB set) or 0 on failure.
        uint64_t demote_blob(uint64_t local_offset);

        // Read remote blob into a RAM buffer (one-shot). Returns empty vector on failure.
        std::vector<char> read_remote_blob(uint64_t remote_id) const;

        // Advise OS that pages in [offset, offset+len) can be dropped from page cache (madvise MADV_DONTNEED).
        void demote_range(uint64_t offset, size_t len);

        // Force-msync a region [offset, offset+len). Returns true on success.
        bool persist_range(uint64_t offset, size_t len, bool synchronous = false) const noexcept;

        // Attempt to grow the backing file / mapping to at least new_size bytes. Returns true on success.
        bool grow_to(size_t new_size);

        size_t used_bytes() const noexcept { return static_cast<size_t>(write_head_.load(std::memory_order_relaxed)); }
        size_t capacity() const noexcept { return capacity_; }
        uint32_t id() const noexcept { return id_; }

        // Reset the arena (dangerous: invalidates offsets)
        void reset() noexcept;

    private:
        std::string generate_remote_filename(uint64_t encoded_remote_id) const;
        std::string backing_filename() const;

        uint32_t id_;
        size_t capacity_;        // current mapped size in bytes
        char *base_addr_;        // mmap base pointer
        int fd_;                 // backing file descriptor (-1 if not open)

        alignas(64) std::atomic<uint64_t> write_head_;

        // remote-file related
        mutable std::mutex remote_mu_;
        std::string remote_dir_;
        size_t max_remote_mmaps_;
        std::atomic<uint64_t> next_remote_id_;
        mutable std::unordered_map<uint64_t, std::pair<const char *, size_t>> remote_mmaps_;
        size_t page_size_;

        // helper
        static inline size_t align_up(size_t v, size_t a) noexcept { return (v + a - 1) & ~(a - 1); }
    };

} // namespace pomai::memory