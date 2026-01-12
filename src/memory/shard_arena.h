#pragma once
// src/memory/shard_arena.h
//
// ShardArena: high-performance per-shard bump allocator optimized for
// single-writer use and lock-free readers.
//
// Design goals:
//  - Atomic bump pointer alloc_blob(): single atomic fetch_add (fast, wait-free).
//  - Readers can convert offsets -> pointer without locks (blob_ptr_from_offset_for_map).
//  - Support demote/read_remote for cold-storage (file-backed).
//  - Minimal locking only for remote-file bookkeeping (infrequent).
//
// Notes:
//  - Offsets returned by offset_from_blob_ptr are uint64_t offsets measured
//    from base_addr_. Offset==0 is reserved and treated as "null".
//  - Remote IDs are encoded with MSB (bit 63) set to 1. Encoded remote id
//    values map deterministically to file paths under remote_dir_.
//  - The allocator aligns allocations to 64 bytes for cache friendliness.
//  - This class expects the caller to ensure single-writer semantics for best perf.
//  - Thread-safe for concurrent readers and single writer. Some remote ops use a mutex.

#include <cstdint>
#include <atomic>
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>

namespace pomai::memory
{

class ShardArena
{
public:
    // Construct a shard arena with a given shard id and capacity in bytes.
    // Throws std::runtime_error on mmap failure.
    ShardArena(uint32_t shard_id, size_t capacity_bytes, const std::string &remote_dir = std::string());
    ~ShardArena();

    ShardArena(const ShardArena &) = delete;
    ShardArena &operator=(const ShardArena &) = delete;
    ShardArena(ShardArena &&) = delete;
    ShardArena &operator=(ShardArena &&) = delete;

    // Allocate a blob of 'len' payload bytes (payload only). Returned pointer
    // points to the start of the header (uint32_t len) within the mapped arena.
    // Returns nullptr on OOM.
    char *alloc_blob(uint32_t len);

    // Convert blob header pointer -> offset (relative to base). Returns UINT64_MAX on error.
    uint64_t offset_from_blob_ptr(const char *ptr) const noexcept;

    // Convert offset -> pointer (for map/index lookups).
    // - If offset==0 : returns nullptr (reserved).
    // - If offset has MSB set: treat as remote id and lazily mmap file and return pointer.
    // - If offset < capacity: returns base_addr_ + offset if offset < current_write_head (otherwise nullptr).
    // This function is safe for concurrent readers.
    const char *blob_ptr_from_offset_for_map(uint64_t offset) const noexcept;

    // Demote a local blob (offset) to a remote file (synchronous).
    // Returns encoded_remote_id (MSB set) on success, 0 on failure.
    uint64_t demote_blob(uint64_t local_offset);

    // Read remote blob file into RAM buffer (including header).
    // Returns empty vector on failure.
    std::vector<char> read_remote_blob(uint64_t remote_id) const;

    // Queryors
    size_t used_bytes() const noexcept { return static_cast<size_t>(write_head_.load(std::memory_order_relaxed)); }
    size_t capacity() const noexcept { return capacity_; }
    uint32_t id() const noexcept { return id_; }

    // Reset arena (dangerous in production; provided for tests)
    void reset() noexcept;

private:
    // Helper: produce deterministic filename for encoded remote id
    std::string generate_remote_filename(uint64_t encoded_remote_id) const;

    uint32_t id_;
    size_t capacity_;
    char *base_addr_;

    // Atomic bump pointer (offset in bytes). Align to cacheline to avoid false sharing.
    alignas(64) std::atomic<uint64_t> write_head_;

    // Remote file bookkeeping (protected by mutex)
    mutable std::mutex remote_mu_;
    std::string remote_dir_;
    // Cache of mmapped remote files: encoded_remote_id -> (addr, size)
    mutable std::unordered_map<uint64_t, std::pair<const char *, size_t>> remote_mmaps_;
    // Next remote id (incremental); actual encoded id will set MSB.
    std::atomic<uint64_t> next_remote_id_;

    // page size for alignment helpers
    size_t page_size_;

    // round-up helper
    static inline size_t align_up(size_t v, size_t a) noexcept { return (v + a - 1) & ~(a - 1); }
};

} // namespace pomai::memory