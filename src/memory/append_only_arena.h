#pragma once
// src/memory/append_only_arena.h
//
// Thread-safe append-only arena backed by a file + mmap.
// - alloc_blob(len) reserves (4 bytes length header) + payload bytes and returns pointer to blob header.
// - offset_from_blob_ptr(ptr) returns uint64 offset (for publishing to index).
// - blob_ptr_from_offset_for_map(offset) returns pointer into mapping (or nullptr if not mapped / out-of-range).
// - persist/flush support msync for durability.
// - Grows underlying file as needed (posix_fallocate or ftruncate).
//
// Production considerations:
//  - Use careful locking on remap/grow, atomic fetch for offsets when under mutex-free fast path not needed.
//  - Caller is responsible for publishing offset atomically to shared metadata (e.g. PPE payload).
//  - This arena is append-only; no free() provided. GC/compaction needs to be built at higher layer.

#include <string>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <cstddef>

namespace pomai::memory
{

    class AppendOnlyArena
    {
    public:
        // Create or open an arena file. initial_size_bytes must be > 0 (rounded up to page size).
        // Throws std::runtime_error on failure.
        static AppendOnlyArena *OpenOrCreate(const std::string &path, size_t initial_size_bytes);

        // Close and destroy object (unmaps and closes file).
        ~AppendOnlyArena();

        // No copy
        AppendOnlyArena(const AppendOnlyArena &) = delete;
        AppendOnlyArena &operator=(const AppendOnlyArena &) = delete;

        // Reserve a blob of payload_size bytes (payload only). Returns pointer to the blob header in mapped region:
        //  blob_hdr[0..3] = uint32_t payload_size (little-endian), payload bytes follow at blob_hdr + 4.
        // Returned pointer is valid until a remap happens (i.e., until resize/grow). Caller must hold onto offset if needed.
        // Throws std::runtime_error on failure.
        void *alloc_blob(uint32_t payload_size);

        // Convert a blob header pointer previously returned by alloc_blob into an arena offset (uint64).
        // Offset is measured from mapping base (i.e., pointer = base + offset).
        uint64_t offset_from_blob_ptr(const void *blob_hdr) const noexcept;

        // Given an offset previously returned by offset_from_blob_ptr or stored in index, return pointer in current map.
        // If offset points beyond current mapping / not yet mapped, return nullptr.
        const char *blob_ptr_from_offset_for_map(uint64_t offset) const noexcept;

        // Ensure mapping is persisted (msync). Returns true on success.
        bool persist_range(uint64_t offset, size_t len, bool synchronous = false) const noexcept;

        // Return capacity (mapped size) in bytes
        size_t get_capacity_bytes() const noexcept { return mapped_size_; }

        // Attempt to grow underlying file to at least new_size bytes (rounded to page). Called internally.
        // Returns true on success.
        bool grow_to(size_t new_size);

        // Path
        std::string path() const noexcept { return path_; }

    private:
        // Private constructor; use OpenOrCreate
        AppendOnlyArena(int fd, void *map_base, size_t mapped_size, const std::string &path, size_t page_size);

        // internal remap (requires lock)
        bool remap_locked(size_t new_size);

        int fd_;
        void *map_base_;
        size_t mapped_size_;
        std::atomic<uint64_t> write_offset_; // next free offset inside mapping
        mutable std::mutex grow_mu_;         // protects grow/remap and alloc path when remapping
        std::string path_;
        size_t page_size_;
    };

} // namespace pomai::memory