// pomai/storage_engine.hpp — Append-only log-structured vector storage engine.
//
// UNIVERSAL I/O PHILOSOPHY (SD-Card First, SSD Compatible):
// Optimize for the worst-case (MicroSD/eMMC) so that design naturally dominates
// the best-case (NVMe). MicroSD/eMMC have catastrophic random-write performance
// and wear-leveling limits; NVMe suffers from write amplification under random I/O.
// This engine eliminates random I/O entirely: append-only writes, sequential flushes,
// zero-copy reads via mmap. Maximum throughput and lifespan on any block device.
//
// - Zero random writes: no fseek/seekp; new data appended; deletes/updates = tombstone.
// - RAM buffer (PomaiArenaAllocator / palloc) flushed in one sequential write (e.g. 32MB).
// - Zero-copy reads: mmap (Linux) / CreateFileMapping (Windows); no std::ifstream.
// - 64-byte-aligned record layout for SIMD/AVX regardless of host architecture.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pomai/metadata.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "palloc_compat.h"

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pomaidb {

constexpr std::size_t kStorageAlign = 64u;

template <typename T>
class PomaiArenaAllocator {
public:
    using value_type = T;
    explicit PomaiArenaAllocator(palloc_heap_t* heap = nullptr) noexcept : heap_(heap) {}
    template <typename U>
    PomaiArenaAllocator(const PomaiArenaAllocator<U>& other) noexcept : heap_(other.heap()) {}
    palloc_heap_t* heap() const noexcept { return heap_; }
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        std::size_t size = n * sizeof(T);
        std::size_t align = alignof(T) < kStorageAlign ? kStorageAlign : alignof(T);
        void* p = heap_ ? palloc_heap_malloc_aligned(heap_, size, align) : palloc_malloc_aligned(size, align);
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { if (p) palloc_free(p); }
    template <typename U>
    bool operator==(const PomaiArenaAllocator<U>& other) const noexcept { return heap_ == other.heap(); }
private:
    palloc_heap_t* heap_;
};

// Arena-backed vector type for buffered inserts (e.g. 1536-dim embeddings).
template <typename T = float>
using ArenaVector = std::vector<T, PomaiArenaAllocator<T>>;

struct alignas(kStorageAlign) StorageFileHeader {
    static constexpr char kMagic[12] = {'P','O','M','A','I','_','L','O','G','v','1','\0'};
    char magic[12];
    std::uint32_t version;
    std::uint32_t dim;
    std::uint32_t reserved_u32[10];
    bool valid() const noexcept {
        return std::memcmp(magic, kMagic, sizeof(kMagic)) == 0 && version == 1u && dim > 0u;
    }
};
static_assert(sizeof(StorageFileHeader) == 64u, "File header 64 bytes");

// Vector record: 64-byte header then payload (metadata blob + dim*sizeof(float)).
// Supports any fixed dim per segment (e.g. 1536). Tombstone flag for deletes/updates.
struct alignas(kStorageAlign) VectorRecordHeader {
    static constexpr std::uint32_t kFlagTombstone = 1u;
    pomai::VectorId id;
    std::uint32_t dim;
    std::uint32_t flags;           // bit0 = tombstone
    std::uint32_t metadata_len;
    std::uint32_t reserved[11];
    bool is_tombstone() const noexcept { return (flags & kFlagTombstone) != 0; }
};
static_assert(sizeof(VectorRecordHeader) == 64u, "Record header 64 bytes");

inline std::size_t RecordSize(std::uint32_t dim, std::uint32_t metadata_len) {
    std::size_t payload = metadata_len + static_cast<std::size_t>(dim) * sizeof(float);
    return ((sizeof(VectorRecordHeader) + payload) + kStorageAlign - 1u) & ~(kStorageAlign - 1u);
}

class StorageEngine {
public:
    static constexpr std::size_t kDefaultFlushThreshold = 32u * 1024u * 1024u;
    StorageEngine() = default;
    ~StorageEngine() { (void)Close(); }
    StorageEngine(const StorageEngine&) = delete;
    StorageEngine& operator=(const StorageEngine&) = delete;

    pomai::Status Open(std::string_view path, std::uint32_t dim, palloc_heap_t* heap = nullptr,
                      std::size_t flush_threshold_bytes = kDefaultFlushThreshold);
    pomai::Status Close();
    // Append: pushes to in-memory buffer only. Call Flush() explicitly from main loop.
    pomai::Status Append(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata* meta = nullptr);
    pomai::Status Delete(pomai::VectorId id);
    // Flush: called explicitly by main loop; writes entire buffer to disk sequentially.
    pomai::Status Flush();

    struct GetResult {
        const float* data = nullptr;
        std::uint32_t dim = 0;
        bool is_tombstone = false;
        const pomai::Metadata* meta = nullptr;
    };
    pomai::Status Get(pomai::VectorId id, GetResult* out) const;
    pomai::Status ReloadMmap();

    std::uint32_t dim() const noexcept { return dim_; }
    bool is_open() const noexcept { return fd_ >= 0; }
    std::size_t pending_bytes() const noexcept { return pending_bytes_; }

private:
    pomai::Status AppendToBuffer(const VectorRecordHeader& hdr, std::span<const std::byte> metadata, std::span<const float> vec);
    pomai::Status FlushBufferToFile();
    pomai::Status BuildIndexFromMmap();

#if defined(_WIN32) || defined(_WIN64)
    using fd_type = void*;
    static constexpr int kInvalidFd = 0;
#else
    using fd_type = int;
    static constexpr int kInvalidFd = -1;
#endif

    std::string path_;
    fd_type fd_ = static_cast<fd_type>(kInvalidFd);
    std::uint32_t dim_ = 0;
    std::size_t flush_threshold_ = kDefaultFlushThreshold;
    std::size_t file_size_ = 0;
    using BufferAlloc = PomaiArenaAllocator<std::byte>;
    std::vector<std::byte, BufferAlloc> buffer_;
    std::size_t pending_bytes_ = 0;
    void* map_addr_ = nullptr;
    std::size_t map_size_ = 0;
    struct IndexEntry { std::size_t offset; std::size_t length; bool tombstone; bool in_buffer; };
    std::vector<std::pair<pomai::VectorId, IndexEntry>> index_;
    palloc_heap_t* heap_ = nullptr;
};

// Implementation in src/storage/storage_engine.cc (flush-threshold, fsync, logging).

} // namespace pomaidb
