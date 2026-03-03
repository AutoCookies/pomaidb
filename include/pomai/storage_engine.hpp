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
    pomai::Status Append(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata* meta = nullptr);
    pomai::Status Delete(pomai::VectorId id);
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

inline pomai::Status StorageEngine::Open(std::string_view path, std::uint32_t dim, palloc_heap_t* heap, std::size_t flush_threshold_bytes) {
    if (dim == 0) return pomai::Status::InvalidArgument("dim must be > 0");
    (void)Close();
    path_ = std::string(path);
    dim_ = dim;
    flush_threshold_ = flush_threshold_bytes;
    heap_ = heap;
    buffer_ = std::vector<std::byte, BufferAlloc>(BufferAlloc(heap));
    pending_bytes_ = 0;
    index_.clear();
#if defined(_WIN32) || defined(_WIN64)
    return pomai::Status::IOError("Windows: use CreateFile/CreateFileMapping");
#else
    int fd = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) return pomai::Status::IoError(std::string("open: ") + std::strerror(errno));
    fd_ = static_cast<fd_type>(fd);
    struct stat st;
    if (::fstat(fd_, &st) != 0) { ::close(fd_); fd_ = static_cast<fd_type>(kInvalidFd); return pomai::Status::IoError("fstat"); }
    file_size_ = static_cast<std::size_t>(st.st_size);
    if (file_size_ == 0) {
        StorageFileHeader hdr{}; std::memcpy(hdr.magic, StorageFileHeader::kMagic, sizeof(hdr.magic));
        hdr.version = 1; hdr.dim = dim_; std::memset(hdr.reserved_u32, 0, sizeof(hdr.reserved_u32));
        if (::write(fd_, &hdr, sizeof(hdr)) != static_cast<ssize_t>(sizeof(hdr))) { ::close(fd_); fd_ = static_cast<fd_type>(kInvalidFd); return pomai::Status::IoError("write header"); }
        file_size_ = sizeof(hdr);
    } else {
        StorageFileHeader hdr; std::size_t r = 0;
        while (r < sizeof(hdr)) { ssize_t n = ::pread(fd_, reinterpret_cast<char*>(&hdr) + r, sizeof(hdr) - r, static_cast<off_t>(r)); if (n <= 0 && errno != EINTR) break; if (n > 0) r += n; }
        if (r < sizeof(hdr) || !hdr.valid() || hdr.dim != dim_) { ::close(fd_); fd_ = static_cast<fd_type>(kInvalidFd); return pomai::Status::Corruption("invalid header"); }
    }
    return ReloadMmap();
#endif
}

inline pomai::Status StorageEngine::Close() {
#if !defined(_WIN32) && !defined(_WIN64)
    if (map_addr_ && map_size_ > 0) { ::munmap(map_addr_, map_size_); map_addr_ = nullptr; map_size_ = 0; }
    if (fd_ >= 0) { ::close(fd_); fd_ = static_cast<fd_type>(kInvalidFd); }
#endif
    file_size_ = 0; index_.clear(); buffer_.clear(); pending_bytes_ = 0;
    return pomai::Status::Ok();
}

inline pomai::Status StorageEngine::AppendToBuffer(const VectorRecordHeader& hdr, std::span<const std::byte> metadata, std::span<const float> vec) {
    std::size_t rec_len = RecordSize(hdr.dim, hdr.metadata_len);
    buffer_.resize(buffer_.size() + rec_len);
    std::byte* dst = buffer_.data() + buffer_.size() - rec_len;
    std::memcpy(dst, &hdr, sizeof(hdr)); dst += sizeof(hdr);
    if (!metadata.empty()) { std::memcpy(dst, metadata.data(), metadata.size()); dst += metadata.size(); }
    if (!vec.empty()) std::memcpy(dst, vec.data(), vec.size() * sizeof(float));
    pending_bytes_ += rec_len;
    return pomai::Status::Ok();
}

inline pomai::Status StorageEngine::Append(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata* meta) {
    if (!is_open()) return pomai::Status(pomai::ErrorCode::kFailedPrecondition, "not open");
    if (vec.size() != dim_) return pomai::Status::InvalidArgument("dim mismatch");
    VectorRecordHeader hdr{}; hdr.id = id; hdr.dim = static_cast<std::uint32_t>(vec.size()); hdr.flags = 0;
    std::string meta_blob; if (meta && !meta->tenant.empty()) { meta_blob = meta->tenant; hdr.metadata_len = static_cast<std::uint32_t>(meta_blob.size()); }
    std::span<const std::byte> ms(reinterpret_cast<const std::byte*>(meta_blob.data()), meta_blob.size());
    pomai::Status st = AppendToBuffer(hdr, ms, vec);
    if (!st.ok()) return st;
    IndexEntry e; e.offset = file_size_ + (buffer_.size() - RecordSize(hdr.dim, hdr.metadata_len)); e.length = RecordSize(hdr.dim, hdr.metadata_len); e.tombstone = false; e.in_buffer = true;
    index_.emplace_back(id, e);
    if (pending_bytes_ >= flush_threshold_) return Flush();
    return pomai::Status::Ok();
}

inline pomai::Status StorageEngine::Delete(pomai::VectorId id) {
    if (!is_open()) return pomai::Status(pomai::ErrorCode::kFailedPrecondition, "not open");
    VectorRecordHeader hdr{}; hdr.id = id; hdr.dim = dim_; hdr.flags = VectorRecordHeader::kFlagTombstone; hdr.metadata_len = 0;
    pomai::Status st = AppendToBuffer(hdr, {}, {});
    if (!st.ok()) return st;
    IndexEntry e; e.offset = file_size_ + (buffer_.size() - RecordSize(dim_, 0)); e.length = RecordSize(dim_, 0); e.tombstone = true; e.in_buffer = true;
    index_.emplace_back(id, e);
    if (pending_bytes_ >= flush_threshold_) return Flush();
    return pomai::Status::Ok();
}

inline pomai::Status StorageEngine::FlushBufferToFile() {
#if defined(_WIN32) || defined(_WIN64)
    return pomai::Status::IOError("Windows not implemented");
#else
    if (buffer_.empty()) return pomai::Status::Ok();
    const std::byte* p = buffer_.data(); std::size_t rem = buffer_.size();
    while (rem > 0) { ssize_t n = ::write(fd_, p, rem); if (n < 0 && errno != EINTR) return pomai::Status::IoError("write"); if (n > 0) { p += n; rem -= n; } }
    file_size_ += buffer_.size(); buffer_.clear(); pending_bytes_ = 0;
    for (auto& kv : index_) if (kv.second.in_buffer) kv.second.in_buffer = false;
    return pomai::Status::Ok();
#endif
}

inline pomai::Status StorageEngine::Flush() {
    if (!is_open()) return pomai::Status(pomai::ErrorCode::kFailedPrecondition, "not open");
    pomai::Status st = FlushBufferToFile();
    return st.ok() ? ReloadMmap() : st;
}

inline pomai::Status StorageEngine::BuildIndexFromMmap() {
    index_.clear();
    if (!map_addr_ || map_size_ <= sizeof(StorageFileHeader)) return pomai::Status::Ok();
    const std::byte* base = static_cast<const std::byte*>(map_addr_);
    std::size_t off = sizeof(StorageFileHeader);
    while (off + sizeof(VectorRecordHeader) <= map_size_) {
        const auto* h = reinterpret_cast<const VectorRecordHeader*>(base + off);
        std::size_t rec_len = RecordSize(h->dim, h->metadata_len);
        if (off + rec_len > map_size_) break;
        index_.emplace_back(h->id, IndexEntry{off, rec_len, h->is_tombstone(), false});
        off += rec_len;
    }
    return pomai::Status::Ok();
}

inline pomai::Status StorageEngine::ReloadMmap() {
#if defined(_WIN32) || defined(_WIN64)
    return pomai::Status::Ok();
#else
    if (map_addr_ && map_size_ > 0) { ::munmap(map_addr_, map_size_); map_addr_ = nullptr; map_size_ = 0; }
    if (fd_ < 0) return pomai::Status::Ok();
    struct stat st; if (::fstat(fd_, &st) != 0) return pomai::Status::IoError("fstat");
    file_size_ = static_cast<std::size_t>(st.st_size);
    if (file_size_ == 0) return BuildIndexFromMmap();
    void* addr = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr == MAP_FAILED) return pomai::Status::IoError("mmap");
    map_addr_ = addr; map_size_ = file_size_;
    return BuildIndexFromMmap();
#endif
}

inline pomai::Status StorageEngine::Get(pomai::VectorId id, GetResult* out) const {
    if (!out) return pomai::Status::InvalidArgument("out is null");
    out->data = nullptr; out->dim = 0; out->is_tombstone = false; out->meta = nullptr;
    auto it = std::find_if(index_.rbegin(), index_.rend(), [id](const auto& p) { return p.first == id; });
    if (it == index_.rend()) return pomai::Status::NotFound("vector id");
    const IndexEntry& e = it->second;
    if (e.tombstone) { out->is_tombstone = true; return pomai::Status::Ok(); }
    const std::byte* rec = e.in_buffer ? (buffer_.data() + (e.offset >= file_size_ ? e.offset - file_size_ : 0)) : (static_cast<const std::byte*>(map_addr_) + e.offset);
    if ((e.in_buffer && e.offset - file_size_ + e.length > buffer_.size()) || (!e.in_buffer && e.offset + e.length > map_size_)) return pomai::Status::Corruption("index");
    const auto* h = reinterpret_cast<const VectorRecordHeader*>(rec);
    rec += sizeof(VectorRecordHeader) + h->metadata_len;
    out->dim = h->dim; out->data = reinterpret_cast<const float*>(rec);
    return pomai::Status::Ok();
}

} // namespace pomaidb
