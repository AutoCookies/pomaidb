// storage_engine.cc — Append-only log-structured StorageEngine implementation.
//
// Flush-and-Map cycle: ingestion into RAM buffer; when buffer >= flush_threshold,
// flush in one sequential write, fsync, then re-mmap for zero-copy reads.
// Single-threaded, no mutexes/atomics. Optimized for MicroSD endurance.

#include "pomai/storage_engine.hpp"

#include <cerrno>
#include <cstring>
#include <utility>

#include "util/logging.h"

#if !defined(_WIN32) && !defined(_WIN64)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pomaidb {

#if !defined(_WIN32) && !defined(_WIN64)

static pomai::Status ErrnoStatus(const char* op) {
    return pomai::Status::IOError(std::string(op) + ": " + std::strerror(errno));
}

pomai::Status StorageEngine::Open(std::string_view path, std::uint32_t dim,
                                  palloc_heap_t* heap,
                                  std::size_t flush_threshold_bytes) {
    if (dim == 0) return pomai::Status::InvalidArgument("dim must be > 0");
    (void)Close();

    path_ = std::string(path);
    dim_ = dim;
    flush_threshold_ = flush_threshold_bytes;
    heap_ = heap;
    buffer_ = std::vector<std::byte, BufferAlloc>(BufferAlloc(heap));
    pending_bytes_ = 0;
    index_.clear();

    int fd = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
        POMAI_LOG_ERROR("StorageEngine::Open open failed: {}", path_);
        return ErrnoStatus("open");
    }
    fd_ = static_cast<fd_type>(fd);

    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        fd_ = kInvalidFd;
        return ErrnoStatus("fstat");
    }
    file_size_ = static_cast<std::size_t>(st.st_size);

    if (file_size_ == 0) {
        StorageFileHeader hdr{};
        std::memcpy(hdr.magic, StorageFileHeader::kMagic, sizeof(hdr.magic));
        hdr.version = 1;
        hdr.dim = dim_;
        std::memset(hdr.reserved_u32, 0, sizeof(hdr.reserved_u32));

        ssize_t n = ::write(fd_, &hdr, sizeof(hdr));
        if (n != static_cast<ssize_t>(sizeof(hdr))) {
            ::close(fd_);
            fd_ = kInvalidFd;
            return ErrnoStatus("write header");
        }
        file_size_ = sizeof(hdr);
        POMAI_LOG_DEBUG("StorageEngine::Open created new file with header: {}", path_);
    } else {
        StorageFileHeader hdr;
        std::size_t r = 0;
        while (r < sizeof(hdr)) {
            ssize_t n = ::pread(fd_, reinterpret_cast<char*>(&hdr) + r,
                               sizeof(hdr) - r, static_cast<off_t>(r));
            if (n <= 0 && errno != EINTR) break;
            if (n > 0) r += static_cast<std::size_t>(n);
        }
        if (r < sizeof(hdr) || !hdr.valid() || hdr.dim != dim_) {
            ::close(fd_);
            fd_ = kInvalidFd;
            POMAI_LOG_ERROR("StorageEngine::Open invalid or mismatched header: {}", path_);
            return pomai::Status::Corruption("invalid header");
        }
    }

    POMAI_LOG_DEBUG("StorageEngine::Open path={} dim={} file_size={}", path_, dim_, file_size_);
    return ReloadMmap();
}

pomai::Status StorageEngine::Close() {
    if (map_addr_ && map_size_ > 0) {
        ::munmap(map_addr_, map_size_);
        map_addr_ = nullptr;
        map_size_ = 0;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = static_cast<fd_type>(kInvalidFd);
    }
    file_size_ = 0;
    index_.clear();
    buffer_.clear();
    pending_bytes_ = 0;
    POMAI_LOG_DEBUG("StorageEngine::Close");
    return pomai::Status::Ok();
}

pomai::Status StorageEngine::AppendToBuffer(const VectorRecordHeader& hdr,
                                            std::span<const std::byte> metadata,
                                            std::span<const float> vec) {
    const std::size_t rec_len = RecordSize(hdr.dim, hdr.metadata_len);
    buffer_.resize(buffer_.size() + rec_len);
    std::byte* dst = buffer_.data() + buffer_.size() - rec_len;
    std::memcpy(dst, &hdr, sizeof(hdr));
    dst += sizeof(hdr);
    if (!metadata.empty()) {
        std::memcpy(dst, metadata.data(), metadata.size());
        dst += metadata.size();
    }
    if (!vec.empty()) {
        std::memcpy(dst, vec.data(), vec.size() * sizeof(float));
    }
    pending_bytes_ += rec_len;
    return pomai::Status::Ok();
}

pomai::Status StorageEngine::Append(pomai::VectorId id, std::span<const float> vec,
                                    const pomai::Metadata* meta) {
    if (!is_open()) return pomai::Status(pomai::ErrorCode::kFailedPrecondition, "not open");
    if (vec.size() != dim_) return pomai::Status::InvalidArgument("dim mismatch");

    VectorRecordHeader hdr{};
    hdr.id = id;
    hdr.dim = static_cast<std::uint32_t>(vec.size());
    hdr.flags = 0;
    std::string meta_blob;
    if (meta && !meta->tenant.empty()) {
        meta_blob = meta->tenant;
        hdr.metadata_len = static_cast<std::uint32_t>(meta_blob.size());
    }
    std::span<const std::byte> ms(reinterpret_cast<const std::byte*>(meta_blob.data()),
                                  meta_blob.size());

    pomai::Status st = AppendToBuffer(hdr, ms, vec);
    if (!st.ok()) return st;

    const std::size_t rec_len = RecordSize(hdr.dim, hdr.metadata_len);
    const std::size_t buf_off = buffer_.size() - rec_len;
    IndexEntry e;
    e.offset = file_size_ + buf_off;
    e.length = rec_len;
    e.tombstone = false;
    e.in_buffer = true;
    index_.emplace_back(id, e);

    // Flush-and-Map: if buffer reached threshold, flush and reset (single-threaded).
    if (pending_bytes_ >= flush_threshold_) {
        POMAI_LOG_DEBUG("StorageEngine::Append auto-flush: pending_bytes={} >= threshold={}",
                        pending_bytes_, flush_threshold_);
        st = Flush();
        if (!st.ok()) return st;
    }
    return pomai::Status::Ok();
}

pomai::Status StorageEngine::Delete(pomai::VectorId id) {
    if (!is_open()) return pomai::Status(pomai::ErrorCode::kFailedPrecondition, "not open");

    VectorRecordHeader hdr{};
    hdr.id = id;
    hdr.dim = dim_;
    hdr.flags = VectorRecordHeader::kFlagTombstone;
    hdr.metadata_len = 0;

    pomai::Status st = AppendToBuffer(hdr, {}, {});
    if (!st.ok()) return st;

    const std::size_t rec_len = RecordSize(dim_, 0);
    const std::size_t buf_off = buffer_.size() - rec_len;
    IndexEntry e;
    e.offset = file_size_ + buf_off;
    e.length = rec_len;
    e.tombstone = true;
    e.in_buffer = true;
    index_.emplace_back(id, e);

    if (pending_bytes_ >= flush_threshold_) {
        POMAI_LOG_DEBUG("StorageEngine::Delete auto-flush: pending_bytes={} >= threshold={}",
                        pending_bytes_, flush_threshold_);
        st = Flush();
        if (!st.ok()) return st;
    }
    return pomai::Status::Ok();
}

pomai::Status StorageEngine::FlushBufferToFile() {
    if (buffer_.empty()) return pomai::Status::Ok();

    const std::byte* p = buffer_.data();
    std::size_t rem = buffer_.size();
    while (rem > 0) {
        ssize_t n = ::write(fd_, p, rem);
        if (n < 0 && errno != EINTR) {
            POMAI_LOG_ERROR("StorageEngine::FlushBufferToFile write failed");
            return ErrnoStatus("write");
        }
        if (n > 0) {
            p += static_cast<std::size_t>(n);
            rem -= static_cast<std::size_t>(n);
        }
    }

    if (::fsync(fd_) != 0) {
        POMAI_LOG_ERROR("StorageEngine::FlushBufferToFile fsync failed");
        return ErrnoStatus("fsync");
    }

    const std::size_t written = buffer_.size();
    file_size_ += written;
    buffer_.clear();
    pending_bytes_ = 0;
    for (auto& kv : index_) {
        if (kv.second.in_buffer) kv.second.in_buffer = false;
    }
    POMAI_LOG_DEBUG("StorageEngine::FlushBufferToFile wrote {} bytes, file_size={}", written, file_size_);
    return pomai::Status::Ok();
}

pomai::Status StorageEngine::Flush() {
    if (!is_open()) return pomai::Status(pomai::ErrorCode::kFailedPrecondition, "not open");
    pomai::Status st = FlushBufferToFile();
    if (!st.ok()) return st;
    POMAI_LOG_DEBUG("StorageEngine::Flush reloading mmap");
    return ReloadMmap();
}

pomai::Status StorageEngine::BuildIndexFromMmap() {
    index_.clear();
    if (!map_addr_ || map_size_ <= sizeof(StorageFileHeader)) return pomai::Status::Ok();

    const std::byte* base = static_cast<const std::byte*>(map_addr_);
    std::size_t off = sizeof(StorageFileHeader);
    while (off + sizeof(VectorRecordHeader) <= map_size_) {
        const auto* h = reinterpret_cast<const VectorRecordHeader*>(base + off);
        const std::size_t rec_len = RecordSize(h->dim, h->metadata_len);
        if (off + rec_len > map_size_) break;
        index_.emplace_back(h->id, IndexEntry{off, rec_len, h->is_tombstone(), false});
        off += rec_len;
    }
    return pomai::Status::Ok();
}

pomai::Status StorageEngine::ReloadMmap() {
    if (map_addr_ && map_size_ > 0) {
        ::munmap(map_addr_, map_size_);
        map_addr_ = nullptr;
        map_size_ = 0;
    }
    if (fd_ < 0) return pomai::Status::Ok();

    struct stat st;
    if (::fstat(fd_, &st) != 0) return ErrnoStatus("fstat");
    file_size_ = static_cast<std::size_t>(st.st_size);

    if (file_size_ == 0) return BuildIndexFromMmap();

    void* addr = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr == MAP_FAILED) return ErrnoStatus("mmap");
    map_addr_ = addr;
    map_size_ = file_size_;
    return BuildIndexFromMmap();
}

pomai::Status StorageEngine::Get(pomai::VectorId id, GetResult* out) const {
    if (!out) return pomai::Status::InvalidArgument("out is null");
    out->data = nullptr;
    out->dim = 0;
    out->is_tombstone = false;
    out->meta = nullptr;

    auto it = std::find_if(index_.rbegin(), index_.rend(),
                           [id](const auto& p) { return p.first == id; });
    if (it == index_.rend()) return pomai::Status::NotFound("vector id");

    const IndexEntry& e = it->second;
    if (e.tombstone) {
        out->is_tombstone = true;
        return pomai::Status::Ok();
    }

    const std::size_t buf_off = e.in_buffer ? (e.offset - file_size_) : 0u;
    const std::byte* rec = e.in_buffer
        ? (buffer_.data() + buf_off)
        : (static_cast<const std::byte*>(map_addr_) + e.offset);

    if (e.in_buffer && (buf_off + e.length > buffer_.size()))
        return pomai::Status::Corruption("index");
    if (!e.in_buffer && (e.offset + e.length > map_size_))
        return pomai::Status::Corruption("index");

    const auto* h = reinterpret_cast<const VectorRecordHeader*>(rec);
    rec += sizeof(VectorRecordHeader) + h->metadata_len;
    out->dim = h->dim;
    out->data = reinterpret_cast<const float*>(rec);
    return pomai::Status::Ok();
}

#else

pomai::Status StorageEngine::Open(std::string_view, std::uint32_t, palloc_heap_t*,
                                  std::size_t) {
    return pomai::Status::IOError("Windows: use CreateFile/CreateFileMapping");
}
pomai::Status StorageEngine::Close() {
    file_size_ = 0;
    index_.clear();
    buffer_.clear();
    pending_bytes_ = 0;
    return pomai::Status::Ok();
}
pomai::Status StorageEngine::AppendToBuffer(const VectorRecordHeader&, std::span<const std::byte>, std::span<const float>) {
    return pomai::Status::IOError("Windows not implemented");
}
pomai::Status StorageEngine::Append(pomai::VectorId, std::span<const float>, const pomai::Metadata*) {
    return pomai::Status::IOError("Windows not implemented");
}
pomai::Status StorageEngine::Delete(pomai::VectorId) {
    return pomai::Status::IOError("Windows not implemented");
}
pomai::Status StorageEngine::FlushBufferToFile() {
    return pomai::Status::IOError("Windows not implemented");
}
pomai::Status StorageEngine::Flush() {
    return pomai::Status::IOError("Windows not implemented");
}
pomai::Status StorageEngine::BuildIndexFromMmap() {
    return pomai::Status::Ok();
}
pomai::Status StorageEngine::ReloadMmap() {
    return pomai::Status::Ok();
}
pomai::Status StorageEngine::Get(pomai::VectorId, GetResult*) const {
    return pomai::Status::IOError("Windows not implemented");
}

#endif

} // namespace pomaidb
