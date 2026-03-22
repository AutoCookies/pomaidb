// memtable.cc — MemTable implementation using FlatHashMemMap.
//
// PHASE 1 UPDATE: Replaced std::unordered_map + shared_mutex with
// FlatHashMemMap (open-addressing, robin-hood, backward-shift deletion).
// Single-writer (VectorRuntime): no lock needed on write path.
// Seqlock protects readers.

#include "table/memtable.h"
#include "palloc_compat.h"
#include <cstring>

namespace pomai::table {

static std::size_t AlignUp(std::size_t x, std::size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

void* Arena::Allocate(std::size_t n, std::size_t align) {
    constexpr std::size_t kBlockAlign = 64;
    if (blocks_.empty() || AlignUp(blocks_.back().used, align) + n > block_bytes_) {
        Block b;
        if (heap_) {
            b.mem = static_cast<std::byte*>(palloc_heap_malloc_aligned(heap_, block_bytes_, kBlockAlign));
        } else {
            b.mem = static_cast<std::byte*>(palloc_malloc_aligned(block_bytes_, kBlockAlign));
        }
        b.used = 0;
        blocks_.push_back(b);
    }
    auto& blk  = blocks_.back();
    blk.used   = AlignUp(blk.used, align);
    void* p    = blk.mem + blk.used;
    blk.used  += n;
    return p;
}

void Arena::Clear() {
    for (auto& b : blocks_) {
        if (b.mem) {
            palloc_free(b.mem);
            b.mem = nullptr;
        }
    }
    blocks_.clear();
}

std::size_t Arena::BytesUsed() const noexcept {
    std::size_t total = 0;
    for (const auto& b : blocks_)
        total += b.used;
    return total;
}

std::size_t MemTable::BytesUsed() const noexcept {
    return arena_.BytesUsed();
}

// ------------------------------------------------
// MemTable constructor
// ------------------------------------------------
MemTable::MemTable(std::uint32_t dim, std::size_t arena_block_bytes, palloc_heap_t* heap)
    : dim_(dim), arena_(arena_block_bytes, heap),
      map_(/* initial_cap = */ 128)
{}

// ------------------------------------------------
// Put (no metadata)
// ------------------------------------------------
pomai::Status MemTable::Put(pomai::VectorId id, pomai::VectorView vec) {
    return Put(id, vec, pomai::Metadata());
}

// ------------------------------------------------
// Put (with metadata)
// ------------------------------------------------
pomai::Status MemTable::Put(pomai::VectorId id, pomai::VectorView vec,
                            const pomai::Metadata& meta) {
    if (vec.dim != dim_)
        return pomai::Status::InvalidArgument("dim mismatch");

    float* dst = static_cast<float*>(arena_.Allocate(vec.size_bytes(), alignof(float)));
    std::memcpy(dst, vec.data, vec.size_bytes());

    seqlock_.BeginWrite();
    map_.Put(id, dst);
    seqlock_.EndWrite();

    // Temporal Index Management
    auto it_old = metadata_.find(id);
    if (it_old != metadata_.end()) {
        uint64_t old_ts = it_old->second.timestamp;
        if (old_ts > 0) {
            auto range = temporal_index_.equal_range(old_ts);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->second == id) {
                    temporal_index_.erase(it);
                    break;
                }
            }
        }
    }

    if (!meta.tenant.empty() || meta.src_vid != 0 || meta.timestamp != 0 || !meta.text.empty()) {
        metadata_[id] = meta;
        if (meta.timestamp > 0) {
            temporal_index_.insert({meta.timestamp, id});
        }
        if (!meta.text.empty()) {
            lexical_index_.Add(id, meta.text);
        }
    } else {
        metadata_.erase(id);
    }
    return pomai::Status::Ok();
}

// ------------------------------------------------
// PutBatch
// ------------------------------------------------
pomai::Status MemTable::PutBatch(const std::vector<pomai::VectorId>& ids,
                                 const std::vector<pomai::VectorView>& vectors) {
    if (ids.size() != vectors.size())
        return pomai::Status::InvalidArgument("ids and vectors size mismatch");
    if (ids.empty())
        return pomai::Status::Ok();

    for (const auto& vec : vectors)
        if (vec.dim != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

    // Allocate all memory first (arena is not thread-safe, writer-only).
    std::vector<float*> ptrs;
    ptrs.reserve(ids.size());
    for (const auto& vec : vectors) {
        float* dst = static_cast<float*>(arena_.Allocate(vec.size_bytes(), alignof(float)));
        std::memcpy(dst, vec.data, vec.size_bytes());
        ptrs.push_back(dst);
    }

    seqlock_.BeginWrite();
    for (std::size_t i = 0; i < ids.size(); ++i)
        map_.Put(ids[i], ptrs[i]);
    seqlock_.EndWrite();

    return pomai::Status::Ok();
}

// ------------------------------------------------
// Delete (tombstone)
// ------------------------------------------------
pomai::Status MemTable::Delete(pomai::VectorId id) {
    seqlock_.BeginWrite();
    map_.Put(id, nullptr); // nullptr = tombstone
    seqlock_.EndWrite();

    auto it = metadata_.find(id);
    if (it != metadata_.end()) {
        uint64_t ts = it->second.timestamp;
        if (ts > 0) {
            auto range = temporal_index_.equal_range(ts);
            for (auto search_it = range.first; search_it != range.second; ++search_it) {
                if (search_it->second == id) {
                    temporal_index_.erase(search_it);
                    break;
                }
            }
        }
    }

    lexical_index_.Remove(id);
    metadata_.erase(id);
    return pomai::Status::Ok();
}

// ------------------------------------------------
// Get (vector pointer only)
// ------------------------------------------------
pomai::Status MemTable::Get(pomai::VectorId id, const float** out_vec) const {
    return Get(id, out_vec, nullptr);
}

// ------------------------------------------------
// Get (vector + metadata)
// ------------------------------------------------
pomai::Status MemTable::Get(pomai::VectorId id, const float** out_vec,
                            pomai::Metadata* out_meta) const {
    if (!out_vec) return Status::InvalidArgument("out_vec is null");

    // Seqlock read: retry if write is in progress.
    float* ptr = nullptr;
    uint64_t seq;
    do {
        seq = seqlock_.BeginRead();
        auto* v = map_.Find(id);
        ptr = v ? *v : nullptr;
    } while (!seqlock_.EndRead(seq));

    if (ptr == nullptr) {
        *out_vec = nullptr;
        return Status::NotFound("vector not found");
    }
    *out_vec = ptr;

    if (out_meta) {
        auto it = metadata_.find(id);
        *out_meta = (it != metadata_.end()) ? it->second : pomai::Metadata{};
    }
    return Status::Ok();
}

// ------------------------------------------------
// Clear
// ------------------------------------------------
void MemTable::Clear() {
    seqlock_.BeginWrite();
    map_.Clear();
    seqlock_.EndWrite();

    metadata_.clear();
    temporal_index_.clear();
    lexical_index_.Clear();
    arena_.Clear();
}

// ------------------------------------------------
// Cursor — snapshot of all slots at creation time
// ------------------------------------------------
MemTable::Cursor MemTable::CreateCursor() const {
    std::vector<Cursor::Entry> snap;
    // Snapshot the table under a seqlock read.
    uint64_t seq;
    do {
        seq = seqlock_.BeginRead();
        snap.clear();
        map_.ForEach([&](const pomai::VectorId& id, float* const& ptr) {
            snap.push_back({id, ptr});
        });
    } while (!seqlock_.EndRead(seq));

    return Cursor(this, std::move(snap));
}

bool MemTable::Cursor::Next(CursorEntry* out) {
    if (!out || idx_ >= snap_.size()) return false;

    const Entry& e         = snap_[idx_++];
    const bool   is_deleted = (e.ptr == nullptr);
    std::span<const float> vec;
    if (!is_deleted) vec = {e.ptr, mem_->dim_};

    const pomai::Metadata* meta_ptr = nullptr;
    if (!is_deleted) {
        auto it = mem_->metadata_.find(e.id);
        if (it != mem_->metadata_.end()) meta_ptr = &it->second;
    }
    *out = CursorEntry{e.id, vec, is_deleted, meta_ptr};
    return true;
}

} // namespace pomai::table
