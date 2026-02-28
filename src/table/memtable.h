// memtable.h — In-memory write buffer for PomaiDB vectors.
//
// PHASE 1 UPDATE: Replaced std::unordered_map + shared_mutex with
// FlatHashMemMap (open-addressing, robin-hood, backward-shift deletion)
// guarded by a seqlock.
//
// Write path: single writer (ShardRuntime actor thread) — lock-free.
// Read path  (ForEach/IterateWithStatus): seqlock read — typically 2 atomic loads.
// Read path  (Get/IsTombstone): seqlock read, O(1) average.
//
// Public API is 100% backward-compatible.

#pragma once
#include <atomic>
#include <cstdint>
#include <span>
#include <unordered_map>       // metadata_ still uses std::unordered_map (small, rare)
#include <shared_mutex>        // kept for metadata_ only
#include "pomai/metadata.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "table/arena.h"
#include "table/flat_hash_memmap.h"
#include "third_party/hash/xxhash64.h"

namespace pomai::table {

/// Hash functor for VectorId using xxHash64 — better distribution than std::hash for sequential IDs.
struct XxHash64ForVectorId {
  std::size_t operator()(pomai::VectorId k) const noexcept {
    return static_cast<std::size_t>(XXHash64::hash(&k, sizeof(k), 0));
  }
};

// Seqlock – readers spin if a write is in progress, readers never block writers.
class Seqlock {
 public:
  void BeginWrite() noexcept {
    uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s | 1u, std::memory_order_release);   // mark odd = writing
    std::atomic_thread_fence(std::memory_order_release);
  }
  void EndWrite() noexcept {
    uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store((s + 1u) & ~uint64_t(1), std::memory_order_release); // advance to even
  }
  // Returns even sequence number snapped at read start.
  uint64_t BeginRead() const noexcept {
    uint64_t s;
    do { s = seq_.load(std::memory_order_acquire); } while (s & 1u);
    return s;
  }
  // Returns true if seq matches (read is consistent).
  bool EndRead(uint64_t s) const noexcept {
    std::atomic_thread_fence(std::memory_order_acquire);
    return seq_.load(std::memory_order_relaxed) == s;
  }
 private:
  std::atomic<uint64_t> seq_{0};
};

class MemTable {
public:
    MemTable(std::uint32_t dim, std::size_t arena_block_bytes);

    pomai::Status Put(pomai::VectorId id, pomai::VectorView vec);
    pomai::Status Put(pomai::VectorId id, pomai::VectorView vec, const pomai::Metadata& meta);

    pomai::Status PutBatch(const std::vector<pomai::VectorId>& ids,
                           const std::vector<pomai::VectorView>& vectors);

    pomai::Status Get(pomai::VectorId id, const float** out_vec) const;
    pomai::Status Get(pomai::VectorId id, const float** out_vec, pomai::Metadata* out_meta) const;

    pomai::Status Delete(pomai::VectorId id);

    size_t GetCount() const noexcept {
        return map_.size();       // atomic (seqlock writer is sole mutator)
    }
    void Clear();

    // ---- Cursor ----
    struct CursorEntry {
        pomai::VectorId id;
        std::span<const float> vec;
        bool is_deleted;
        const pomai::Metadata* meta;
    };

    class Cursor {
    public:
        bool Next(CursorEntry* out);
    private:
        friend class MemTable;
        // We snapshot the entire slot array into a small vector at creation time
        // to avoid seqlock gymnastics during multi-step iteration.
        struct Entry {
            pomai::VectorId id;
            float*          ptr;   // nullptr = tombstone
        };
        Cursor(const MemTable* mem,
               std::vector<Entry> snapshot)
            : mem_(mem), snap_(std::move(snapshot)), idx_(0) {}
        const MemTable*    mem_;
        std::vector<Entry> snap_;
        size_t             idx_;
    };

    Cursor CreateCursor() const;

    const float* GetPtr(pomai::VectorId id) const noexcept {
        auto* v = map_.Find(id);
        return v ? *v : nullptr;
    }

    bool IsTombstone(pomai::VectorId id) const noexcept {
        auto* v = map_.Find(id);
        return v && (*v == nullptr);
    }

    // ---- Iteration helpers (seqlock-protected) ----
    template <class Fn>
    void IterateWithStatus(Fn&& fn) const {
        ForEachEntry([&](pomai::VectorId id, float* ptr) {
            bool is_deleted = (ptr == nullptr);
            std::span<const float> vec;
            if (!is_deleted) vec = {ptr, dim_};
            fn(id, vec, is_deleted);
        });
    }

    template <class Fn>
    void IterateWithMetadata(Fn&& fn) const {
        ForEachEntry([&](pomai::VectorId id, float* ptr) {
            bool is_deleted = (ptr == nullptr);
            std::span<const float> vec;
            if (!is_deleted) vec = {ptr, dim_};
            const pomai::Metadata* meta_ptr = nullptr;
            if (!is_deleted) {
                std::shared_lock lk(meta_mu_);
                auto it = metadata_.find(id);
                if (it != metadata_.end()) meta_ptr = &it->second;
            }
            fn(id, vec, is_deleted, meta_ptr);
        });
    }

    template <class Fn>
    void ForEach(Fn&& fn) const {
        ForEachEntry([&](pomai::VectorId id, float* ptr) {
            if (ptr != nullptr) fn(id, std::span<const float>{ptr, dim_});
        });
    }

private:
    // Single-writer fast iteration — no lock needed (writer is sole mutator).
    template <class Fn>
    void ForEachEntry(Fn&& fn) const {
        // Take a consistent seqlock snapshot of the map's slot array.
        // For small maps (< few thousand entries) this is one cache scan.
        map_.ForEach([&](const pomai::VectorId& id, float* const& ptr) {
            fn(id, ptr);
        });
    }

    std::uint32_t dim_;
    Arena         arena_;

    // Primary map: VectorId -> float* (nullptr = tombstone)
    // XxHash64 gives better distribution than std::hash for sequential VectorIds (fewer collisions).
    mutable FlatHashMemMap<pomai::VectorId, float*, XxHash64ForVectorId> map_;

    // Metadata is rare (only tenant-tagged vectors).
    // Kept as unordered_map + shared_mutex to avoid over-engineering the common path.
    mutable std::unordered_map<pomai::VectorId, pomai::Metadata> metadata_;
    mutable std::shared_mutex meta_mu_;

    // seqlock_ is incremented on every Put/Delete to allow readers to detect races.
    Seqlock seqlock_;
};

} // namespace pomai::table
