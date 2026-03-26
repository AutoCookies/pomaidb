// memtable.h — In-memory write buffer for PomaiDB vectors.
//
// PHASE 1 UPDATE: Replaced std::unordered_map + shared_mutex with
// FlatHashMemMap (open-addressing, robin-hood, backward-shift deletion)
// guarded by a seqlock.
//
// Write path: single writer (VectorRuntime) — lock-free.
// Read path  (ForEach/IterateWithStatus): seqlock read — typically 2 atomic loads.
// Read path  (Get/IsTombstone): seqlock read, O(1) average.
//
// Public API is 100% backward-compatible.

#pragma once
#include <cstdint>
#include <span>
#include <map>
#include <unordered_map>
#include "pomai/metadata.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "core/query/lexical_index.h"
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

// Single-threaded: no concurrency needed.

class MemTable {
public:
    /// @param quantize_inmem  When true vectors are stored as per-vector SQ8 codes
    ///                        (uint8_t[dim] + float min + float inv_scale) in the arena,
    ///                        reducing in-memory footprint ~4x at the cost of a decode on Get.
    MemTable(std::uint32_t dim, std::size_t arena_block_bytes,
             palloc_heap_t* heap = nullptr, bool quantize_inmem = false);

    pomai::Status Put(pomai::VectorId id, pomai::VectorView vec);
    pomai::Status Put(pomai::VectorId id, pomai::VectorView vec, const pomai::Metadata& meta);

    pomai::Status PutBatch(const std::vector<pomai::VectorId>& ids,
                           const std::vector<pomai::VectorView>& vectors);

    pomai::Status Get(pomai::VectorId id, const float** out_vec) const;
    pomai::Status Get(pomai::VectorId id, const float** out_vec, pomai::Metadata* out_meta) const;

    bool IsQuantizedInMem() const noexcept { return quantize_inmem_; }

    pomai::Status Delete(pomai::VectorId id);

    size_t GetCount() const noexcept {
        return map_.size();       // atomic (seqlock writer is sole mutator)
    }
    /** Approximate bytes used by active memtable (arena + map overhead). For pressure backpressure. */
    std::size_t BytesUsed() const noexcept;
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
        struct Entry {
            pomai::VectorId id;
            float*          ptr;   // nullptr = tombstone
        };
        Cursor(const MemTable* mem, std::vector<Entry> snapshot, bool quantized)
            : mem_(mem), snap_(std::move(snapshot)), idx_(0), quantized_(quantized) {}
        const MemTable*    mem_;
        std::vector<Entry> snap_;
        size_t             idx_;
        bool               quantized_{false};
        std::vector<float> decode_buf_; // reused across Next() calls for quantized decode
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
        if (quantize_inmem_) {
            // Quantized path: decode per-vector SQ8 on the fly
            thread_local std::vector<float> decode_buf;
            ForEachEntry([&](pomai::VectorId id, float* ptr) {
                bool is_deleted = (ptr == nullptr);
                std::span<const float> vec;
                const pomai::Metadata* meta_ptr = nullptr;
                if (!is_deleted) {
                    // Layout: [float min][float inv_scale][uint8_t codes[dim_]]
                    float vmin = ptr[0];
                    float inv_scale = ptr[1];
                    const uint8_t* codes = reinterpret_cast<const uint8_t*>(ptr + 2);
                    decode_buf.resize(dim_);
                    const float scale = inv_scale > 0.0f ? (1.0f / inv_scale) : 0.0f;
                    for (uint32_t i = 0; i < dim_; ++i)
                        decode_buf[i] = vmin + codes[i] * scale;
                    vec = {decode_buf.data(), dim_};
                    auto it = metadata_.find(id);
                    if (it != metadata_.end()) meta_ptr = &it->second;
                }
                fn(id, vec, is_deleted, meta_ptr);
            });
        } else {
            ForEachEntry([&](pomai::VectorId id, float* ptr) {
                bool is_deleted = (ptr == nullptr);
                std::span<const float> vec;
                if (!is_deleted) vec = {ptr, dim_};
                const pomai::Metadata* meta_ptr = nullptr;
                if (!is_deleted) {
                    auto it = metadata_.find(id);
                    if (it != metadata_.end()) meta_ptr = &it->second;
                }
                fn(id, vec, is_deleted, meta_ptr);
            });
        }
    }

    /// Optimized iteration for quantized MemTable: yields raw SQ8 codes + header.
    /// Fn signature: (VectorId, float min, float inv_scale, const uint8_t* codes, bool is_deleted, const Metadata*)
    template <class Fn>
    void IterateQuantizedWithMetadata(Fn&& fn) const {
        ForEachEntry([&](pomai::VectorId id, float* ptr) {
            bool is_deleted = (ptr == nullptr);
            if (is_deleted) {
                fn(id, 0.0f, 0.0f, nullptr, true, nullptr);
                return;
            }
            const pomai::Metadata* meta_ptr = nullptr;
            auto it = metadata_.find(id);
            if (it != metadata_.end()) meta_ptr = &it->second;
            if (quantize_inmem_) {
                float vmin = ptr[0];
                float inv_scale = ptr[1];
                const uint8_t* codes = reinterpret_cast<const uint8_t*>(ptr + 2);
                fn(id, vmin, inv_scale, codes, false, meta_ptr);
            } else {
                // Not quantized — yield float data wrapped as codes = nullptr
                fn(id, 0.0f, 0.0f, nullptr, false, meta_ptr);
            }
        });
    }

    template <class Fn>
    void ForEach(Fn&& fn) const {
        ForEachEntry([&](pomai::VectorId id, float* ptr) {
            if (ptr != nullptr) fn(id, std::span<const float>{ptr, dim_});
        });
    }

    /**
     * @brief Range scan by timestamp.
     */
    void GetByTimeRange(uint64_t start, uint64_t end, std::vector<pomai::VectorId>* out) const {
        if (!out) return;
        auto it_start = temporal_index_.lower_bound(start);
        auto it_end = temporal_index_.upper_bound(end);
        for (auto it = it_start; it != it_end; ++it) {
            out->push_back(it->second);
        }
    }

    /**
     * @brief Search by keyword.
     */
    void SearchLexical(const std::string& query, uint32_t topk, std::vector<core::LexicalHit>* out) const {
         lexical_index_.Search(query, topk, out);
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
    bool          quantize_inmem_{false};
    Arena         arena_;

    // Primary map: VectorId -> float* (nullptr = tombstone)
    // XxHash64 gives better distribution than std::hash for sequential VectorIds (fewer collisions).
    mutable FlatHashMemMap<pomai::VectorId, float*, XxHash64ForVectorId> map_;

    mutable std::unordered_map<pomai::VectorId, pomai::Metadata> metadata_;
    mutable std::multimap<uint64_t, pomai::VectorId> temporal_index_;
    mutable core::LexicalIndex lexical_index_;
};

} // namespace pomai::table
