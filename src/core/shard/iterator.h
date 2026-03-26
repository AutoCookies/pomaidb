#pragma once
#include <memory>
#include <unordered_set>
#include <vector>
#include "pomai/iterator.h"
#include "pomai/types.h"
#include "pomai/metadata.h"
#include "core/shard/snapshot.h"

namespace pomai::core
{
    // VectorIterator: Concrete implementation of SnapshotIterator
    //
    // Iterates over frozen memtables + segments in newest-first order.
    // Tracks seen IDs to skip duplicates and tombstones.
    
    class VectorIterator : public SnapshotIterator
    {
    public:
        explicit VectorIterator(std::shared_ptr<VectorSnapshot> snapshot, uint32_t ttl_sec, std::uint64_t now_sec);

        bool Next() override;
        VectorId id() const override;
        std::span<const float> vector() const override;
        bool Valid() const override;

    private:
        // Snapshot (holds frozen memtables + segments)
        std::shared_ptr<VectorSnapshot> snapshot_;

        // Current position
        enum class Source { FROZEN_MEM, SEGMENT, DONE };
        Source source_;
        size_t source_idx_;       // Index into frozen_memtables or segments
        size_t entry_idx_;        // Entry index within current source

        // Current entry data
        VectorId current_id_;
        std::vector<float> current_vec_;  // Copy of vector data

        // Deduplication: track seen IDs to skip old versions
        std::unordered_set<VectorId> seen_;

        uint32_t ttl_sec_{0};
        std::uint64_t now_sec_{0};

        bool IsExpired(const pomai::Metadata& meta) const {
            if (ttl_sec_ == 0) return false;
            if (meta.timestamp == 0) return false;
            if (now_sec_ < meta.timestamp) return false;
            return (now_sec_ - meta.timestamp) >= static_cast<std::uint64_t>(ttl_sec_);
        }

        // Internal helpers
        void AdvanceToNextLive();  // Advance to next valid (unseen, non-tombstone) entry
        bool TryReadFromFrozenMem();
        bool TryReadFromSegment();
    };

} // namespace pomai::core
