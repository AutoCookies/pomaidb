#pragma once
#include <memory>
#include <unordered_set>
#include <vector>
#include "pomai/iterator.h"
#include "pomai/types.h"
#include "core/shard/snapshot.h"

namespace pomai::core
{
    // ShardIterator: Concrete implementation of SnapshotIterator
    //
    // Iterates over frozen memtables + segments in newest-first order.
    // Tracks seen IDs to skip duplicates and tombstones.
    
    class ShardIterator : public SnapshotIterator
    {
    public:
        explicit ShardIterator(std::shared_ptr<ShardSnapshot> snapshot);

        bool Next() override;
        VectorId id() const override;
        std::span<const float> vector() const override;
        bool Valid() const override;

    private:
        // Snapshot (holds frozen memtables + segments)
        std::shared_ptr<ShardSnapshot> snapshot_;

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

        // Internal helpers
        void AdvanceToNextLive();  // Advance to next valid (unseen, non-tombstone) entry
        bool TryReadFromFrozenMem();
        bool TryReadFromSegment();
    };

} // namespace pomai::core
