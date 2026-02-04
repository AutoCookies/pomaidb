#pragma once
#include <memory>
#include "types.h"
#include <span>

namespace pomai
{
    // SnapshotIterator: Point-in-time iteration over all vectors
    // 
    // Provides snapshot-isolated full-scan access to all live vectors.
    // Guarantees:
    // - Snapshot isolation (point-in-time consistency)
    // - Newest-wins semantics (duplicates resolved to newest version)
    // - Tombstone filtering (deleted IDs not returned)
    //
    // Usage:
    //   std::unique_ptr<SnapshotIterator> it;
    //   db->NewIterator("membrane", &it);
    //   while (it->Valid()) {
    //       VectorId id = it->id();
    //       std::span<const float> vec = it->vector();
    //       // process...
    //       it->Next();
    //   }
    
    class SnapshotIterator
    {
    public:
        virtual ~SnapshotIterator() = default;

        // Advance to next live vector
        // Returns: true if advanced, false if no more vectors
        virtual bool Next() = 0;

        // Current vector ID (valid only if Valid() == true)
        virtual VectorId id() const = 0;

        // Current vector data (valid only if Valid() == true)
        // Returns view into snapshot (no copy)
        virtual std::span<const float> vector() const = 0;

        // Is iterator positioned at valid entry?
        virtual bool Valid() const = 0;
    };

} // namespace pomai
