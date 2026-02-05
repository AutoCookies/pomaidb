#pragma once
#include <memory>
#include <vector>
#include "pomai/snapshot.h"
#include "core/shard/snapshot.h"

namespace pomai::core {
    
    // Concrete implementation of public opaque Snapshot.
    // Currently wraps a single shard snapshot (since engine only supports single shard iterator).
    // Future: wrap multiple shard snapshots.
    class SnapshotWrapper : public pomai::Snapshot {
    public:
        explicit SnapshotWrapper(std::shared_ptr<ShardSnapshot> s) : snap_(std::move(s)) {}
        
        std::shared_ptr<ShardSnapshot> GetInternal() const { return snap_; }

    private:
        std::shared_ptr<ShardSnapshot> snap_;
    };

} // namespace pomai::core
