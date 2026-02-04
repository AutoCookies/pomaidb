#pragma once
#include <vector>
#include <memory>
#include <chrono>
#include "table/segment.h"
#include "table/memtable.h"

namespace pomai::core
{
    struct ShardSnapshot
    {
        // Snapshot holds shared ownership of immutable data.
        // Frozen memtables are read-only.
        std::uint64_t version{0};
        std::chrono::steady_clock::time_point created_at;
        
        std::vector<std::shared_ptr<table::MemTable>> frozen_memtables;
        std::vector<std::shared_ptr<table::SegmentReader>> segments;
        
        // Active memtable (not owned by snapshot, but valid while snapshot is held)
        // Wait, snapshot might outlive active memtable if Rotate happens?
        // But Rotate moves mem_ to frozen_mem_.
        // So old mem_ is kept alive in frozen_mem_.
        // Ah, if snap->mem points to mem_.get(), and mem_ is moved...
        // Then snap->mem becomes dangling if it points to the unique_ptr's managed object?
        // unique_ptr manages the object. std::move transfers ownership.
        // The address of the object stays same.
        // So strictly speaking, it is safe IF the object is kept alive.
        // It IS kept alive by frozen_mem_ (which the snapshot also holds!).
        // So yes, it is safe.
        const table::MemTable* mem{nullptr};
    };
}
