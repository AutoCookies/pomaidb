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
    };
}
