#pragma once
#include <vector>
#include <memory>
#include "table/segment.h"
#include "table/memtable.h"

namespace pomai::core
{
    struct ShardSnapshot
    {
        // Snapshot holds shared ownership of immutable data.
        // Frozen memtables are read-only.
        std::vector<std::shared_ptr<table::MemTable>> frozen_memtables;
        std::vector<std::shared_ptr<table::SegmentReader>> segments;
    };
}
