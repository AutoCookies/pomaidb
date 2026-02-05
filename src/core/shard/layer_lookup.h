#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "core/shard/snapshot.h"
#include "pomai/metadata.h"
#include "pomai/types.h"
#include "table/memtable.h"

namespace pomai::core {

enum class LookupState {
    kFound,
    kTombstone,
    kNotFound,
};

struct LookupResult {
    LookupState state{LookupState::kNotFound};
    std::span<const float> vec{};
    pomai::Metadata meta{};
};

// Canonical newest-wins lookup across shard layers.
// Merge order: active memtable -> frozen memtables (newest->oldest) -> segments (newest->oldest).
LookupResult LookupById(const std::shared_ptr<table::MemTable>& active,
                        const std::shared_ptr<ShardSnapshot>& snapshot,
                        pomai::VectorId id,
                        std::uint32_t dim);

}  // namespace pomai::core
