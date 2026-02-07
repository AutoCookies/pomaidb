#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/routing/routing_table.h"

namespace pomai::core::routing {

RoutingTable BuildInitialTable(std::span<const float> samples,
                               std::uint32_t sample_count,
                               std::uint32_t dim,
                               std::uint32_t k,
                               std::uint32_t shard_count,
                               std::uint32_t lloyd_iters,
                               std::uint64_t seed);

void OnlineUpdate(RoutingTable* table, std::span<const float> vec);

} // namespace pomai::core::routing
