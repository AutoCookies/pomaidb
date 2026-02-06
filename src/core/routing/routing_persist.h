#pragma once

#include <optional>
#include <string>

#include "core/routing/routing_table.h"
#include "pomai/status.h"

namespace pomai::core::routing {

std::string RoutingPath(const std::string& root_path);
std::string RoutingPrevPath(const std::string& root_path);

pomai::Status SaveRoutingTableAtomic(const std::string& root_path, const RoutingTable& table, bool keep_prev);
std::optional<RoutingTable> LoadRoutingTable(const std::string& root_path);
std::optional<RoutingTable> LoadRoutingPrevTable(const std::string& root_path);

} // namespace pomai::core::routing
