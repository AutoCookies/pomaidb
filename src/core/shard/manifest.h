#pragma once

#include <string>
#include <vector>
#include "pomai/status.h"

namespace pomai::core
{
    // detailed: Manages "manifest.current" in shard directory.
    // Format: text file, one relative filename per line.
    class ShardManifest
    {
    public:
        // Load segments list from stored manifest.
        // If manifest doesn't exist, returns Ok with empty list.
        static pomai::Status Load(const std::string &shard_dir, std::vector<std::string> *out_segments);

        // Atomic update of manifest.
        // 1. Write manifest.new
        // 2. Fsync
        // 3. Rename to manifest.current
        // 4. Fsync dir
        static pomai::Status Commit(const std::string &shard_dir, const std::vector<std::string> &segments);
    };

} // namespace pomai::core
