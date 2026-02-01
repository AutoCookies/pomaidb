#pragma once
#include <cstdint>
#include <filesystem>
#include <string>

#include "pomai/status.h"

namespace pomai::core
{
    // MANIFEST format (binary, little-endian):
    // [u32 magic] [u32 version]
    // [u64 checkpoint_seq]
    // [u64 wal_start_id]
    // [u32 len snapshot_rel][bytes...]
    // [u32 len blob_idx_rel][bytes...]
    // [u32 len hnsw_rel][bytes...]  (can be empty)
    struct Manifest
    {
        std::uint64_t checkpoint_seq{0};
        std::uint64_t wal_start_id{1};

        // relative to MetaDir()
        std::string snapshot_rel;
        std::string blob_idx_rel;
        std::string hnsw_rel; // optional empty

        // Load MANIFEST if exists.
        // - OK(): loaded
        // - NotFound(): manifest missing
        // - IO(): corrupted / truncated / read error
        static pomai::Status Load(const std::filesystem::path &path, Manifest &out);

        // Atomic publish:
        // write temp -> fsync(temp) -> rename(temp, path) -> fsync(dir)
        pomai::Status SaveAtomic(const std::filesystem::path &path) const;
    };

} // namespace pomai::core
