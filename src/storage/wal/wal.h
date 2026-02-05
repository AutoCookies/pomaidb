#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "pomai/metadata.h"
#include "pomai/options.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::table
{
    class MemTable;
}

namespace pomai::storage
{

    class Wal
    {
    public:
        Wal(std::string db_path,
            std::uint32_t shard_id,
            std::size_t segment_bytes,
            pomai::FsyncPolicy fsync);

        pomai::Status Open();

        pomai::Status AppendPut(pomai::VectorId id, std::span<const float> vec);
        pomai::Status AppendPut(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata& meta); // Added
        pomai::Status AppendDelete(pomai::VectorId id);
        
        // Batch append: Write multiple Put records with single fsync (5-10x faster)
        pomai::Status AppendBatch(const std::vector<pomai::VectorId>& ids,
                                  const std::vector<std::span<const float>>& vectors);

        pomai::Status Flush();
        pomai::Status ReplayInto(pomai::table::MemTable &mem);
        
        // Closes current log, deletes all WAL files, and resets state.
        // Used after successful MemTable flush to segments.
        pomai::Status Reset();

    private:
        std::string SegmentPath(std::uint64_t gen) const;
        pomai::Status RotateIfNeeded(std::size_t add_bytes);

        std::string db_path_;
        std::uint32_t shard_id_;
        std::size_t segment_bytes_;
        pomai::FsyncPolicy fsync_;

        std::uint64_t gen_ = 0;
        std::uint64_t seq_ = 0;

        std::uint64_t file_off_ = 0;
        std::size_t bytes_in_seg_ = 0;

        // POSIX file (append by pwrite at tracked offset)
        class Impl;
        Impl *impl_ = nullptr;
    };

} // namespace pomai::storage
