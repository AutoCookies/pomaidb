#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "palloc_compat.h"
#include "pomai/env.h"
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
        /** \a env VFS for file I/O; nullptr = use Env::Default(). \a heap optional. */
        Wal(pomai::Env* env,
            std::string db_path,
            std::uint32_t shard_id,
            std::size_t segment_bytes,
            pomai::FsyncPolicy fsync,
            palloc_heap_t* heap = nullptr);
        ~Wal();

        Wal(const Wal &) = delete;
        Wal &operator=(const Wal &) = delete;

        pomai::Status Open();

        pomai::Status AppendPut(pomai::VectorId id, pomai::VectorView vec);
        pomai::Status AppendPut(pomai::VectorId id, pomai::VectorView vec, const pomai::Metadata& meta); // Added
        pomai::Status AppendDelete(pomai::VectorId id);
        
        /** Generic KV append for Graph and other metadata. */
        pomai::Status AppendRawKV(std::uint8_t op, pomai::Slice key, pomai::Slice value);
        
        // Batch append: Write multiple Put records with single fsync (5-10x faster)
        pomai::Status AppendBatch(const std::vector<pomai::VectorId>& ids,
                                  const std::vector<pomai::VectorView>& vectors);

        pomai::Status Flush();
        pomai::Status ReplayInto(pomai::table::MemTable &mem);
        
        // Closes current log, deletes all WAL files, and resets state.
        // Used after successful MemTable flush to segments.
        pomai::Status Reset();
        
        // Transaction support: group multiple records into one atomic sync unit.
        pomai::Status BeginBatch();
        pomai::Status EndBatch();

        std::uint64_t GetLastLSN() const { return seq_; }

    private:
        std::string SegmentPath(std::uint64_t gen) const;
        pomai::Status RotateIfNeeded(std::size_t add_bytes);

        pomai::Env* env_ = nullptr;
        std::string db_path_;
        std::uint32_t shard_id_;
        std::size_t segment_bytes_;
        pomai::FsyncPolicy fsync_;

        std::uint64_t gen_ = 0;
        std::uint64_t seq_ = 0;

        std::uint64_t file_off_ = 0;
        std::size_t bytes_in_seg_ = 0;

        palloc_heap_t* heap_ = nullptr;
        class Impl;
        Impl* impl_ = nullptr;
        std::vector<std::uint8_t> scratch_;
    };

} // namespace pomai::storage
