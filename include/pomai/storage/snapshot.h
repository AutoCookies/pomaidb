#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>

#include <pomai/core/seed.h>
#include <pomai/storage/manifest.h>

namespace pomai::storage
{
    struct SchemaConfig
    {
        std::uint32_t dim{0};
        std::uint32_t metric{0};
        std::uint32_t shards{0};
        std::uint32_t index_kind{0};
    };

    struct ShardSnapshot
    {
        std::uint32_t shard_id{0};
        std::vector<Seed::PersistedState> segments;
        Seed::PersistedState live;
    };

    struct SnapshotData
    {
        SchemaConfig schema;
        std::vector<ShardSnapshot> shards;
        std::vector<std::uint64_t> shard_lsns;
    };

    struct SnapshotWriteResult
    {
        std::uint64_t total_size{0};
        std::uint64_t total_crc64{0};
    };

    bool WriteSnapshotFile(const std::string &path, const SnapshotData &data, SnapshotWriteResult *out, std::string *err = nullptr);
    bool ReadSnapshotFile(const std::string &path, SnapshotData &out, std::string *err = nullptr);
    bool VerifySnapshotFile(const std::string &path, std::string *err = nullptr);

    struct IndexArtifactData
    {
        std::string kind;
        std::vector<std::uint8_t> bytes;
    };

    struct CommitResult
    {
        Manifest manifest;
    };

    bool CommitCheckpointAtomically(const std::string &db_dir,
                                    const SnapshotData &snapshot,
                                    const std::vector<IndexArtifactData> &index_artifacts,
                                    CommitResult *out,
                                    std::string *err = nullptr);
}
