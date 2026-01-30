#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace pomai::storage
{
    struct IndexArtifact
    {
        std::string kind;
        std::string path;
        std::uint64_t crc64{0};
    };

    struct Manifest
    {
        std::uint32_t version{1};
        std::string checkpoint_path;
        std::uint64_t checkpoint_epoch{0};
        std::uint64_t checkpoint_lsn{0};
        std::vector<std::uint64_t> shard_lsns;
        std::string dict_path;
        std::uint64_t dict_crc64{0};
        std::vector<IndexArtifact> indexes;
    };

    enum class ManifestStatus
    {
        Ok,
        NotFound,
        Corrupt
    };

    ManifestStatus LoadManifest(const std::string &db_dir, Manifest &out, std::string *err = nullptr);
    bool WriteManifestAtomic(const std::string &db_dir, const Manifest &m, std::string *err = nullptr);
}
