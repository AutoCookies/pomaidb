#pragma once

#include <cstdint>
#include <string>

#include <pomai/storage/manifest.h>
#include <pomai/storage/snapshot.h>

namespace pomai::storage
{
    bool VerifyManifest(const std::string &db_dir, Manifest &manifest, std::string *err = nullptr);
    bool VerifyAll(const std::string &db_dir, Manifest &manifest, std::string *err = nullptr);
    bool VerifyDictionaryFile(const std::string &path, std::uint64_t expected_crc, std::string *err = nullptr);
    bool VerifyIndexFile(const std::string &path, std::uint64_t expected_crc, std::string *err = nullptr);
    bool RecoverLatestCheckpoint(const std::string &db_dir, SnapshotData &snapshot, Manifest &manifest, std::string *err = nullptr);
}
