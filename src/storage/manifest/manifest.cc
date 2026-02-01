#include "storage/manifest/manifest.h"

#include <filesystem>
#include <string>

#include "util/posix_file.h"

namespace fs = std::filesystem;

namespace pomai::storage
{

    static std::string ManifestPath(const std::string &p)
    {
        return (fs::path(p) / "MANIFEST").string();
    }

    static std::string ManifestTmpPath(const std::string &p)
    {
        return (fs::path(p) / "MANIFEST.tmp").string();
    }

    pomai::Status Manifest::EnsureInitialized(const std::string &db_path,
                                              std::uint32_t shard_count,
                                              std::uint32_t dim)
    {
        fs::create_directories(db_path);

        const auto mp = ManifestPath(db_path);
        if (fs::exists(mp))
            return pomai::Status::Ok();

        // 1) write temp
        pomai::util::PosixFile f;
        auto st = pomai::util::PosixFile::CreateTrunc(ManifestTmpPath(db_path), &f);
        if (!st.ok())
            return st;

        const std::string content =
            "pomai.manifest.v1\n"
            "shards " +
            std::to_string(shard_count) + "\n" +
            "dim " + std::to_string(dim) + "\n";

        st = f.PWrite(0, content.data(), content.size());
        if (!st.ok())
            return st;

        // 2) fsync temp file
        st = f.SyncAll();
        if (!st.ok())
            return st;
        st = f.Close();
        if (!st.ok())
            return st;

        // 3) rename temp -> MANIFEST (atomic)
        std::error_code ec;
        fs::rename(ManifestTmpPath(db_path), mp, ec);
        if (ec)
            return pomai::Status::IoError("manifest rename failed");

        // 4) fsync directory to persist rename
        st = pomai::util::FsyncDir(db_path);
        if (!st.ok())
            return st;

        return pomai::Status::Ok();
    }

} // namespace pomai::storage
