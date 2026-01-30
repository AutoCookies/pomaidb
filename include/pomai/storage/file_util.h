#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace pomai::storage
{
    struct DbPaths
    {
        std::string db_dir;
        std::string manifest_path;
        std::string checkpoints_dir;
        std::string indexes_dir;
        std::string meta_dir;
        std::string wal_dir;
    };

    DbPaths MakeDbPaths(const std::string &db_dir);

    bool EnsureDirExists(const std::string &path);
    bool EnsureDbDirs(const DbPaths &paths);

    void ThrowSys(const std::string &what);

    bool WriteFull(int fd, const void *buf, std::size_t len);
    bool ReadFull(int fd, void *buf, std::size_t len);

    bool FsyncFile(int fd);
    bool FsyncDirPath(const std::string &dir);
    bool FsyncParentDir(const std::string &path);

    bool AtomicRename(const std::string &tmp_path, const std::string &final_path);

    bool FailpointHit(std::string_view name);
}
