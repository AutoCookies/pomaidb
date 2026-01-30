#include <pomai/storage/file_util.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace pomai::storage
{
    DbPaths MakeDbPaths(const std::string &db_dir)
    {
        DbPaths out;
        out.db_dir = db_dir;
        out.manifest_path = db_dir + "/MANIFEST";
        out.checkpoints_dir = db_dir + "/checkpoints";
        out.indexes_dir = db_dir + "/indexes";
        out.meta_dir = db_dir + "/meta";
        out.wal_dir = db_dir + "/wal";
        return out;
    }

    bool EnsureDirExists(const std::string &path)
    {
        std::error_code ec;
        if (std::filesystem::exists(path, ec))
            return std::filesystem::is_directory(path, ec);
        return std::filesystem::create_directories(path, ec);
    }

    bool EnsureDbDirs(const DbPaths &paths)
    {
        return EnsureDirExists(paths.db_dir) &&
               EnsureDirExists(paths.checkpoints_dir) &&
               EnsureDirExists(paths.indexes_dir) &&
               EnsureDirExists(paths.meta_dir) &&
               EnsureDirExists(paths.wal_dir);
    }

    void ThrowSys(const std::string &what)
    {
        throw std::runtime_error(what + ": " + std::strerror(errno));
    }

    bool WriteFull(int fd, const void *buf, std::size_t len)
    {
        const auto *p = static_cast<const std::uint8_t *>(buf);
        std::size_t rem = len;
        while (rem > 0)
        {
            ssize_t w = ::write(fd, p, rem);
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                return false;
            }
            if (w == 0)
                return false;
            p += static_cast<std::size_t>(w);
            rem -= static_cast<std::size_t>(w);
        }
        return true;
    }

    bool ReadFull(int fd, void *buf, std::size_t len)
    {
        auto *p = static_cast<std::uint8_t *>(buf);
        std::size_t rem = len;
        while (rem > 0)
        {
            ssize_t r = ::read(fd, p, rem);
            if (r < 0)
            {
                if (errno == EINTR)
                    continue;
                return false;
            }
            if (r == 0)
                return false;
            p += static_cast<std::size_t>(r);
            rem -= static_cast<std::size_t>(r);
        }
        return true;
    }

    bool FsyncFile(int fd)
    {
        if (fd < 0)
            return false;
        return ::fdatasync(fd) == 0;
    }

    bool FsyncDirPath(const std::string &dir)
    {
        int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd < 0)
            return false;
        int rc = ::fsync(dfd);
        ::close(dfd);
        return rc == 0;
    }

    bool FsyncParentDir(const std::string &path)
    {
        std::filesystem::path p(path);
        std::string dir = p.has_parent_path() ? p.parent_path().string() : std::string(".");
        return FsyncDirPath(dir);
    }

    bool AtomicRename(const std::string &tmp_path, const std::string &final_path)
    {
        return ::rename(tmp_path.c_str(), final_path.c_str()) == 0;
    }

    bool FailpointHit(std::string_view name)
    {
#ifndef NDEBUG
        const char *fp = ::getenv("POMAI_FAILPOINT");
        if (!fp)
            return false;
        return name == fp;
#else
        (void)name;
        return false;
#endif
    }
}
