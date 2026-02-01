#include "pomai/core/manifest.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace pomai::core
{
    static constexpr std::uint32_t kManifestMagic = 0x49414D50; // "PMAI" (little endian)
    static constexpr std::uint32_t kManifestVersion = 1;

    static pomai::Status IOErr(const char *what)
    {
        return pomai::Status::IO(std::string(what) + ": " + std::strerror(errno));
    }

    static pomai::Status Corrupt(std::string msg)
    {
        // Pomai Status doesn't have Corruption => treat as IO corruption.
        return pomai::Status::IO("corruption: " + std::move(msg));
    }

    template <class T>
    static bool ReadExact(std::ifstream &ifs, T &out)
    {
        ifs.read(reinterpret_cast<char *>(&out), sizeof(T));
        return static_cast<bool>(ifs) && ifs.gcount() == static_cast<std::streamsize>(sizeof(T));
    }

    static bool ReadBytes(std::ifstream &ifs, std::vector<char> &buf, std::size_t n)
    {
        buf.resize(n);
        if (n == 0)
            return true;
        ifs.read(buf.data(), static_cast<std::streamsize>(n));
        return static_cast<bool>(ifs) && ifs.gcount() == static_cast<std::streamsize>(n);
    }

    static bool ReadString(std::ifstream &ifs, std::string &out)
    {
        std::uint32_t len = 0;
        if (!ReadExact(ifs, len))
            return false;

        std::vector<char> tmp;
        if (!ReadBytes(ifs, tmp, static_cast<std::size_t>(len)))
            return false;

        out.assign(tmp.begin(), tmp.end());
        return true;
    }

    template <class T>
    static void WriteLE(std::ofstream &ofs, const T &v)
    {
        ofs.write(reinterpret_cast<const char *>(&v), sizeof(T));
    }

    static void WriteString(std::ofstream &ofs, const std::string &s)
    {
        std::uint32_t len = static_cast<std::uint32_t>(s.size());
        WriteLE(ofs, len);
        if (!s.empty())
            ofs.write(s.data(), static_cast<std::streamsize>(s.size()));
    }

    static pomai::Status FsyncFileAndDir(const std::filesystem::path &file_path)
    {
#if defined(__linux__) || defined(__APPLE__)
        // fsync file
        int fd = ::open(file_path.c_str(), O_RDONLY);
        if (fd < 0)
            return IOErr("open for fsync");

        if (::fsync(fd) != 0)
        {
            ::close(fd);
            return IOErr("fsync file");
        }
        ::close(fd);

        // fsync parent dir
        auto dir = file_path.parent_path();
        int dfd = ::open(dir.c_str(), O_RDONLY);
        if (dfd < 0)
            return IOErr("open dir for fsync");

        if (::fsync(dfd) != 0)
        {
            ::close(dfd);
            return IOErr("fsync dir");
        }
        ::close(dfd);

        return pomai::Status::OK();
#else
        (void)file_path;
        return pomai::Status::OK();
#endif
    }

    pomai::Status Manifest::Load(const std::filesystem::path &path, Manifest &out)
    {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec))
            return pomai::Status::NotFound("manifest missing");

        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open())
            return pomai::Status::IO("open manifest failed");

        std::uint32_t magic = 0;
        std::uint32_t ver = 0;

        if (!ReadExact(ifs, magic))
            return Corrupt("manifest truncated (magic)");
        if (magic != kManifestMagic)
            return Corrupt("manifest magic mismatch");

        if (!ReadExact(ifs, ver))
            return Corrupt("manifest truncated (version)");
        if (ver != kManifestVersion)
            return Corrupt("manifest version mismatch");

        if (!ReadExact(ifs, out.checkpoint_seq))
            return Corrupt("manifest truncated (checkpoint_seq)");
        if (!ReadExact(ifs, out.wal_start_id))
            return Corrupt("manifest truncated (wal_start_id)");

        if (!ReadString(ifs, out.snapshot_rel))
            return Corrupt("manifest truncated (snapshot_rel)");
        if (!ReadString(ifs, out.blob_idx_rel))
            return Corrupt("manifest truncated (blob_idx_rel)");
        if (!ReadString(ifs, out.hnsw_rel))
            return Corrupt("manifest truncated (hnsw_rel)");

        // Basic sanity
        if (out.wal_start_id == 0)
            return Corrupt("wal_start_id=0 is invalid");
        if (out.snapshot_rel.empty() || out.blob_idx_rel.empty())
            return Corrupt("missing checkpoint artifact paths");

        return pomai::Status::OK();
    }

    pomai::Status Manifest::SaveAtomic(const std::filesystem::path &path) const
    {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);

        const auto tmp = path.string() + ".tmp";

        {
            std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
            if (!ofs.is_open())
                return pomai::Status::IO("create manifest tmp failed");

            const std::uint32_t magic = kManifestMagic;
            const std::uint32_t ver = kManifestVersion;

            WriteLE(ofs, magic);
            WriteLE(ofs, ver);
            WriteLE(ofs, checkpoint_seq);
            WriteLE(ofs, wal_start_id);
            WriteString(ofs, snapshot_rel);
            WriteString(ofs, blob_idx_rel);
            WriteString(ofs, hnsw_rel);

            if (ofs.bad())
                return pomai::Status::IO("write manifest tmp failed");
        }

        // fsync tmp file + parent dir to guarantee crash-safety
        auto st = FsyncFileAndDir(tmp);
        if (!st.ok())
            return st;

        // rename tmp -> path (atomic on POSIX)
        std::filesystem::rename(tmp, path, ec);
        if (ec)
            return pomai::Status::IO("rename manifest tmp failed");

        // fsync manifest + dir to guarantee the rename is durable
        st = FsyncFileAndDir(path);
        if (!st.ok())
            return st;

        return pomai::Status::OK();
    }

} // namespace pomai::core
