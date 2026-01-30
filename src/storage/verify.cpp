#include <pomai/storage/verify.h>
#include <pomai/core/posix_compat.h>
#include <pomai/storage/file_util.h>
#include <pomai/storage/crc64.h>

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace pomai::storage
{
    namespace
    {
        std::string ResolvePath(const std::string &db_dir, const std::string &path)
        {
            if (path.empty())
                return {};
            if (path.front() == '/')
                return path;
            return db_dir + "/" + path;
        }

        bool ComputeFileCrc64(const std::string &path, std::uint64_t &out, std::string *err)
        {
            constexpr std::size_t kBufSize = 1024 * 1024;
            int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0)
            {
                if (err)
                    *err = "open failed";
                return false;
            }
            std::vector<std::uint8_t> buf(kBufSize);
            std::uint64_t crc = 0;
            while (true)
            {
                ssize_t r = ::read(fd, buf.data(), buf.size());
                if (r < 0)
                {
                    if (errno == EINTR)
                        continue;
                    ::close(fd);
                    if (err)
                        *err = "read failed";
                    return false;
                }
                if (r == 0)
                    break;
                crc = crc64(crc, buf.data(), static_cast<std::size_t>(r));
            }
            ::close(fd);
            out = crc;
            return true;
        }

        bool ParseCheckpointName(const std::string &name, std::uint64_t &epoch, std::uint64_t &lsn)
        {
            if (name.rfind("chk_", 0) != 0)
                return false;
            if (name.size() < 9)
                return false;
            if (name.substr(name.size() - 6) != ".pomai")
                return false;
            std::string core = name.substr(4, name.size() - 4 - 6);
            auto pos = core.find('_');
            if (pos == std::string::npos)
                return false;
            try
            {
                epoch = std::stoull(core.substr(0, pos));
                lsn = std::stoull(core.substr(pos + 1));
            }
            catch (...)
            {
                return false;
            }
            return true;
        }
    } // namespace

    bool VerifyManifest(const std::string &db_dir, Manifest &manifest, std::string *err)
    {
        auto status = LoadManifest(db_dir, manifest, err);
        return status == ManifestStatus::Ok;
    }

    bool VerifyDictionaryFile(const std::string &path, std::uint64_t expected_crc, std::string *err)
    {
        crc64_init();
        std::uint64_t crc = 0;
        if (!ComputeFileCrc64(path, crc, err))
            return false;
        if (crc != expected_crc)
        {
            if (err)
                *err = "dict crc mismatch";
            return false;
        }
        return true;
    }

    bool VerifyIndexFile(const std::string &path, std::uint64_t expected_crc, std::string *err)
    {
        crc64_init();
        std::uint64_t crc = 0;
        if (!ComputeFileCrc64(path, crc, err))
            return false;
        if (crc != expected_crc)
        {
            if (err)
                *err = "index crc mismatch";
            return false;
        }
        return true;
    }

    bool VerifyAll(const std::string &db_dir, Manifest &manifest, std::string *err)
    {
        if (!VerifyManifest(db_dir, manifest, err))
            return false;
        const std::string snapshot_path = ResolvePath(db_dir, manifest.checkpoint_path);
        if (!snapshot_path.empty() && !VerifySnapshotFile(snapshot_path, err))
            return false;
        if (!manifest.dict_path.empty())
        {
            const std::string dict_path = ResolvePath(db_dir, manifest.dict_path);
            if (!VerifyDictionaryFile(dict_path, manifest.dict_crc64, err))
                return false;
        }
        for (const auto &idx : manifest.indexes)
        {
            const std::string idx_path = ResolvePath(db_dir, idx.path);
            if (!VerifyIndexFile(idx_path, idx.crc64, err))
                return false;
        }
        return true;
    }

    bool RecoverLatestCheckpoint(const std::string &db_dir, SnapshotData &snapshot, Manifest &manifest, std::string *err)
    {
        Manifest loaded;
        std::string load_err;
        auto status = LoadManifest(db_dir, loaded, &load_err);
        if (status == ManifestStatus::Ok)
        {
            const std::string snapshot_path = ResolvePath(db_dir, loaded.checkpoint_path);
            if (!snapshot_path.empty() && ReadSnapshotFile(snapshot_path, snapshot, err))
            {
                if (!loaded.dict_path.empty())
                {
                    const std::string dict_path = ResolvePath(db_dir, loaded.dict_path);
                    if (!VerifyDictionaryFile(dict_path, loaded.dict_crc64, err))
                        goto fallback;
                }
                for (const auto &idx : loaded.indexes)
                {
                    const std::string idx_path = ResolvePath(db_dir, idx.path);
                    if (!VerifyIndexFile(idx_path, idx.crc64, err))
                        goto fallback;
                }
                manifest = std::move(loaded);
                return true;
            }
        }
    fallback:

        std::vector<std::pair<std::uint64_t, std::uint64_t>> candidates;
        DbPaths paths = MakeDbPaths(db_dir);
        if (std::filesystem::exists(paths.checkpoints_dir))
        {
            for (const auto &entry : std::filesystem::directory_iterator(paths.checkpoints_dir))
            {
                if (!entry.is_regular_file())
                    continue;
                std::uint64_t epoch = 0;
                std::uint64_t lsn = 0;
                std::string name = entry.path().filename().string();
                if (ParseCheckpointName(name, epoch, lsn))
                    candidates.emplace_back(epoch, lsn);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b)
                  { return a.first == b.first ? a.second > b.second : a.first > b.first; });

        for (const auto &cand : candidates)
        {
            std::string name = "chk_" + std::to_string(cand.first) + "_" + std::to_string(cand.second) + ".pomai";
            std::string path = paths.checkpoints_dir + "/" + name;
            SnapshotData tmp;
            if (!ReadSnapshotFile(path, tmp, err))
                continue;
            snapshot = std::move(tmp);
            manifest.checkpoint_path = "checkpoints/" + name;
            manifest.checkpoint_epoch = cand.first;
            manifest.checkpoint_lsn = cand.second;
            manifest.version = 1;
            manifest.shard_lsns.clear();
            manifest.indexes.clear();
            manifest.dict_path.clear();
            manifest.dict_crc64 = 0;
            return true;
        }

        if (err && !load_err.empty())
            *err = load_err;
        return false;
    }
} // namespace pomai::storage
