#include <pomai/storage/manifest.h>
#include <pomai/storage/file_util.h>
#include <pomai/storage/crc64.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace pomai::storage
{
    namespace
    {
        constexpr std::array<char, 16> kManifestMagic = {'P', 'O', 'M', 'A', 'I', '_', 'M', 'A', 'N', 'I', 'F', 'E', 'S', 'T', '\0', '\0'};
        constexpr std::uint32_t kManifestVersion = 1;
        constexpr std::uint32_t kEndianMarker = 0x01020304U;

        enum : std::uint32_t
        {
            kTlvCheckpointPath = 1,
            kTlvCheckpointEpoch = 2,
            kTlvCheckpointLsn = 3,
            kTlvShardLsns = 4,
            kTlvDictPath = 5,
            kTlvDictCrc = 6,
            kTlvIndexArtifacts = 7
        };

        std::uint32_t HostToLe32(std::uint32_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap32(v);
#else
            return v;
#endif
        }

        std::uint64_t HostToLe64(std::uint64_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap64(v);
#else
            return v;
#endif
        }

        std::uint32_t Le32ToHost(std::uint32_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap32(v);
#else
            return v;
#endif
        }

        std::uint64_t Le64ToHost(std::uint64_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap64(v);
#else
            return v;
#endif
        }

        void PutBytes(std::vector<std::uint8_t> &buf, const void *data, std::size_t len)
        {
            const auto *p = static_cast<const std::uint8_t *>(data);
            buf.insert(buf.end(), p, p + len);
        }

        void PutU32(std::vector<std::uint8_t> &buf, std::uint32_t v)
        {
            std::uint32_t le = HostToLe32(v);
            PutBytes(buf, &le, sizeof(le));
        }

        void PutU64(std::vector<std::uint8_t> &buf, std::uint64_t v)
        {
            std::uint64_t le = HostToLe64(v);
            PutBytes(buf, &le, sizeof(le));
        }

        void PutString(std::vector<std::uint8_t> &buf, const std::string &s)
        {
            PutU32(buf, static_cast<std::uint32_t>(s.size()));
            PutBytes(buf, s.data(), s.size());
        }

        bool ReadU32(const std::uint8_t *&p, const std::uint8_t *end, std::uint32_t &out)
        {
            if (end - p < 4)
                return false;
            std::uint32_t v;
            std::memcpy(&v, p, 4);
            p += 4;
            out = Le32ToHost(v);
            return true;
        }

        bool ReadU64(const std::uint8_t *&p, const std::uint8_t *end, std::uint64_t &out)
        {
            if (end - p < 8)
                return false;
            std::uint64_t v;
            std::memcpy(&v, p, 8);
            p += 8;
            out = Le64ToHost(v);
            return true;
        }

        bool ReadString(const std::uint8_t *&p, const std::uint8_t *end, std::string &out)
        {
            std::uint32_t len = 0;
            if (!ReadU32(p, end, len))
                return false;
            if (static_cast<std::size_t>(end - p) < len)
                return false;
            out.assign(reinterpret_cast<const char *>(p), reinterpret_cast<const char *>(p + len));
            p += len;
            return true;
        }
    }

    ManifestStatus LoadManifest(const std::string &db_dir, Manifest &out, std::string *err)
    {
        crc64_init();
        const DbPaths paths = MakeDbPaths(db_dir);
        int fd = ::open(paths.manifest_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
        {
            if (errno == ENOENT)
                return ManifestStatus::NotFound;
            if (err)
                *err = std::string("open manifest failed: ") + std::strerror(errno);
            return ManifestStatus::Corrupt;
        }

        std::array<char, 16> magic;
        std::uint32_t version_le = 0;
        std::uint32_t endian_le = 0;
        std::uint64_t header_crc_le = 0;
        std::uint64_t payload_len_le = 0;
        std::uint64_t payload_crc_le = 0;

        if (!ReadFull(fd, magic.data(), magic.size()) ||
            !ReadFull(fd, &version_le, sizeof(version_le)) ||
            !ReadFull(fd, &endian_le, sizeof(endian_le)) ||
            !ReadFull(fd, &header_crc_le, sizeof(header_crc_le)) ||
            !ReadFull(fd, &payload_len_le, sizeof(payload_len_le)) ||
            !ReadFull(fd, &payload_crc_le, sizeof(payload_crc_le)))
        {
            ::close(fd);
            if (err)
                *err = "manifest header read failed";
            return ManifestStatus::Corrupt;
        }

        if (magic != kManifestMagic)
        {
            ::close(fd);
            if (err)
                *err = "manifest magic mismatch";
            return ManifestStatus::Corrupt;
        }

        std::uint32_t version = Le32ToHost(version_le);
        std::uint32_t endian = Le32ToHost(endian_le);
        std::uint64_t header_crc = Le64ToHost(header_crc_le);
        std::uint64_t payload_len = Le64ToHost(payload_len_le);
        std::uint64_t payload_crc = Le64ToHost(payload_crc_le);

        if (endian != kEndianMarker)
        {
            ::close(fd);
            if (err)
                *err = "manifest endian mismatch";
            return ManifestStatus::Corrupt;
        }
        if (version != kManifestVersion)
        {
            ::close(fd);
            if (err)
                *err = "manifest version mismatch";
            return ManifestStatus::Corrupt;
        }

        std::vector<std::uint8_t> header_bytes;
        header_bytes.insert(header_bytes.end(), magic.begin(), magic.end());
        std::uint32_t ver_tmp = HostToLe32(version);
        std::uint32_t end_tmp = HostToLe32(endian);
        PutBytes(header_bytes, &ver_tmp, sizeof(ver_tmp));
        PutBytes(header_bytes, &end_tmp, sizeof(end_tmp));
        std::uint64_t calc_header_crc = crc64(0, header_bytes.data(), header_bytes.size());
        if (calc_header_crc != header_crc)
        {
            ::close(fd);
            if (err)
                *err = "manifest header crc mismatch";
            return ManifestStatus::Corrupt;
        }

        std::vector<std::uint8_t> payload(payload_len);
        if (!payload.empty())
        {
            if (!ReadFull(fd, payload.data(), payload.size()))
            {
                ::close(fd);
                if (err)
                    *err = "manifest payload read failed";
                return ManifestStatus::Corrupt;
            }
        }
        ::close(fd);

        std::uint64_t calc_payload_crc = payload.empty() ? 0 : crc64(0, payload.data(), payload.size());
        if (calc_payload_crc != payload_crc)
        {
            if (err)
                *err = "manifest payload crc mismatch";
            return ManifestStatus::Corrupt;
        }

        Manifest m;
        m.version = version;

        const std::uint8_t *p = payload.data();
        const std::uint8_t *end = payload.data() + payload.size();
        while (p < end)
        {
            std::uint32_t type = 0;
            std::uint64_t len = 0;
            if (!ReadU32(p, end, type) || !ReadU64(p, end, len))
            {
                if (err)
                    *err = "manifest tlv header invalid";
                return ManifestStatus::Corrupt;
            }
            if (static_cast<std::size_t>(end - p) < len)
            {
                if (err)
                    *err = "manifest tlv length invalid";
                return ManifestStatus::Corrupt;
            }
            const std::uint8_t *seg_end = p + len;
            switch (type)
            {
            case kTlvCheckpointPath:
                if (!ReadString(p, seg_end, m.checkpoint_path))
                    return ManifestStatus::Corrupt;
                break;
            case kTlvCheckpointEpoch:
                if (!ReadU64(p, seg_end, m.checkpoint_epoch))
                    return ManifestStatus::Corrupt;
                break;
            case kTlvCheckpointLsn:
                if (!ReadU64(p, seg_end, m.checkpoint_lsn))
                    return ManifestStatus::Corrupt;
                break;
            case kTlvShardLsns:
            {
                std::uint32_t count = 0;
                if (!ReadU32(p, seg_end, count))
                    return ManifestStatus::Corrupt;
                m.shard_lsns.clear();
                m.shard_lsns.reserve(count);
                for (std::uint32_t i = 0; i < count; ++i)
                {
                    std::uint64_t l = 0;
                    if (!ReadU64(p, seg_end, l))
                        return ManifestStatus::Corrupt;
                    m.shard_lsns.push_back(l);
                }
                break;
            }
            case kTlvDictPath:
                if (!ReadString(p, seg_end, m.dict_path))
                    return ManifestStatus::Corrupt;
                break;
            case kTlvDictCrc:
                if (!ReadU64(p, seg_end, m.dict_crc64))
                    return ManifestStatus::Corrupt;
                break;
            case kTlvIndexArtifacts:
            {
                std::uint32_t count = 0;
                if (!ReadU32(p, seg_end, count))
                    return ManifestStatus::Corrupt;
                m.indexes.clear();
                m.indexes.reserve(count);
                for (std::uint32_t i = 0; i < count; ++i)
                {
                    IndexArtifact ia;
                    if (!ReadString(p, seg_end, ia.kind))
                        return ManifestStatus::Corrupt;
                    if (!ReadString(p, seg_end, ia.path))
                        return ManifestStatus::Corrupt;
                    if (!ReadU64(p, seg_end, ia.crc64))
                        return ManifestStatus::Corrupt;
                    m.indexes.push_back(std::move(ia));
                }
                break;
            }
            default:
                p = seg_end;
                break;
            }
            if (p < seg_end)
                p = seg_end;
        }

        out = std::move(m);
        return ManifestStatus::Ok;
    }

    bool WriteManifestAtomic(const std::string &db_dir, const Manifest &m, std::string *err)
    {
        crc64_init();
        const DbPaths paths = MakeDbPaths(db_dir);
        if (!EnsureDirExists(paths.db_dir))
        {
            if (err)
                *err = "manifest db_dir missing";
            return false;
        }

        std::vector<std::uint8_t> payload;
        auto put_tlv = [&](std::uint32_t type, const std::vector<std::uint8_t> &data)
        {
            PutU32(payload, type);
            PutU64(payload, static_cast<std::uint64_t>(data.size()));
            payload.insert(payload.end(), data.begin(), data.end());
        };

        if (!m.checkpoint_path.empty())
        {
            std::vector<std::uint8_t> data;
            PutString(data, m.checkpoint_path);
            put_tlv(kTlvCheckpointPath, data);
        }
        {
            std::vector<std::uint8_t> data;
            PutU64(data, m.checkpoint_epoch);
            put_tlv(kTlvCheckpointEpoch, data);
        }
        {
            std::vector<std::uint8_t> data;
            PutU64(data, m.checkpoint_lsn);
            put_tlv(kTlvCheckpointLsn, data);
        }
        {
            std::vector<std::uint8_t> data;
            PutU32(data, static_cast<std::uint32_t>(m.shard_lsns.size()));
            for (auto l : m.shard_lsns)
                PutU64(data, l);
            put_tlv(kTlvShardLsns, data);
        }
        if (!m.dict_path.empty())
        {
            std::vector<std::uint8_t> data;
            PutString(data, m.dict_path);
            put_tlv(kTlvDictPath, data);
        }
        {
            std::vector<std::uint8_t> data;
            PutU64(data, m.dict_crc64);
            put_tlv(kTlvDictCrc, data);
        }
        if (!m.indexes.empty())
        {
            std::vector<std::uint8_t> data;
            PutU32(data, static_cast<std::uint32_t>(m.indexes.size()));
            for (const auto &ia : m.indexes)
            {
                PutString(data, ia.kind);
                PutString(data, ia.path);
                PutU64(data, ia.crc64);
            }
            put_tlv(kTlvIndexArtifacts, data);
        }

        std::vector<std::uint8_t> header_bytes;
        header_bytes.insert(header_bytes.end(), kManifestMagic.begin(), kManifestMagic.end());
        std::uint32_t ver_le = HostToLe32(kManifestVersion);
        std::uint32_t end_le = HostToLe32(kEndianMarker);
        PutBytes(header_bytes, &ver_le, sizeof(ver_le));
        PutBytes(header_bytes, &end_le, sizeof(end_le));
        std::uint64_t header_crc = crc64(0, header_bytes.data(), header_bytes.size());
        std::uint64_t payload_crc = payload.empty() ? 0 : crc64(0, payload.data(), payload.size());

        std::string tmp = paths.manifest_path + ".tmp";
        int fd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
        if (fd < 0)
        {
            if (err)
                *err = std::string("manifest open tmp failed: ") + std::strerror(errno);
            return false;
        }

        if (!WriteFull(fd, kManifestMagic.data(), kManifestMagic.size()) ||
            !WriteFull(fd, &ver_le, sizeof(ver_le)) ||
            !WriteFull(fd, &end_le, sizeof(end_le)))
        {
            ::close(fd);
            if (err)
                *err = "manifest write header failed";
            return false;
        }
        std::uint64_t header_crc_le = HostToLe64(header_crc);
        std::uint64_t payload_len_le = HostToLe64(static_cast<std::uint64_t>(payload.size()));
        std::uint64_t payload_crc_le = HostToLe64(payload_crc);
        if (!WriteFull(fd, &header_crc_le, sizeof(header_crc_le)) ||
            !WriteFull(fd, &payload_len_le, sizeof(payload_len_le)) ||
            !WriteFull(fd, &payload_crc_le, sizeof(payload_crc_le)))
        {
            ::close(fd);
            if (err)
                *err = "manifest write trailer failed";
            return false;
        }
        if (!payload.empty())
        {
            if (!WriteFull(fd, payload.data(), payload.size()))
            {
                ::close(fd);
                if (err)
                    *err = "manifest write payload failed";
                return false;
            }
        }
        if (!FsyncFile(fd))
        {
            ::close(fd);
            if (err)
                *err = "manifest fsync failed";
            return false;
        }
        ::close(fd);

        if (FailpointHit("before_manifest_rename"))
        {
            if (err)
                *err = "failpoint before_manifest_rename";
            return false;
        }

        if (!AtomicRename(tmp, paths.manifest_path))
        {
            if (err)
                *err = std::string("manifest rename failed: ") + std::strerror(errno);
            return false;
        }
        if (!FsyncDirPath(paths.db_dir))
        {
            if (err)
                *err = "manifest dir fsync failed";
            return false;
        }
        return true;
    }
}
