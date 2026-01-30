#include <pomai/storage/snapshot.h>
#include <pomai/storage/crc64.h>
#include <pomai/storage/file_util.h>
#include <pomai/storage/manifest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pomai::storage
{
    namespace
    {
        constexpr std::array<char, 10> kSnapMagic = {'P', 'O', 'M', 'A', 'I', '_', 'S', 'N', 'A', 'P'};
        constexpr std::array<char, 8> kFooterMagic = {'P', 'O', 'M', 'A', 'I', 'E', 'N', 'D'};
        constexpr std::uint32_t kSnapVersion = 1;
        constexpr std::uint32_t kEndianMarker = 0x01020304U;
        constexpr std::array<char, 16> kManifestMagic = {'P', 'O', 'M', 'A', 'I', '_', 'M', 'A', 'N', 'I', 'F', 'E', 'S', 'T', '\0', '\0'};

        enum SectionType : std::uint32_t
        {
            kSectionSchema = 1,
            kSectionShard = 2,
            kSectionDictionary = 3
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

        void PutU32(std::vector<std::uint8_t> &buf, std::uint32_t v)
        {
            std::uint32_t le = HostToLe32(v);
            const auto *p = reinterpret_cast<const std::uint8_t *>(&le);
            buf.insert(buf.end(), p, p + sizeof(le));
        }

        void PutU64(std::vector<std::uint8_t> &buf, std::uint64_t v)
        {
            std::uint64_t le = HostToLe64(v);
            const auto *p = reinterpret_cast<const std::uint8_t *>(&le);
            buf.insert(buf.end(), p, p + sizeof(le));
        }

        bool ReadU32(int fd, std::uint32_t &out)
        {
            std::uint32_t tmp = 0;
            if (!ReadFull(fd, &tmp, sizeof(tmp)))
                return false;
            out = Le32ToHost(tmp);
            return true;
        }

        bool ReadU64(int fd, std::uint64_t &out)
        {
            std::uint64_t tmp = 0;
            if (!ReadFull(fd, &tmp, sizeof(tmp)))
                return false;
            out = Le64ToHost(tmp);
            return true;
        }

        std::uint64_t SeedStateSize(const Seed::PersistedState &state)
        {
            std::uint64_t size = 0;
            size += sizeof(std::uint32_t);
            size += sizeof(std::uint64_t) * 8;
            size += sizeof(std::uint8_t);
            size += sizeof(std::uint64_t) * 2;
            size += state.ids.size() * sizeof(Id);
            size += state.qdata.size() * sizeof(std::uint8_t);
            size += state.qmins.size() * sizeof(float);
            size += state.qmaxs.size() * sizeof(float);
            size += state.qscales.size() * sizeof(float);
            size += state.namespace_ids.size() * sizeof(std::uint32_t);
            size += state.tag_offsets.size() * sizeof(std::uint32_t);
            size += state.tag_ids.size() * sizeof(TagId);
            return size;
        }

        void UpdateCrc(std::uint64_t &crc, const void *data, std::size_t len)
        {
            if (len == 0)
                return;
            crc = crc64(crc, reinterpret_cast<const unsigned char *>(data), len);
        }

        bool WriteSectionHeader(int fd, std::uint32_t type, std::uint64_t size, std::uint64_t crc, std::uint64_t &total_crc, std::uint64_t &total_size)
        {
            std::uint32_t type_le = HostToLe32(type);
            std::uint64_t size_le = HostToLe64(size);
            std::uint64_t crc_le = HostToLe64(crc);
            if (!WriteFull(fd, &type_le, sizeof(type_le)) ||
                !WriteFull(fd, &size_le, sizeof(size_le)) ||
                !WriteFull(fd, &crc_le, sizeof(crc_le)))
                return false;
            UpdateCrc(total_crc, &type_le, sizeof(type_le));
            UpdateCrc(total_crc, &size_le, sizeof(size_le));
            UpdateCrc(total_crc, &crc_le, sizeof(crc_le));
            total_size += sizeof(type_le) + sizeof(size_le) + sizeof(crc_le);
            return true;
        }

        bool WriteSeedState(int fd, const Seed::PersistedState &state, std::uint64_t &section_crc, std::uint64_t &total_crc, std::uint64_t &total_size)
        {
            std::uint32_t dim_le = HostToLe32(state.dim);
            std::uint64_t ids_le = HostToLe64(static_cast<std::uint64_t>(state.ids.size()));
            std::uint64_t qdata_le = HostToLe64(static_cast<std::uint64_t>(state.qdata.size()));
            std::uint64_t qmins_le = HostToLe64(static_cast<std::uint64_t>(state.qmins.size()));
            std::uint64_t qmaxs_le = HostToLe64(static_cast<std::uint64_t>(state.qmaxs.size()));
            std::uint64_t qscales_le = HostToLe64(static_cast<std::uint64_t>(state.qscales.size()));
            std::uint64_t ns_le = HostToLe64(static_cast<std::uint64_t>(state.namespace_ids.size()));
            std::uint64_t offsets_le = HostToLe64(static_cast<std::uint64_t>(state.tag_offsets.size()));
            std::uint64_t tags_le = HostToLe64(static_cast<std::uint64_t>(state.tag_ids.size()));
            std::uint8_t fixed = state.is_fixed ? 1 : 0;
            std::uint64_t ingested_le = HostToLe64(state.total_ingested);
            std::uint64_t fixed_after_le = HostToLe64(state.fixed_bounds_after);

            const void *header_fields[] = {&dim_le, &ids_le, &qdata_le, &qmins_le, &qmaxs_le, &qscales_le, &ns_le, &offsets_le, &tags_le, &fixed, &ingested_le, &fixed_after_le};
            const std::size_t header_sizes[] = {sizeof(dim_le), sizeof(ids_le), sizeof(qdata_le), sizeof(qmins_le), sizeof(qmaxs_le), sizeof(qscales_le), sizeof(ns_le), sizeof(offsets_le), sizeof(tags_le), sizeof(fixed), sizeof(ingested_le), sizeof(fixed_after_le)};

            for (std::size_t i = 0; i < std::size(header_fields); ++i)
            {
                if (!WriteFull(fd, header_fields[i], header_sizes[i]))
                    return false;
                UpdateCrc(section_crc, header_fields[i], header_sizes[i]);
                UpdateCrc(total_crc, header_fields[i], header_sizes[i]);
                total_size += header_sizes[i];
            }

            if (!state.ids.empty())
            {
                if (!WriteFull(fd, state.ids.data(), state.ids.size() * sizeof(Id)))
                    return false;
                UpdateCrc(section_crc, state.ids.data(), state.ids.size() * sizeof(Id));
                UpdateCrc(total_crc, state.ids.data(), state.ids.size() * sizeof(Id));
                total_size += state.ids.size() * sizeof(Id);
            }
            if (!state.qdata.empty())
            {
                if (!WriteFull(fd, state.qdata.data(), state.qdata.size()))
                    return false;
                UpdateCrc(section_crc, state.qdata.data(), state.qdata.size());
                UpdateCrc(total_crc, state.qdata.data(), state.qdata.size());
                total_size += state.qdata.size();
            }
            if (!state.qmins.empty())
            {
                if (!WriteFull(fd, state.qmins.data(), state.qmins.size() * sizeof(float)))
                    return false;
                UpdateCrc(section_crc, state.qmins.data(), state.qmins.size() * sizeof(float));
                UpdateCrc(total_crc, state.qmins.data(), state.qmins.size() * sizeof(float));
                total_size += state.qmins.size() * sizeof(float);
            }
            if (!state.qmaxs.empty())
            {
                if (!WriteFull(fd, state.qmaxs.data(), state.qmaxs.size() * sizeof(float)))
                    return false;
                UpdateCrc(section_crc, state.qmaxs.data(), state.qmaxs.size() * sizeof(float));
                UpdateCrc(total_crc, state.qmaxs.data(), state.qmaxs.size() * sizeof(float));
                total_size += state.qmaxs.size() * sizeof(float);
            }
            if (!state.qscales.empty())
            {
                if (!WriteFull(fd, state.qscales.data(), state.qscales.size() * sizeof(float)))
                    return false;
                UpdateCrc(section_crc, state.qscales.data(), state.qscales.size() * sizeof(float));
                UpdateCrc(total_crc, state.qscales.data(), state.qscales.size() * sizeof(float));
                total_size += state.qscales.size() * sizeof(float);
            }
            if (!state.namespace_ids.empty())
            {
                if (!WriteFull(fd, state.namespace_ids.data(), state.namespace_ids.size() * sizeof(std::uint32_t)))
                    return false;
                UpdateCrc(section_crc, state.namespace_ids.data(), state.namespace_ids.size() * sizeof(std::uint32_t));
                UpdateCrc(total_crc, state.namespace_ids.data(), state.namespace_ids.size() * sizeof(std::uint32_t));
                total_size += state.namespace_ids.size() * sizeof(std::uint32_t);
            }
            if (!state.tag_offsets.empty())
            {
                if (!WriteFull(fd, state.tag_offsets.data(), state.tag_offsets.size() * sizeof(std::uint32_t)))
                    return false;
                UpdateCrc(section_crc, state.tag_offsets.data(), state.tag_offsets.size() * sizeof(std::uint32_t));
                UpdateCrc(total_crc, state.tag_offsets.data(), state.tag_offsets.size() * sizeof(std::uint32_t));
                total_size += state.tag_offsets.size() * sizeof(std::uint32_t);
            }
            if (!state.tag_ids.empty())
            {
                if (!WriteFull(fd, state.tag_ids.data(), state.tag_ids.size() * sizeof(TagId)))
                    return false;
                UpdateCrc(section_crc, state.tag_ids.data(), state.tag_ids.size() * sizeof(TagId));
                UpdateCrc(total_crc, state.tag_ids.data(), state.tag_ids.size() * sizeof(TagId));
                total_size += state.tag_ids.size() * sizeof(TagId);
            }
            return true;
        }

        void UpdateSeedStateCrc(std::uint64_t &crc, const Seed::PersistedState &state)
        {
            std::uint32_t dim_le = HostToLe32(state.dim);
            std::uint64_t ids_le = HostToLe64(static_cast<std::uint64_t>(state.ids.size()));
            std::uint64_t qdata_le = HostToLe64(static_cast<std::uint64_t>(state.qdata.size()));
            std::uint64_t qmins_le = HostToLe64(static_cast<std::uint64_t>(state.qmins.size()));
            std::uint64_t qmaxs_le = HostToLe64(static_cast<std::uint64_t>(state.qmaxs.size()));
            std::uint64_t qscales_le = HostToLe64(static_cast<std::uint64_t>(state.qscales.size()));
            std::uint64_t ns_le = HostToLe64(static_cast<std::uint64_t>(state.namespace_ids.size()));
            std::uint64_t offsets_le = HostToLe64(static_cast<std::uint64_t>(state.tag_offsets.size()));
            std::uint64_t tags_le = HostToLe64(static_cast<std::uint64_t>(state.tag_ids.size()));
            std::uint8_t fixed = state.is_fixed ? 1 : 0;
            std::uint64_t ingested_le = HostToLe64(state.total_ingested);
            std::uint64_t fixed_after_le = HostToLe64(state.fixed_bounds_after);

            UpdateCrc(crc, &dim_le, sizeof(dim_le));
            UpdateCrc(crc, &ids_le, sizeof(ids_le));
            UpdateCrc(crc, &qdata_le, sizeof(qdata_le));
            UpdateCrc(crc, &qmins_le, sizeof(qmins_le));
            UpdateCrc(crc, &qmaxs_le, sizeof(qmaxs_le));
            UpdateCrc(crc, &qscales_le, sizeof(qscales_le));
            UpdateCrc(crc, &ns_le, sizeof(ns_le));
            UpdateCrc(crc, &offsets_le, sizeof(offsets_le));
            UpdateCrc(crc, &tags_le, sizeof(tags_le));
            UpdateCrc(crc, &fixed, sizeof(fixed));
            UpdateCrc(crc, &ingested_le, sizeof(ingested_le));
            UpdateCrc(crc, &fixed_after_le, sizeof(fixed_after_le));

            if (!state.ids.empty())
                UpdateCrc(crc, state.ids.data(), state.ids.size() * sizeof(Id));
            if (!state.qdata.empty())
                UpdateCrc(crc, state.qdata.data(), state.qdata.size());
            if (!state.qmins.empty())
                UpdateCrc(crc, state.qmins.data(), state.qmins.size() * sizeof(float));
            if (!state.qmaxs.empty())
                UpdateCrc(crc, state.qmaxs.data(), state.qmaxs.size() * sizeof(float));
            if (!state.qscales.empty())
                UpdateCrc(crc, state.qscales.data(), state.qscales.size() * sizeof(float));
            if (!state.namespace_ids.empty())
                UpdateCrc(crc, state.namespace_ids.data(), state.namespace_ids.size() * sizeof(std::uint32_t));
            if (!state.tag_offsets.empty())
                UpdateCrc(crc, state.tag_offsets.data(), state.tag_offsets.size() * sizeof(std::uint32_t));
            if (!state.tag_ids.empty())
                UpdateCrc(crc, state.tag_ids.data(), state.tag_ids.size() * sizeof(TagId));
        }

        bool ReadSeedState(int fd, Seed::PersistedState &state, std::uint64_t &section_crc, std::uint64_t &total_crc, std::uint64_t &total_size)
        {
            std::uint32_t dim = 0;
            std::uint64_t ids = 0;
            std::uint64_t qdata_bytes = 0;
            std::uint64_t qmins = 0;
            std::uint64_t qmaxs = 0;
            std::uint64_t qscales = 0;
            std::uint64_t ns = 0;
            std::uint64_t offsets = 0;
            std::uint64_t tags = 0;
            std::uint8_t fixed = 0;
            std::uint64_t ingested = 0;
            std::uint64_t fixed_after = 0;

            if (!ReadU32(fd, dim) ||
                !ReadU64(fd, ids) ||
                !ReadU64(fd, qdata_bytes) ||
                !ReadU64(fd, qmins) ||
                !ReadU64(fd, qmaxs) ||
                !ReadU64(fd, qscales) ||
                !ReadU64(fd, ns) ||
                !ReadU64(fd, offsets) ||
                !ReadU64(fd, tags))
                return false;
            if (!ReadFull(fd, &fixed, sizeof(fixed)) ||
                !ReadU64(fd, ingested) ||
                !ReadU64(fd, fixed_after))
                return false;

            std::vector<std::uint8_t> header;
            PutU32(header, dim);
            PutU64(header, ids);
            PutU64(header, qdata_bytes);
            PutU64(header, qmins);
            PutU64(header, qmaxs);
            PutU64(header, qscales);
            PutU64(header, ns);
            PutU64(header, offsets);
            PutU64(header, tags);
            header.push_back(fixed);
            PutU64(header, ingested);
            PutU64(header, fixed_after);

            UpdateCrc(section_crc, header.data(), header.size());
            UpdateCrc(total_crc, header.data(), header.size());
            total_size += header.size();

            state.dim = dim;
            state.ids.resize(ids);
            state.qdata.resize(qdata_bytes);
            state.qmins.resize(qmins);
            state.qmaxs.resize(qmaxs);
            state.qscales.resize(qscales);
            state.namespace_ids.resize(ns);
            state.tag_offsets.resize(offsets);
            state.tag_ids.resize(tags);
            state.is_fixed = fixed != 0;
            state.total_ingested = ingested;
            state.fixed_bounds_after = fixed_after;

            if (!state.ids.empty())
            {
                if (!ReadFull(fd, state.ids.data(), state.ids.size() * sizeof(Id)))
                    return false;
                UpdateCrc(section_crc, state.ids.data(), state.ids.size() * sizeof(Id));
                UpdateCrc(total_crc, state.ids.data(), state.ids.size() * sizeof(Id));
                total_size += state.ids.size() * sizeof(Id);
            }
            if (!state.qdata.empty())
            {
                if (!ReadFull(fd, state.qdata.data(), state.qdata.size()))
                    return false;
                UpdateCrc(section_crc, state.qdata.data(), state.qdata.size());
                UpdateCrc(total_crc, state.qdata.data(), state.qdata.size());
                total_size += state.qdata.size();
            }
            if (!state.qmins.empty())
            {
                if (!ReadFull(fd, state.qmins.data(), state.qmins.size() * sizeof(float)))
                    return false;
                UpdateCrc(section_crc, state.qmins.data(), state.qmins.size() * sizeof(float));
                UpdateCrc(total_crc, state.qmins.data(), state.qmins.size() * sizeof(float));
                total_size += state.qmins.size() * sizeof(float);
            }
            if (!state.qmaxs.empty())
            {
                if (!ReadFull(fd, state.qmaxs.data(), state.qmaxs.size() * sizeof(float)))
                    return false;
                UpdateCrc(section_crc, state.qmaxs.data(), state.qmaxs.size() * sizeof(float));
                UpdateCrc(total_crc, state.qmaxs.data(), state.qmaxs.size() * sizeof(float));
                total_size += state.qmaxs.size() * sizeof(float);
            }
            if (!state.qscales.empty())
            {
                if (!ReadFull(fd, state.qscales.data(), state.qscales.size() * sizeof(float)))
                    return false;
                UpdateCrc(section_crc, state.qscales.data(), state.qscales.size() * sizeof(float));
                UpdateCrc(total_crc, state.qscales.data(), state.qscales.size() * sizeof(float));
                total_size += state.qscales.size() * sizeof(float);
            }
            if (!state.namespace_ids.empty())
            {
                if (!ReadFull(fd, state.namespace_ids.data(), state.namespace_ids.size() * sizeof(std::uint32_t)))
                    return false;
                UpdateCrc(section_crc, state.namespace_ids.data(), state.namespace_ids.size() * sizeof(std::uint32_t));
                UpdateCrc(total_crc, state.namespace_ids.data(), state.namespace_ids.size() * sizeof(std::uint32_t));
                total_size += state.namespace_ids.size() * sizeof(std::uint32_t);
            }
            if (!state.tag_offsets.empty())
            {
                if (!ReadFull(fd, state.tag_offsets.data(), state.tag_offsets.size() * sizeof(std::uint32_t)))
                    return false;
                UpdateCrc(section_crc, state.tag_offsets.data(), state.tag_offsets.size() * sizeof(std::uint32_t));
                UpdateCrc(total_crc, state.tag_offsets.data(), state.tag_offsets.size() * sizeof(std::uint32_t));
                total_size += state.tag_offsets.size() * sizeof(std::uint32_t);
            }
            if (!state.tag_ids.empty())
            {
                if (!ReadFull(fd, state.tag_ids.data(), state.tag_ids.size() * sizeof(TagId)))
                    return false;
                UpdateCrc(section_crc, state.tag_ids.data(), state.tag_ids.size() * sizeof(TagId));
                UpdateCrc(total_crc, state.tag_ids.data(), state.tag_ids.size() * sizeof(TagId));
                total_size += state.tag_ids.size() * sizeof(TagId);
            }
            return true;
        }

        bool BuildDictionary(const SnapshotData &data, std::vector<std::uint32_t> &namespaces, std::vector<TagId> &tags)
        {
            namespaces.clear();
            tags.clear();
            std::vector<std::uint32_t> ns;
            std::vector<TagId> tg;
            for (const auto &shard : data.shards)
            {
                auto collect = [&](const Seed::PersistedState &state)
                {
                    ns.insert(ns.end(), state.namespace_ids.begin(), state.namespace_ids.end());
                    tg.insert(tg.end(), state.tag_ids.begin(), state.tag_ids.end());
                };
                collect(shard.live);
                for (const auto &seg : shard.segments)
                    collect(seg);
            }
            std::sort(ns.begin(), ns.end());
            ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
            std::sort(tg.begin(), tg.end());
            tg.erase(std::unique(tg.begin(), tg.end()), tg.end());
            namespaces = std::move(ns);
            tags = std::move(tg);
            return true;
        }

        bool WriteDictionaryFile(const std::string &path, const std::vector<std::uint32_t> &namespaces, const std::vector<TagId> &tags, std::uint64_t &crc_out, std::string *err)
        {
            int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
            if (fd < 0)
            {
                if (err)
                    *err = std::string("dict open failed: ") + std::strerror(errno);
                return false;
            }
            std::uint64_t crc = 0;
            std::uint64_t ns_count = HostToLe64(static_cast<std::uint64_t>(namespaces.size()));
            std::uint64_t tags_count = HostToLe64(static_cast<std::uint64_t>(tags.size()));
            if (!WriteFull(fd, &ns_count, sizeof(ns_count)) || !WriteFull(fd, &tags_count, sizeof(tags_count)))
            {
                ::close(fd);
                if (err)
                    *err = "dict write header failed";
                return false;
            }
            UpdateCrc(crc, &ns_count, sizeof(ns_count));
            UpdateCrc(crc, &tags_count, sizeof(tags_count));
            if (!namespaces.empty())
            {
                if (!WriteFull(fd, namespaces.data(), namespaces.size() * sizeof(std::uint32_t)))
                {
                    ::close(fd);
                    if (err)
                        *err = "dict write namespaces failed";
                    return false;
                }
                UpdateCrc(crc, namespaces.data(), namespaces.size() * sizeof(std::uint32_t));
            }
            if (!tags.empty())
            {
                if (!WriteFull(fd, tags.data(), tags.size() * sizeof(TagId)))
                {
                    ::close(fd);
                    if (err)
                        *err = "dict write tags failed";
                    return false;
                }
                UpdateCrc(crc, tags.data(), tags.size() * sizeof(TagId));
            }
            if (!FsyncFile(fd))
            {
                ::close(fd);
                if (err)
                    *err = "dict fsync failed";
                return false;
            }
            ::close(fd);
            crc_out = crc;
            return true;
        }

        bool VerifyDictionaryFile(const std::string &path, std::uint64_t expected_crc, std::string *err)
        {
            int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0)
            {
                if (err)
                    *err = "dict open failed";
                return false;
            }
            std::uint64_t crc = 0;
            std::uint64_t ns_count = 0;
            std::uint64_t tags_count = 0;
            if (!ReadU64(fd, ns_count) || !ReadU64(fd, tags_count))
            {
                ::close(fd);
                if (err)
                    *err = "dict header read failed";
                return false;
            }
            std::vector<std::uint8_t> header;
            PutU64(header, ns_count);
            PutU64(header, tags_count);
            UpdateCrc(crc, header.data(), header.size());
            std::vector<std::uint32_t> namespaces;
            std::vector<TagId> tags;
            namespaces.resize(ns_count);
            tags.resize(tags_count);
            if (ns_count > 0)
            {
                if (!ReadFull(fd, namespaces.data(), namespaces.size() * sizeof(std::uint32_t)))
                {
                    ::close(fd);
                    if (err)
                        *err = "dict namespaces read failed";
                    return false;
                }
                UpdateCrc(crc, namespaces.data(), namespaces.size() * sizeof(std::uint32_t));
            }
            if (tags_count > 0)
            {
                if (!ReadFull(fd, tags.data(), tags.size() * sizeof(TagId)))
                {
                    ::close(fd);
                    if (err)
                        *err = "dict tags read failed";
                    return false;
                }
                UpdateCrc(crc, tags.data(), tags.size() * sizeof(TagId));
            }
            ::close(fd);
            if (crc != expected_crc)
            {
                if (err)
                    *err = "dict crc mismatch";
                return false;
            }
            return true;
        }

        bool SerializeManifest(const Manifest &m, std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &header_bytes)
        {
            payload.clear();
            header_bytes.clear();
            auto put_tlv = [&](std::uint32_t type, const std::vector<std::uint8_t> &data)
            {
                PutU32(payload, type);
                PutU64(payload, static_cast<std::uint64_t>(data.size()));
                payload.insert(payload.end(), data.begin(), data.end());
            };

            if (!m.checkpoint_path.empty())
            {
                std::vector<std::uint8_t> data;
                PutU32(data, static_cast<std::uint32_t>(m.checkpoint_path.size()));
                data.insert(data.end(), m.checkpoint_path.begin(), m.checkpoint_path.end());
                put_tlv(1, data);
            }
            {
                std::vector<std::uint8_t> data;
                PutU64(data, m.checkpoint_epoch);
                put_tlv(2, data);
            }
            {
                std::vector<std::uint8_t> data;
                PutU64(data, m.checkpoint_lsn);
                put_tlv(3, data);
            }
            {
                std::vector<std::uint8_t> data;
                PutU32(data, static_cast<std::uint32_t>(m.shard_lsns.size()));
                for (auto l : m.shard_lsns)
                    PutU64(data, l);
                put_tlv(4, data);
            }
            if (!m.dict_path.empty())
            {
                std::vector<std::uint8_t> data;
                PutU32(data, static_cast<std::uint32_t>(m.dict_path.size()));
                data.insert(data.end(), m.dict_path.begin(), m.dict_path.end());
                put_tlv(5, data);
            }
            {
                std::vector<std::uint8_t> data;
                PutU64(data, m.dict_crc64);
                put_tlv(6, data);
            }
            if (!m.indexes.empty())
            {
                std::vector<std::uint8_t> data;
                PutU32(data, static_cast<std::uint32_t>(m.indexes.size()));
                for (const auto &ia : m.indexes)
                {
                    PutU32(data, static_cast<std::uint32_t>(ia.kind.size()));
                    data.insert(data.end(), ia.kind.begin(), ia.kind.end());
                    PutU32(data, static_cast<std::uint32_t>(ia.path.size()));
                    data.insert(data.end(), ia.path.begin(), ia.path.end());
                    PutU64(data, ia.crc64);
                }
                put_tlv(7, data);
            }

        header_bytes.insert(header_bytes.end(), kManifestMagic.begin(), kManifestMagic.end());
        std::uint32_t ver_le = HostToLe32(1);
        std::uint32_t end_le = HostToLe32(kEndianMarker);
            const auto *vptr = reinterpret_cast<const std::uint8_t *>(&ver_le);
            header_bytes.insert(header_bytes.end(), vptr, vptr + sizeof(ver_le));
            const auto *eptr = reinterpret_cast<const std::uint8_t *>(&end_le);
            header_bytes.insert(header_bytes.end(), eptr, eptr + sizeof(end_le));
            return true;
        }
    } // namespace

    bool WriteSnapshotFile(const std::string &path, const SnapshotData &data, SnapshotWriteResult *out, std::string *err)
    {
        crc64_init();
        int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
        if (fd < 0)
        {
            if (err)
                *err = std::string("snapshot open failed: ") + std::strerror(errno);
            return false;
        }

        std::uint64_t total_crc = 0;
        std::uint64_t total_size = 0;

        std::vector<std::uint8_t> header_bytes;
        header_bytes.insert(header_bytes.end(), kSnapMagic.begin(), kSnapMagic.end());
        std::uint32_t ver_le = HostToLe32(kSnapVersion);
        std::uint32_t end_le = HostToLe32(kEndianMarker);
        std::uint32_t flags_le = HostToLe32(0);
        header_bytes.insert(header_bytes.end(), reinterpret_cast<const std::uint8_t *>(&ver_le), reinterpret_cast<const std::uint8_t *>(&ver_le) + sizeof(ver_le));
        header_bytes.insert(header_bytes.end(), reinterpret_cast<const std::uint8_t *>(&end_le), reinterpret_cast<const std::uint8_t *>(&end_le) + sizeof(end_le));
        header_bytes.insert(header_bytes.end(), reinterpret_cast<const std::uint8_t *>(&flags_le), reinterpret_cast<const std::uint8_t *>(&flags_le) + sizeof(flags_le));
        std::uint64_t header_crc = crc64(0, header_bytes.data(), header_bytes.size());
        std::uint64_t header_crc_le = HostToLe64(header_crc);

        if (!WriteFull(fd, header_bytes.data(), header_bytes.size()) || !WriteFull(fd, &header_crc_le, sizeof(header_crc_le)))
        {
            ::close(fd);
            if (err)
                *err = "snapshot header write failed";
            return false;
        }
        UpdateCrc(total_crc, header_bytes.data(), header_bytes.size());
        UpdateCrc(total_crc, &header_crc_le, sizeof(header_crc_le));
        total_size += header_bytes.size() + sizeof(header_crc_le);

        std::uint64_t schema_size = sizeof(std::uint32_t) * 4;
        std::uint64_t schema_crc = 0;
        {
            std::vector<std::uint8_t> schema_buf;
            PutU32(schema_buf, data.schema.dim);
            PutU32(schema_buf, data.schema.metric);
            PutU32(schema_buf, data.schema.shards);
            PutU32(schema_buf, data.schema.index_kind);
            schema_crc = crc64(0, schema_buf.data(), schema_buf.size());
            if (!WriteSectionHeader(fd, kSectionSchema, schema_size, schema_crc, total_crc, total_size))
            {
                ::close(fd);
                if (err)
                    *err = "snapshot schema header failed";
                return false;
            }
            if (!WriteFull(fd, schema_buf.data(), schema_buf.size()))
            {
                ::close(fd);
                if (err)
                    *err = "snapshot schema data failed";
                return false;
            }
            UpdateCrc(total_crc, schema_buf.data(), schema_buf.size());
            total_size += schema_buf.size();
        }

        for (const auto &shard : data.shards)
        {
            std::uint64_t section_crc = 0;
            std::uint64_t section_size = sizeof(std::uint32_t) * 2;
            for (const auto &seg : shard.segments)
                section_size += SeedStateSize(seg);
            section_size += SeedStateSize(shard.live);

            std::uint32_t shard_id_le = HostToLe32(shard.shard_id);
            std::uint32_t segs_le = HostToLe32(static_cast<std::uint32_t>(shard.segments.size()));
            UpdateCrc(section_crc, &shard_id_le, sizeof(shard_id_le));
            UpdateCrc(section_crc, &segs_le, sizeof(segs_le));
            for (const auto &seg : shard.segments)
                UpdateSeedStateCrc(section_crc, seg);
            UpdateSeedStateCrc(section_crc, shard.live);

            if (!WriteSectionHeader(fd, kSectionShard, section_size, section_crc, total_crc, total_size))
            {
                ::close(fd);
                if (err)
                    *err = "snapshot shard header failed";
                return false;
            }

            std::uint64_t shard_crc = 0;
            if (!WriteFull(fd, &shard_id_le, sizeof(shard_id_le)) || !WriteFull(fd, &segs_le, sizeof(segs_le)))
            {
                ::close(fd);
                if (err)
                    *err = "snapshot shard header payload failed";
                return false;
            }
            UpdateCrc(shard_crc, &shard_id_le, sizeof(shard_id_le));
            UpdateCrc(shard_crc, &segs_le, sizeof(segs_le));
            UpdateCrc(total_crc, &shard_id_le, sizeof(shard_id_le));
            UpdateCrc(total_crc, &segs_le, sizeof(segs_le));
            total_size += sizeof(shard_id_le) + sizeof(segs_le);

            for (const auto &seg : shard.segments)
            {
                if (!WriteSeedState(fd, seg, shard_crc, total_crc, total_size))
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot segment write failed";
                    return false;
                }
            }
            if (!WriteSeedState(fd, shard.live, shard_crc, total_crc, total_size))
            {
                ::close(fd);
                if (err)
                    *err = "snapshot live write failed";
                return false;
            }
        }

        std::vector<std::uint32_t> namespaces;
        std::vector<TagId> tags;
        BuildDictionary(data, namespaces, tags);
        std::uint64_t dict_size = sizeof(std::uint64_t) * 2 + namespaces.size() * sizeof(std::uint32_t) + tags.size() * sizeof(TagId);
        std::uint64_t dict_crc = 0;
        {
            std::vector<std::uint8_t> buf;
            PutU64(buf, namespaces.size());
            PutU64(buf, tags.size());
            dict_crc = crc64(0, buf.data(), buf.size());
            if (!namespaces.empty())
                dict_crc = crc64(dict_crc,
                                 reinterpret_cast<const unsigned char *>(namespaces.data()),
                                 namespaces.size() * sizeof(std::uint32_t));
            if (!tags.empty())
                dict_crc = crc64(dict_crc,
                                 reinterpret_cast<const unsigned char *>(tags.data()),
                                 tags.size() * sizeof(TagId));
        }
        if (!WriteSectionHeader(fd, kSectionDictionary, dict_size, dict_crc, total_crc, total_size))
        {
            ::close(fd);
            if (err)
                *err = "snapshot dict header failed";
            return false;
        }
        std::uint64_t ns_le = HostToLe64(namespaces.size());
        std::uint64_t tags_le = HostToLe64(tags.size());
        if (!WriteFull(fd, &ns_le, sizeof(ns_le)) || !WriteFull(fd, &tags_le, sizeof(tags_le)))
        {
            ::close(fd);
            if (err)
                *err = "snapshot dict data failed";
            return false;
        }
        UpdateCrc(total_crc, &ns_le, sizeof(ns_le));
        UpdateCrc(total_crc, &tags_le, sizeof(tags_le));
        total_size += sizeof(ns_le) + sizeof(tags_le);
        if (!namespaces.empty())
        {
            if (!WriteFull(fd, namespaces.data(), namespaces.size() * sizeof(std::uint32_t)))
            {
                ::close(fd);
                if (err)
                    *err = "snapshot dict namespaces failed";
                return false;
            }
            UpdateCrc(total_crc, namespaces.data(), namespaces.size() * sizeof(std::uint32_t));
            total_size += namespaces.size() * sizeof(std::uint32_t);
        }
        if (!tags.empty())
        {
            if (!WriteFull(fd, tags.data(), tags.size() * sizeof(TagId)))
            {
                ::close(fd);
                if (err)
                    *err = "snapshot dict tags failed";
                return false;
            }
            UpdateCrc(total_crc, tags.data(), tags.size() * sizeof(TagId));
            total_size += tags.size() * sizeof(TagId);
        }

        std::uint64_t total_size_le = HostToLe64(total_size);
        std::uint64_t total_crc_le = HostToLe64(total_crc);
        if (!WriteFull(fd, &total_size_le, sizeof(total_size_le)) ||
            !WriteFull(fd, &total_crc_le, sizeof(total_crc_le)) ||
            !WriteFull(fd, kFooterMagic.data(), kFooterMagic.size()))
        {
            ::close(fd);
            if (err)
                *err = "snapshot footer write failed";
            return false;
        }
        total_size += sizeof(total_size_le) + sizeof(total_crc_le) + kFooterMagic.size();
        ::close(fd);
        if (out)
        {
            out->total_size = total_size;
            out->total_crc64 = total_crc;
        }
        return true;
    }

    bool ReadSnapshotFile(const std::string &path, SnapshotData &out, std::string *err)
    {
        crc64_init();
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
        {
            if (err)
                *err = "snapshot open failed";
            return false;
        }

        struct stat st;
        if (::fstat(fd, &st) != 0)
        {
            ::close(fd);
            if (err)
                *err = "snapshot stat failed";
            return false;
        }
        std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);

        std::array<char, 10> magic{};
        if (!ReadFull(fd, magic.data(), magic.size()))
        {
            ::close(fd);
            if (err)
                *err = "snapshot magic read failed";
            return false;
        }
        if (magic != kSnapMagic)
        {
            ::close(fd);
            if (err)
                *err = "snapshot magic mismatch";
            return false;
        }
        std::uint32_t version = 0;
        std::uint32_t endian = 0;
        std::uint32_t flags = 0;
        std::uint64_t header_crc = 0;
        if (!ReadU32(fd, version) || !ReadU32(fd, endian) || !ReadU32(fd, flags) || !ReadU64(fd, header_crc))
        {
            ::close(fd);
            if (err)
                *err = "snapshot header read failed";
            return false;
        }
        if (version != kSnapVersion || endian != kEndianMarker)
        {
            ::close(fd);
            if (err)
                *err = "snapshot version/endian mismatch";
            return false;
        }
        std::vector<std::uint8_t> header_bytes;
        header_bytes.insert(header_bytes.end(), magic.begin(), magic.end());
        PutU32(header_bytes, version);
        PutU32(header_bytes, endian);
        PutU32(header_bytes, flags);
        if (crc64(0, header_bytes.data(), header_bytes.size()) != header_crc)
        {
            ::close(fd);
            if (err)
                *err = "snapshot header crc mismatch";
            return false;
        }

        SnapshotData data;
        std::uint64_t total_crc = 0;
        std::uint64_t total_size = 0;
        UpdateCrc(total_crc, header_bytes.data(), header_bytes.size());
        std::uint64_t header_crc_le = HostToLe64(header_crc);
        UpdateCrc(total_crc, &header_crc_le, sizeof(header_crc_le));
        total_size += header_bytes.size() + sizeof(header_crc_le);

        const std::uint64_t footer_size = sizeof(std::uint64_t) * 2 + kFooterMagic.size();
        while (total_size + footer_size < file_size)
        {
            std::uint32_t type = 0;
            std::uint64_t size = 0;
            std::uint64_t crc = 0;
            if (!ReadU32(fd, type) || !ReadU64(fd, size) || !ReadU64(fd, crc))
            {
                ::close(fd);
                if (err)
                    *err = "snapshot section header read failed";
                return false;
            }
            std::uint32_t type_le = HostToLe32(type);
            std::uint64_t size_le = HostToLe64(size);
            std::uint64_t crc_le = HostToLe64(crc);
            UpdateCrc(total_crc, &type_le, sizeof(type_le));
            UpdateCrc(total_crc, &size_le, sizeof(size_le));
            UpdateCrc(total_crc, &crc_le, sizeof(crc_le));
            total_size += sizeof(type_le) + sizeof(size_le) + sizeof(crc_le);

            if (type == kSectionSchema)
            {
                const std::uint64_t section_start = total_size;
                std::uint32_t dim = 0;
                std::uint32_t metric = 0;
                std::uint32_t shards = 0;
                std::uint32_t index_kind = 0;
                if (!ReadU32(fd, dim) || !ReadU32(fd, metric) || !ReadU32(fd, shards) || !ReadU32(fd, index_kind))
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot schema read failed";
                    return false;
                }
                std::vector<std::uint8_t> buf;
                PutU32(buf, dim);
                PutU32(buf, metric);
                PutU32(buf, shards);
                PutU32(buf, index_kind);
                if (crc64(0, buf.data(), buf.size()) != crc)
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot schema crc mismatch";
                    return false;
                }
                UpdateCrc(total_crc, buf.data(), buf.size());
                total_size += buf.size();
                data.schema.dim = dim;
                data.schema.metric = metric;
                data.schema.shards = shards;
                data.schema.index_kind = index_kind;
                if (total_size - section_start != size)
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot schema size mismatch";
                    return false;
                }
                continue;
            }
            if (type == kSectionShard)
            {
                const std::uint64_t section_start = total_size;
                std::uint64_t section_crc = 0;
                std::uint32_t shard_id = 0;
                std::uint32_t segs = 0;
                if (!ReadU32(fd, shard_id) || !ReadU32(fd, segs))
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot shard header read failed";
                    return false;
                }
                std::vector<std::uint8_t> header;
                PutU32(header, shard_id);
                PutU32(header, segs);
                UpdateCrc(section_crc, header.data(), header.size());
                UpdateCrc(total_crc, header.data(), header.size());
                total_size += header.size();
                ShardSnapshot shard;
                shard.shard_id = shard_id;
                shard.segments.reserve(segs);
                for (std::uint32_t i = 0; i < segs; ++i)
                {
                    Seed::PersistedState seg;
                    if (!ReadSeedState(fd, seg, section_crc, total_crc, total_size))
                    {
                        ::close(fd);
                        if (err)
                            *err = "snapshot segment read failed";
                        return false;
                    }
                    shard.segments.push_back(std::move(seg));
                }
                if (!ReadSeedState(fd, shard.live, section_crc, total_crc, total_size))
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot live read failed";
                    return false;
                }
                if (section_crc != crc)
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot shard crc mismatch";
                    return false;
                }
                if (total_size - section_start != size)
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot shard size mismatch";
                    return false;
                }
                data.shards.push_back(std::move(shard));
                continue;
            }
            if (type == kSectionDictionary)
            {
                const std::uint64_t section_start = total_size;
                std::uint64_t ns = 0;
                std::uint64_t tags = 0;
                if (!ReadU64(fd, ns) || !ReadU64(fd, tags))
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot dict header read failed";
                    return false;
                }
                std::vector<std::uint8_t> buf;
                PutU64(buf, ns);
                PutU64(buf, tags);
                std::uint64_t calc = crc64(0, buf.data(), buf.size());
                std::vector<std::uint32_t> namespaces(ns);
                std::vector<TagId> tag_ids(tags);
                if (ns > 0)
                {
                    if (!ReadFull(fd, namespaces.data(), namespaces.size() * sizeof(std::uint32_t)))
                    {
                        ::close(fd);
                        if (err)
                            *err = "snapshot dict read failed";
                        return false;
                    }
                    calc = crc64(calc,
                                 reinterpret_cast<const unsigned char *>(namespaces.data()),
                                 namespaces.size() * sizeof(std::uint32_t));
                }
                if (tags > 0)
                {
                    if (!ReadFull(fd, tag_ids.data(), tag_ids.size() * sizeof(TagId)))
                    {
                        ::close(fd);
                        if (err)
                            *err = "snapshot dict read failed";
                        return false;
                    }
                    calc = crc64(calc,
                                 reinterpret_cast<const unsigned char *>(tag_ids.data()),
                                 tag_ids.size() * sizeof(TagId));
                }
                if (calc != crc)
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot dict crc mismatch";
                    return false;
                }
                UpdateCrc(total_crc, buf.data(), buf.size());
                total_size += buf.size();
                if (!namespaces.empty())
                {
                    UpdateCrc(total_crc, namespaces.data(), namespaces.size() * sizeof(std::uint32_t));
                    total_size += namespaces.size() * sizeof(std::uint32_t);
                }
                if (!tag_ids.empty())
                {
                    UpdateCrc(total_crc, tag_ids.data(), tag_ids.size() * sizeof(TagId));
                    total_size += tag_ids.size() * sizeof(TagId);
                }
                if (total_size - section_start != size)
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot dict size mismatch";
                    return false;
                }
                continue;
            }
            if (size > 0)
            {
                std::vector<std::uint8_t> skip(size);
                if (!ReadFull(fd, skip.data(), skip.size()))
                {
                    ::close(fd);
                    if (err)
                        *err = "snapshot unknown section read failed";
                    return false;
                }
                UpdateCrc(total_crc, skip.data(), skip.size());
                total_size += skip.size();
            }
        }

        std::uint64_t total_size_file = 0;
        std::uint64_t total_crc_file = 0;
        if (!ReadU64(fd, total_size_file) || !ReadU64(fd, total_crc_file))
        {
            ::close(fd);
            if (err)
                *err = "snapshot footer read failed";
            return false;
        }
        std::array<char, 8> footer{};
        if (!ReadFull(fd, footer.data(), footer.size()))
        {
            ::close(fd);
            if (err)
                *err = "snapshot footer magic read failed";
            return false;
        }
        ::close(fd);
        if (footer != kFooterMagic)
        {
            if (err)
                *err = "snapshot footer magic mismatch";
            return false;
        }
        if (total_size != total_size_file)
        {
            if (err)
                *err = "snapshot total size mismatch";
            return false;
        }
        if (total_crc != total_crc_file)
        {
            if (err)
                *err = "snapshot total crc mismatch";
            return false;
        }
        out = std::move(data);
        return true;
    }

    bool VerifySnapshotFile(const std::string &path, std::string *err)
    {
        SnapshotData tmp;
        return ReadSnapshotFile(path, tmp, err);
    }

    bool CommitCheckpointAtomically(const std::string &db_dir,
                                    const SnapshotData &snapshot,
                                    const std::vector<IndexArtifactData> &index_artifacts,
                                    CommitResult *out,
                                    std::string *err)
    {
        crc64_init();
        DbPaths paths = MakeDbPaths(db_dir);
        if (!EnsureDbDirs(paths))
        {
            if (err)
                *err = "db dirs missing";
            return false;
        }

        auto now = std::chrono::system_clock::now().time_since_epoch();
        std::uint64_t epoch = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(now).count());
        std::uint64_t lsn = 0;
        std::vector<std::uint64_t> shard_lsns = snapshot.shard_lsns;
        if (shard_lsns.empty())
            shard_lsns.resize(snapshot.shards.size(), 0);
        for (auto shard_lsn : shard_lsns)
            lsn = std::max(lsn, shard_lsn);

        std::string checkpoint_name = "chk_" + std::to_string(epoch) + "_" + std::to_string(lsn) + ".pomai";
        std::string checkpoint_path = paths.checkpoints_dir + "/" + checkpoint_name;
        std::string checkpoint_tmp = checkpoint_path + ".tmp";

        SnapshotWriteResult snap_res;
        if (!WriteSnapshotFile(checkpoint_tmp, snapshot, &snap_res, err))
            return false;
        int fd = ::open(checkpoint_tmp.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0 || !FsyncFile(fd))
        {
            if (fd >= 0)
                ::close(fd);
            if (err)
                *err = "snapshot fsync failed";
            return false;
        }
        ::close(fd);
        if (FailpointHit("after_snapshot_fsync"))
        {
            if (err)
                *err = "failpoint after_snapshot_fsync";
            return false;
        }
        if (!AtomicRename(checkpoint_tmp, checkpoint_path))
        {
            if (err)
                *err = "snapshot rename failed";
            return false;
        }
        if (!FsyncDirPath(paths.checkpoints_dir))
        {
            if (err)
                *err = "snapshot dir fsync failed";
            return false;
        }

        std::vector<std::uint32_t> namespaces;
        std::vector<TagId> tags;
        BuildDictionary(snapshot, namespaces, tags);
        std::string dict_name = "dict_" + std::to_string(epoch) + "_" + std::to_string(lsn) + ".bin";
        std::string dict_path = paths.meta_dir + "/" + dict_name;
        std::string dict_tmp = dict_path + ".tmp";
        std::uint64_t dict_crc = 0;
        if (!WriteDictionaryFile(dict_tmp, namespaces, tags, dict_crc, err))
            return false;
        if (!AtomicRename(dict_tmp, dict_path))
        {
            if (err)
                *err = "dict rename failed";
            return false;
        }
        if (!FsyncDirPath(paths.meta_dir))
        {
            if (err)
                *err = "dict dir fsync failed";
            return false;
        }

        std::vector<IndexArtifact> index_manifest;
        for (const auto &artifact : index_artifacts)
        {
            std::string idx_name = "idx_" + std::to_string(epoch) + "_" + std::to_string(lsn) + "_" + artifact.kind + ".bin";
            std::string idx_path = paths.indexes_dir + "/" + idx_name;
            std::string idx_tmp = idx_path + ".tmp";
            int ifd = ::open(idx_tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
            if (ifd < 0)
            {
                if (err)
                    *err = "index tmp open failed";
                return false;
            }
            std::uint64_t crc = artifact.bytes.empty() ? 0 : crc64(0, artifact.bytes.data(), artifact.bytes.size());
            if (!artifact.bytes.empty() && !WriteFull(ifd, artifact.bytes.data(), artifact.bytes.size()))
            {
                ::close(ifd);
                if (err)
                    *err = "index write failed";
                return false;
            }
            if (!FsyncFile(ifd))
            {
                ::close(ifd);
                if (err)
                    *err = "index fsync failed";
                return false;
            }
            ::close(ifd);
            if (!AtomicRename(idx_tmp, idx_path))
            {
                if (err)
                    *err = "index rename failed";
                return false;
            }
            if (!FsyncDirPath(paths.indexes_dir))
            {
                if (err)
                    *err = "index dir fsync failed";
                return false;
            }
            index_manifest.push_back(IndexArtifact{artifact.kind, idx_name, crc});
        }

        Manifest manifest;
        manifest.version = 1;
        manifest.checkpoint_path = "checkpoints/" + checkpoint_name;
        manifest.checkpoint_epoch = epoch;
        manifest.checkpoint_lsn = lsn;
        manifest.shard_lsns = shard_lsns;
        manifest.dict_path = "meta/" + dict_name;
        manifest.dict_crc64 = dict_crc;
        manifest.indexes = std::move(index_manifest);

        std::vector<std::uint8_t> payload;
        std::vector<std::uint8_t> header_bytes;
        if (!SerializeManifest(manifest, payload, header_bytes))
        {
            if (err)
                *err = "manifest serialize failed";
            return false;
        }
        std::uint64_t header_crc = crc64(0, header_bytes.data(), header_bytes.size());
        std::uint64_t payload_crc = payload.empty() ? 0 : crc64(0, payload.data(), payload.size());

        std::string manifest_tmp = paths.manifest_path + ".tmp";
        int mfd = ::open(manifest_tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
        if (mfd < 0)
        {
            if (err)
                *err = "manifest tmp open failed";
            return false;
        }
        std::uint64_t header_crc_le = HostToLe64(header_crc);
        std::uint64_t payload_len_le = HostToLe64(static_cast<std::uint64_t>(payload.size()));
        std::uint64_t payload_crc_le = HostToLe64(payload_crc);
        if (!WriteFull(mfd, header_bytes.data(), header_bytes.size()) ||
            !WriteFull(mfd, &header_crc_le, sizeof(header_crc_le)) ||
            !WriteFull(mfd, &payload_len_le, sizeof(payload_len_le)) ||
            !WriteFull(mfd, &payload_crc_le, sizeof(payload_crc_le)) ||
            (!payload.empty() && !WriteFull(mfd, payload.data(), payload.size())))
        {
            ::close(mfd);
            if (err)
                *err = "manifest write failed";
            return false;
        }
        if (!FsyncFile(mfd))
        {
            ::close(mfd);
            if (err)
                *err = "manifest fsync failed";
            return false;
        }
        ::close(mfd);
        if (FailpointHit("before_manifest_rename"))
        {
            if (err)
                *err = "failpoint before_manifest_rename";
            return false;
        }
        if (!AtomicRename(manifest_tmp, paths.manifest_path))
        {
            if (err)
                *err = "manifest rename failed";
            return false;
        }
        if (!FsyncDirPath(paths.db_dir))
        {
            if (err)
                *err = "manifest dir fsync failed";
            return false;
        }

        if (out)
            out->manifest = std::move(manifest);
        return true;
    }
} // namespace pomai::storage
