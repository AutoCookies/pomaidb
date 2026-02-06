#include "core/routing/routing_persist.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "util/crc32c.h"
#include "util/posix_file.h"

namespace pomai::core::routing {
namespace fs = std::filesystem;

namespace {
constexpr char kMagic[8] = {'P','R','O','U','T','E','1','\0'};

void AppendU32(std::string* s, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) s->push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
}
void AppendU64(std::string* s, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) s->push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
}
bool ReadU32(const std::string& s, std::size_t* off, std::uint32_t* out) {
    if (*off + 4 > s.size()) return false;
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(static_cast<unsigned char>(s[*off + i])) << (8 * i);
    *off += 4; *out = v; return true;
}
bool ReadU64(const std::string& s, std::size_t* off, std::uint64_t* out) {
    if (*off + 8 > s.size()) return false;
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(static_cast<unsigned char>(s[*off + i])) << (8 * i);
    *off += 8; *out = v; return true;
}

std::string Serialize(const RoutingTable& t) {
    std::string out;
    out.append(kMagic, sizeof(kMagic));
    AppendU64(&out, t.epoch);
    AppendU32(&out, t.k);
    AppendU32(&out, t.dim);
    for (float v : t.centroids) out.append(reinterpret_cast<const char*>(&v), sizeof(float));
    for (std::uint32_t v : t.owner_shard) AppendU32(&out, v);
    for (std::uint64_t v : t.counts) AppendU64(&out, v);
    return out;
}

bool Deserialize(const std::string& payload, RoutingTable* t) {
    if (payload.size() < sizeof(kMagic) + 16) return false;
    if (std::memcmp(payload.data(), kMagic, sizeof(kMagic)) != 0) return false;
    std::size_t off = sizeof(kMagic);
    if (!ReadU64(payload, &off, &t->epoch)) return false;
    if (!ReadU32(payload, &off, &t->k)) return false;
    if (!ReadU32(payload, &off, &t->dim)) return false;

    const std::size_t csz = static_cast<std::size_t>(t->k) * t->dim;
    const std::size_t centroid_bytes = csz * sizeof(float);
    if (off + centroid_bytes > payload.size()) return false;
    t->centroids.resize(csz);
    std::memcpy(t->centroids.data(), payload.data() + off, centroid_bytes);
    off += centroid_bytes;

    t->owner_shard.resize(t->k);
    for (std::uint32_t i = 0; i < t->k; ++i) if (!ReadU32(payload, &off, &t->owner_shard[i])) return false;
    t->counts.resize(t->k);
    for (std::uint32_t i = 0; i < t->k; ++i) if (!ReadU64(payload, &off, &t->counts[i])) return false;

    return off == payload.size() && t->Valid();
}

pomai::Status AtomicWriteFileWithCrc(const std::string& final_path, const std::string& payload) {
    const std::string tmp = final_path + ".tmp";
    pomai::util::PosixFile pf;
    auto st = pomai::util::PosixFile::CreateTrunc(tmp, &pf);
    if (!st.ok()) return st;

    st = pf.PWrite(0, payload.data(), payload.size());
    if (!st.ok()) return st;
    const std::uint32_t crc = pomai::util::Crc32c(payload.data(), payload.size());
    char crc_buf[4];
    crc_buf[0] = static_cast<char>(crc & 0xFFu);
    crc_buf[1] = static_cast<char>((crc >> 8) & 0xFFu);
    crc_buf[2] = static_cast<char>((crc >> 16) & 0xFFu);
    crc_buf[3] = static_cast<char>((crc >> 24) & 0xFFu);
    st = pf.PWrite(payload.size(), crc_buf, 4);
    if (!st.ok()) return st;
    st = pf.SyncData();
    if (!st.ok()) return st;
    st = pf.Close();
    if (!st.ok()) return st;

    std::error_code ec;
    fs::rename(tmp, final_path, ec);
    if (ec) return pomai::Status::IOError("rename failed");
    return pomai::util::FsyncDir(fs::path(final_path).parent_path().string());
}

std::optional<RoutingTable> LoadPath(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return std::nullopt;
    in.seekg(0, std::ios::end);
    const std::streamoff n = in.tellg();
    in.seekg(0, std::ios::beg);
    if (n < 4) return std::nullopt;
    std::string buf(static_cast<std::size_t>(n), '\0');
    in.read(buf.data(), n);
    if (!in.good()) return std::nullopt;
    const std::size_t payload_len = static_cast<std::size_t>(n) - 4;
    std::uint32_t stored_crc = 0;
    for (int i = 0; i < 4; ++i) {
        stored_crc |= static_cast<std::uint32_t>(static_cast<unsigned char>(buf[payload_len + i])) << (8 * i);
    }
    const std::uint32_t computed = pomai::util::Crc32c(buf.data(), payload_len);
    if (stored_crc != computed) return std::nullopt;
    RoutingTable table;
    if (!Deserialize(buf.substr(0, payload_len), &table)) return std::nullopt;
    return table;
}

} // namespace

std::string RoutingPath(const std::string& root_path) {
    return (fs::path(root_path) / "ROUTING").string();
}
std::string RoutingPrevPath(const std::string& root_path) {
    return (fs::path(root_path) / "ROUTING.prev").string();
}

pomai::Status SaveRoutingTableAtomic(const std::string& root_path, const RoutingTable& table, bool keep_prev) {
    if (!table.Valid()) return pomai::Status::InvalidArgument("invalid routing table");
    const auto final_path = RoutingPath(root_path);
    const auto prev_path = RoutingPrevPath(root_path);
    std::error_code ec;
    if (keep_prev && fs::exists(final_path, ec) && !ec) {
        fs::copy_file(final_path, prev_path, fs::copy_options::overwrite_existing, ec);
        (void)ec; // best effort
    }
    return AtomicWriteFileWithCrc(final_path, Serialize(table));
}

std::optional<RoutingTable> LoadRoutingTable(const std::string& root_path) {
    return LoadPath(RoutingPath(root_path));
}

std::optional<RoutingTable> LoadRoutingPrevTable(const std::string& root_path) {
    return LoadPath(RoutingPrevPath(root_path));
}

} // namespace pomai::core::routing
