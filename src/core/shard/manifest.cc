#include "core/shard/manifest.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/storage/io_provider.h"
#include "util/crc32c.h"

namespace pomai::core {

namespace fs = std::filesystem;
using namespace pomai::storage;

namespace {
    constexpr std::string_view kManifestHeader = "pomai.manifest.v2";
    constexpr std::string_view kManifestHeaderAlt = "pomai.shard_manifest.v2";
    constexpr size_t kCrcSize = 4;

    // Read entire file into string; returns false on read error.
    static bool ReadFile(const fs::path& path, std::string* out) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) return false;
        in.seekg(0, std::ios::end);
        auto n = in.tellg();
        in.seekg(0, std::ios::beg);
        if (n <= 0) {
            out->clear();
            return true;
        }
        out->resize(static_cast<size_t>(n));
        return static_cast<bool>(in.read(out->data(), n));
    }

    // Parse segment list from payload (after header line). One segment name per line.
    static void ParseSegmentLines(std::string_view payload, std::vector<std::string>* out_segments) {
        out_segments->clear();
        size_t start = 0;
        while (start < payload.size()) {
            size_t end = payload.find('\n', start);
            if (end == std::string_view::npos) break;
            std::string_view line = payload.substr(start, end - start);
            if (!line.empty())
                out_segments->push_back(std::string(line));
            start = end + 1;
        }
    }

    // Try to load from one manifest file. Returns true if loaded and CRC valid (or legacy format).
    static pomai::Status TryLoadOne(const fs::path& path, std::vector<std::string>* out_segments) {
        std::string raw;
        if (!ReadFile(path, &raw))
            return pomai::Status::IOError("shard manifest read failed");

        if (raw.empty()) {
            out_segments->clear();
            return pomai::Status::Ok();
        }

        // Legacy format: no header, just segment names per line (no trailing CRC).
        std::string_view content = raw;
        if (content.size() < kCrcSize || (!content.starts_with(kManifestHeader) && !content.starts_with(kManifestHeaderAlt))) {
            ParseSegmentLines(content, out_segments);
            return pomai::Status::Ok();
        }

        // New format: content + 4-byte CRC at end.
        const size_t content_len = raw.size() - kCrcSize;
        std::string_view content_part(raw.data(), content_len);
        uint32_t stored_crc = 0;
        std::memcpy(&stored_crc, raw.data() + content_len, kCrcSize);
        uint32_t computed = pomai::util::Crc32c(raw.data(), content_len);
        if (computed != stored_crc)
            return pomai::Status::Corruption("shard manifest CRC mismatch");

        // Payload after first line (header).
        size_t first_nl = content_part.find('\n');
        if (first_nl == std::string_view::npos)
            return pomai::Status::Corruption("shard manifest truncated");
        std::string_view payload = content_part.substr(first_nl + 1);
        ParseSegmentLines(payload, out_segments);
        return pomai::Status::Ok();
    }
}

pomai::Status ShardManifest::Load(const std::string& shard_dir, std::vector<std::string>* out_segments) {
    fs::path curr = fs::path(shard_dir) / "manifest.current";
    fs::path prev = fs::path(shard_dir) / "manifest.prev";

    if (!fs::exists(curr)) {
        out_segments->clear();
        return pomai::Status::Ok();
    }

    pomai::Status st = TryLoadOne(curr, out_segments);
    if (st.ok())
        return st;
    // On corruption/CRC failure, fall back to manifest.prev for crash safety.
    if (fs::exists(prev)) {
        st = TryLoadOne(prev, out_segments);
        if (st.ok())
            return st;
    }
    return st;
}

pomai::Status ShardManifest::Commit(const std::string& shard_dir, const std::vector<std::string>& segments) {
    fs::path tmp = fs::path(shard_dir) / "manifest.tmp";
    fs::path curr = fs::path(shard_dir) / "manifest.current";
    fs::path prev = fs::path(shard_dir) / "manifest.prev";

    std::string buffer;
    buffer.append(kManifestHeader);
    buffer.append("\n");
    for (const auto& s : segments) {
        buffer.append(s);
        buffer.append("\n");
    }
    uint32_t crc = pomai::util::Crc32c(buffer.data(), buffer.size());
    buffer.push_back(static_cast<char>(crc & 0xFF));
    buffer.push_back(static_cast<char>((crc >> 8) & 0xFF));
    buffer.push_back(static_cast<char>((crc >> 16) & 0xFF));
    buffer.push_back(static_cast<char>((crc >> 24) & 0xFF));

    std::unique_ptr<WritableFile> file;
    pomai::Status st = PosixIOProvider::NewWritableFile(tmp, &file);
    if (!st.ok()) return st;

    st = file->Append(pomai::Slice(buffer));
    if (!st.ok()) return st;
    st = file->Sync();
    if (!st.ok()) return st;
    st = file->Close();
    if (!st.ok()) return st;

    std::error_code ec;
    if (fs::exists(curr)) {
        fs::rename(curr, prev, ec);
        if (ec) return pomai::Status::IOError("Manifest prev rename failed");
    }
    fs::rename(tmp, curr, ec);
    if (ec) return pomai::Status::IOError("Manifest rename failed");

    int dir_fd = open(shard_dir.c_str(), O_DIRECTORY | O_RDONLY);
    if (dir_fd >= 0) {
        fsync(dir_fd);
        close(dir_fd);
    }

    return pomai::Status::Ok();
}

} // namespace pomai::core
