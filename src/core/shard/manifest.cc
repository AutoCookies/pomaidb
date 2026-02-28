#include "core/shard/manifest.h"
#include <filesystem>
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

    struct ManifestContent {
        uint32_t crc;
        std::vector<std::string> segments;
    };
}

pomai::Status ShardManifest::Load(const std::string& shard_dir, std::vector<std::string>* out_segments) {
    fs::path curr = fs::path(shard_dir) / "manifest.current";
    if (!fs::exists(curr)) {
        out_segments->clear();
        return Status::Ok();
    }

    std::unique_ptr<SequentialFile> file;
    auto st = PosixIOProvider::NewSequentialFile(curr, &file);
    if (!st.ok()) return st;

    char scratch[4096];
    Slice result;
    st = file->Read(4096, &result, scratch);
    if (!st.ok()) return st;

    std::string_view content = result.ToStringView();
    if (!content.starts_with(kManifestHeader)) {
        return Status::Corruption("Manifest header mismatch");
    }

    // Simplified parsing: one segment per line after header and CRC
    // In a real distillation, we'd use a more robust append-only log format.
    // For Phase 4, we maintain the "Atomic Rename" pattern but use IOProvider.
    std::string_view remaining = content.substr(kManifestHeader.size() + 1);
    size_t next_line = remaining.find('\n');
    if (next_line == std::string_view::npos) return Status::Corruption("Manifest truncated");

    out_segments->clear();
    std::string_view payload = remaining.substr(next_line + 1);
    size_t start = 0;
    while (start < payload.size()) {
        size_t end = payload.find('\n', start);
        if (end == std::string_view::npos) break;
        out_segments->push_back(std::string(payload.substr(start, end - start)));
        start = end + 1;
    }

    return Status::Ok();
}

pomai::Status ShardManifest::Commit(const std::string& shard_dir, const std::vector<std::string>& segments) {
    fs::path tmp = fs::path(shard_dir) / "manifest.tmp";
    fs::path curr = fs::path(shard_dir) / "manifest.current";

    std::unique_ptr<WritableFile> file;
    auto st = PosixIOProvider::NewWritableFile(tmp, &file);
    if (!st.ok()) return st;

    std::string buffer;
    buffer.append(kManifestHeader);
    buffer.append("\nCRC: 0\n"); // Placeholder for distillation simplicity
    for (const auto& s : segments) {
        buffer.append(s);
        buffer.append("\n");
    }

    st = file->Append(Slice(buffer));
    if (!st.ok()) return st;
    st = file->Sync();
    if (!st.ok()) return st;
    st = file->Close();
    if (!st.ok()) return st;

    std::error_code ec;
    fs::rename(tmp, curr, ec);
    if (ec) return Status::IOError("Manifest rename failed");

    // Fsync directory to ensure metadata is durable
    int dir_fd = open(shard_dir.c_str(), O_DIRECTORY | O_RDONLY);
    if (dir_fd >= 0) {
        fsync(dir_fd);
        close(dir_fd);
    }

    return Status::Ok();
}

} // namespace pomai::core
