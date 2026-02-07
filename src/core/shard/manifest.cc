#include "core/shard/manifest.h"
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cctype>
#include <charconv>
#include "util/posix_file.h" // For FsyncDir
#include "util/crc32c.h"

namespace pomai::core
{
    namespace fs = std::filesystem;

    namespace {
        constexpr std::string_view kManifestHeader = "pomai.shard_manifest.v2";

        bool IsDigits(std::string_view s) {
            if (s.empty()) return false;
            for (char c : s) {
                if (!std::isdigit(static_cast<unsigned char>(c))) return false;
            }
            return true;
        }

        std::string BuildPayload(const std::vector<std::string>& segments) {
            std::string payload;
            for (const auto& s : segments) {
                payload += s;
                payload += "\n";
            }
            return payload;
        }

        pomai::Status ParseV2Manifest(const std::string& content, std::vector<std::string>* out_segments) {
            std::istringstream in(content);
            std::string line;
            if (!std::getline(in, line) || line != kManifestHeader) {
                return pomai::Status::Corruption("bad shard manifest header");
            }

            if (!std::getline(in, line) || line.rfind("crc32c ", 0) != 0) {
                return pomai::Status::Corruption("missing shard manifest crc32c");
            }

            const std::string crc_text = line.substr(7);
            if (!IsDigits(crc_text)) {
                return pomai::Status::Corruption("invalid shard manifest crc32c value");
            }

            std::string payload;
            while (std::getline(in, line)) {
                payload += line;
                payload += "\n";
            }

            std::uint32_t expected_crc = 0;
            const char* begin = crc_text.data();
            const char* end = crc_text.data() + crc_text.size();
            auto [ptr, ec] = std::from_chars(begin, end, expected_crc);
            if (ec != std::errc{} || ptr != end) {
                return pomai::Status::Corruption("invalid shard manifest crc32c parse");
            }
            const std::uint32_t actual_crc = pomai::util::Crc32c(payload.data(), payload.size());
            if (expected_crc != actual_crc) {
                return pomai::Status::Corruption("shard manifest crc32c mismatch");
            }

            out_segments->clear();
            std::istringstream payload_in(payload);
            while (std::getline(payload_in, line)) {
                if (!line.empty()) out_segments->push_back(line);
            }
            return pomai::Status::Ok();
        }

        pomai::Status ParseLegacyManifest(const std::string& content, std::vector<std::string>* out_segments) {
            out_segments->clear();
            std::istringstream in(content);
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty()) out_segments->push_back(line);
            }
            return pomai::Status::Ok();
        }
    }

    pomai::Status ShardManifest::Load(const std::string &shard_dir, std::vector<std::string> *out_segments)
    {
        fs::path p = fs::path(shard_dir) / "manifest.current";
        fs::path prev = fs::path(shard_dir) / "manifest.prev";
        std::error_code ec;
        if (!fs::exists(p, ec))
        {
            if (ec) return pomai::Status::IOError(ec.message());
            out_segments->clear();
            return pomai::Status::Ok();
        }

        auto parse_one = [&](const fs::path& path) -> pomai::Status {
            std::ifstream in(path);
            if (!in.is_open()) return pomai::Status::IOError("failed to open " + path.filename().string());
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (content.rfind(std::string(kManifestHeader), 0) == 0) {
                return ParseV2Manifest(content, out_segments);
            }
            return ParseLegacyManifest(content, out_segments);
        };

        auto st = parse_one(p);
        if (st.ok()) {
            return st;
        }

        if (fs::exists(prev, ec) && !ec) {
            st = parse_one(prev);
            if (st.ok()) return st;
        }

        return st;
    }

    pomai::Status ShardManifest::Commit(const std::string &shard_dir, const std::vector<std::string> &segments)
    {
        fs::path tmp = fs::path(shard_dir) / "manifest.new";
        fs::path curr = fs::path(shard_dir) / "manifest.current";
        fs::path prev = fs::path(shard_dir) / "manifest.prev";

        const std::string payload = BuildPayload(segments);
        const std::uint32_t crc = pomai::util::Crc32c(payload.data(), payload.size());
        std::string content;
        content.reserve(payload.size() + 64);
        content += std::string(kManifestHeader) + "\n";
        content += "crc32c " + std::to_string(crc) + "\n";
        content += payload;
        
        pomai::util::PosixFile pf;
        auto st = pomai::util::PosixFile::CreateTrunc(tmp.string(), &pf);
        if (!st.ok()) return st;
        st = pf.PWrite(0, content.data(), content.size());
        if (!st.ok()) return st;
        st = pf.SyncData();
        if (!st.ok()) return st;
        st = pf.Close();
        if (!st.ok()) return st;

        // 2. Keep best-effort backup of last committed manifest
        std::error_code ec;
        if (fs::exists(curr, ec) && !ec) {
            fs::remove(prev, ec);
            ec.clear();
            fs::rename(curr, prev, ec);
            if (ec) return pomai::Status::IOError("backup rename failed: " + ec.message());
        }

        // 3. Publish new manifest
        fs::rename(tmp, curr, ec);
        if (ec) return pomai::Status::IOError("rename failed: " + ec.message());

        // 4. Sync Dir
        return pomai::util::FsyncDir(shard_dir);
    }

} // namespace pomai::core
