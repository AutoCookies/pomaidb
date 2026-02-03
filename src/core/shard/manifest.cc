#include "core/shard/manifest.h"
#include <fstream>
#include <filesystem>
#include "util/posix_file.h" // For FsyncDir

namespace pomai::core
{
    namespace fs = std::filesystem;

    pomai::Status ShardManifest::Load(const std::string &shard_dir, std::vector<std::string> *out_segments)
    {
        fs::path p = fs::path(shard_dir) / "manifest.current";
        std::error_code ec;
        if (!fs::exists(p, ec))
        {
            if (ec) return pomai::Status::IOError(ec.message());
            out_segments->clear();
            return pomai::Status::Ok();
        }

        std::ifstream in(p);
        if (!in.is_open()) return pomai::Status::IOError("failed to open manifest.current");

        out_segments->clear();
        std::string line;
        while (std::getline(in, line))
        {
            if (!line.empty())
            {
                out_segments->push_back(line);
            }
        }
        return pomai::Status::Ok();
    }

    pomai::Status ShardManifest::Commit(const std::string &shard_dir, const std::vector<std::string> &segments)
    {
        fs::path tmp = fs::path(shard_dir) / "manifest.new";
        fs::path curr = fs::path(shard_dir) / "manifest.current";

        // 1. Write tmp
        {
            std::ofstream out(tmp, std::ios::trunc);
            if (!out.is_open()) return pomai::Status::IOError("failed to create manifest.new");
            
            for (const auto &s : segments)
            {
                out << s << "\n";
            }
            out.flush();
            if (out.fail()) return pomai::Status::IOError("write failed");
            // fsync?
            // std::ofstream doesn't have easy fsync. Use util? or close and reopen with PosixFile?
            // Or just rely on OS? "Crash safe" requirement implies fsync.
            // Let's us PosixFile for writing if we want to be correct.
        }
        
        // Re-open with PosixFile to Sync? Or just write whole thing with PosixFile.
        // Let's use PosixFile for writing.
        std::string content;
        for (const auto &s : segments)
        {
            content += s;
            content += "\n";
        }
        
        pomai::util::PosixFile pf;
        auto st = pomai::util::PosixFile::CreateTrunc(tmp.string(), &pf);
        if (!st.ok()) return st;
        st = pf.PWrite(0, content.data(), content.size());
        if (!st.ok()) return st;
        st = pf.SyncData();
        if (!st.ok()) return st;
        st = pf.Close();
        if (!st.ok()) return st;

        // 2. Rename
        std::error_code ec;
        fs::rename(tmp, curr, ec);
        if (ec) return pomai::Status::IOError("rename failed: " + ec.message());

        // 3. Sync Dir
        return pomai::util::FsyncDir(shard_dir);
    }

} // namespace pomai::core
