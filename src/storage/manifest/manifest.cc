#include "storage/manifest/manifest.h"
#include "util/crc32c.h"
#include "util/posix_file.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace pomai::storage
{
    namespace fs = std::filesystem;

    namespace
    {
        static bool IsValidName(std::string_view s)
        {
            if (s.empty() || s.size() > 64)
                return false;
            if (s == "." || s == "..")
                return false;
            for (unsigned char c : s)
            {
                const bool ok = (c >= 'a' && c <= 'z') ||
                                (c >= 'A' && c <= 'Z') ||
                                (c >= '0' && c <= '9') ||
                                c == '_' || c == '-' || c == '.';
                if (!ok)
                    return false;
            }
            return true;
        }

        static std::string RootManifestPath(std::string_view root_path)
        {
            return (fs::path(std::string(root_path)) / "MANIFEST").string();
        }

        static std::string MembraneDir(std::string_view root_path, std::string_view name)
        {
            return (fs::path(std::string(root_path)) / "membranes" / std::string(name)).string();
        }

        static std::string MembraneManifestPath(std::string_view root_path, std::string_view name)
        {
            return (fs::path(std::string(root_path)) / "membranes" / std::string(name) / "MANIFEST").string();
        }

        static pomai::Status ReadAll(const std::string &path, std::string *out)
        {
            std::ifstream in(path, std::ios::binary);
            if (!in.is_open())
                return pomai::Status::IOError("read failed: open");

            in.seekg(0, std::ios::end);
            std::streamoff n = in.tellg();
            in.seekg(0, std::ios::beg);

            std::string buf;
            if (n > 0)
            {
                buf.resize(static_cast<std::size_t>(n));
                in.read(buf.data(), n);
            }

            if (!in.good())
                return pomai::Status::IOError("read failed");

            // CRC validation
            if (n < 4)
                return pomai::Status::Corruption("file too short for CRC");

            uint32_t stored_crc;
            const size_t content_len = static_cast<size_t>(n) - 4;
            // stored CRC is last 4 bytes (little endian ideally, but we assume same endianness for now)
            // Just copying for simplicity
            unsigned char crc_buf[4];
            for(int i=0; i<4; ++i) crc_buf[i] = buf[content_len + i];
            
            stored_crc = (uint32_t)crc_buf[0] | 
                         ((uint32_t)crc_buf[1] << 8) | 
                         ((uint32_t)crc_buf[2] << 16) | 
                         ((uint32_t)crc_buf[3] << 24);

            uint32_t computed = pomai::util::Crc32c(buf.data(), content_len);
            if (computed != stored_crc)
                return pomai::Status::Corruption("CRC mismatch");

            *out = buf.substr(0, content_len);
            return pomai::Status::Ok();
        }

        static pomai::Status AtomicWriteFile(const std::string &final_path, std::string_view content)
        {
            const std::string tmp = final_path + ".tmp";
            
            // Use PosixFile for explicit sync control
            pomai::util::PosixFile pf;
            auto st = pomai::util::PosixFile::CreateTrunc(tmp, &pf);
            if (!st.ok()) return st;

            // Write content
            st = pf.PWrite(0, content.data(), content.size());
            if (!st.ok()) return st;
            
            // Calculate and write CRC
            uint32_t crc = pomai::util::Crc32c(content.data(), content.size());
            char crc_buf[4];
            crc_buf[0] = static_cast<char>(crc & 0xFF);
            crc_buf[1] = static_cast<char>((crc >> 8) & 0xFF);
            crc_buf[2] = static_cast<char>((crc >> 16) & 0xFF);
            crc_buf[3] = static_cast<char>((crc >> 24) & 0xFF);

            st = pf.PWrite(content.size(), crc_buf, 4);
            if (!st.ok()) return st;

            // Critical: Fsync data to disk before rename
            st = pf.SyncData();
            if (!st.ok()) return st;

            st = pf.Close();
            if (!st.ok()) return st;

            std::error_code ec;
            fs::rename(tmp, final_path, ec);
            if (ec)
                return pomai::Status::IOError("rename failed");
            
            // Directory fsync for rename durability?
            // The caller (Manifest logic) should probably handle dir fsync or we do it here.
            // ShardManifest::Commit does dir fsync. Global manifest should too.
            // Getting parent dir:
            fs::path p(final_path);
            return pomai::util::FsyncDir(p.parent_path().string());
        }

        struct RootEntry
        {
            std::string name;
        };

        static pomai::Status ParseU32(std::string_view tok, std::uint32_t *out)
        {
            if (!out)
                return pomai::Status::InvalidArgument("out=null");
            if (tok.empty())
                return pomai::Status::InvalidArgument("empty number");

            std::uint64_t v = 0;
            for (char ch : tok)
            {
                if (ch < '0' || ch > '9')
                    return pomai::Status::InvalidArgument("invalid number");
                v = v * 10 + static_cast<std::uint64_t>(ch - '0');
                if (v > 0xFFFFFFFFull)
                    return pomai::Status::InvalidArgument("number too large");
            }
            *out = static_cast<std::uint32_t>(v);
            return pomai::Status::Ok();
        }

        static std::vector<std::string_view> SplitWs(std::string_view line)
        {
            std::vector<std::string_view> out;
            std::size_t i = 0;
            while (i < line.size())
            {
                while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r'))
                    ++i;
                if (i >= line.size())
                    break;
                std::size_t j = i;
                while (j < line.size() && line[j] != ' ' && line[j] != '\t' && line[j] != '\r')
                    ++j;
                out.push_back(line.substr(i, j - i));
                i = j;
            }
            return out;
        }

        static pomai::Status LoadRoot(std::string_view root_path, std::vector<RootEntry> *out_entries)
        {
            out_entries->clear();

            std::string content;
            auto st = ReadAll(RootManifestPath(root_path), &content);
            if (!st.ok())
                return st;

            std::string_view sv(content);

            std::size_t p = sv.find('\n');
            std::string_view header = (p == std::string_view::npos) ? sv : sv.substr(0, p);
            
            // Checking for v3
            if (header != "pomai.manifest.v3")
                return pomai::Status::Corruption("bad manifest header: expected v3");
            
            sv = (p == std::string_view::npos) ? std::string_view{} : sv.substr(p + 1);

            while (!sv.empty())
            {
                std::size_t eol = sv.find('\n');
                std::string_view line = (eol == std::string_view::npos) ? sv : sv.substr(0, eol);
                sv = (eol == std::string_view::npos) ? std::string_view{} : sv.substr(eol + 1);

                if (line.empty())
                    continue;

                auto toks = SplitWs(line);
                if (toks.empty()) continue;

                if (toks[0] == "version") {
                     // Global version, ignore for now or store it if we had a struct for it
                     continue;
                }

                if (toks[0] == "membrane") {
                    if (toks.size() < 2) return pomai::Status::Corruption("bad membrane line");
                    RootEntry e;
                    e.name = std::string(toks[1]);
                    if (!IsValidName(e.name))
                        return pomai::Status::Corruption("invalid membrane name");
                    out_entries->push_back(std::move(e));
                }
            }

            std::sort(out_entries->begin(), out_entries->end(),
                      [](const RootEntry &a, const RootEntry &b)
                      { return a.name < b.name; });

            return pomai::Status::Ok();
        }

        static std::string SerializeRoot(const std::vector<RootEntry> &entries)
        {
            std::string out;
            out += "pomai.manifest.v3\n";
            out += "version 1\n"; // Hardcoded global version for now
            for (const auto &e : entries)
            {
                out += "membrane " + e.name + "\n";
            }
            return out;
        }

        static std::string MembraneKindToString(pomai::MembraneKind kind)
        {
            return kind == pomai::MembraneKind::kRag ? "RAG" : "VECTOR";
        }

        static pomai::MembraneKind ParseMembraneKind(std::string_view tok)
        {
            if (tok == "RAG") return pomai::MembraneKind::kRag;
            return pomai::MembraneKind::kVector;
        }

        static pomai::Status WriteMembraneManifest(std::string_view root_path, const pomai::MembraneSpec &spec)
        {
            std::string out;
            out += "pomai.membrane.v3\n";
            out += "name " + spec.name + "\n";
            out += "shards " + std::to_string(spec.shard_count) + "\n";
            out += "dim " + std::to_string(spec.dim) + "\n";
            out += "kind " + MembraneKindToString(spec.kind) + "\n";
            
            std::string mtype = "L2";
            if (spec.metric == pomai::MetricType::kInnerProduct) mtype = "IP";
            else if (spec.metric == pomai::MetricType::kCosine) mtype = "COS";
            out += "metric " + mtype + "\n";

            out += "index_params " + std::to_string(spec.index_params.nlist) + " " + 
                   std::to_string(spec.index_params.nprobe) + "\n";

            return AtomicWriteFile(MembraneManifestPath(root_path, spec.name), out);
        }

        static pomai::Status LoadMembraneManifest(std::string_view root_path, std::string_view name, pomai::MembraneSpec *spec) {
            std::string content;
            auto st = ReadAll(MembraneManifestPath(root_path, name), &content);
            if (!st.ok()) return st;

            std::string_view sv(content);
            std::size_t p = sv.find('\n');
            std::string_view header = (p == std::string_view::npos) ? sv : sv.substr(0, p);

            const bool v2 = header == "pomai.membrane.v2";
            const bool v3 = header == "pomai.membrane.v3";
            if (!v2 && !v3)
                 return pomai::Status::Corruption("bad membrane manifest header: expected v2/v3");

            sv = (p == std::string_view::npos) ? std::string_view{} : sv.substr(p + 1);
            
            spec->name = std::string(name);
            // defaults
            spec->shard_count = 0;
            spec->dim = 0;
            spec->metric = pomai::MetricType::kL2;
            spec->kind = pomai::MembraneKind::kVector;

            while (!sv.empty()) {
                std::size_t eol = sv.find('\n');
                std::string_view line = (eol == std::string_view::npos) ? sv : sv.substr(0, eol);
                sv = (eol == std::string_view::npos) ? std::string_view{} : sv.substr(eol + 1);

                if (line.empty()) continue;
                auto toks = SplitWs(line);
                if (toks.empty()) continue;

                if (toks[0] == "name") {
                    // verify name matches?
                } else if (toks[0] == "shards") {
                    if (toks.size() > 1) ParseU32(toks[1], &spec->shard_count);
                } else if (toks[0] == "dim") {
                    if (toks.size() > 1) ParseU32(toks[1], &spec->dim);
                } else if (toks[0] == "metric") {
                    if (toks.size() > 1) {
                        if (toks[1] == "IP") spec->metric = pomai::MetricType::kInnerProduct;
                        else if (toks[1] == "COS") spec->metric = pomai::MetricType::kCosine;
                        else spec->metric = pomai::MetricType::kL2;
                    }
                } else if (toks[0] == "kind") {
                    if (toks.size() > 1) {
                        spec->kind = ParseMembraneKind(toks[1]);
                    }
                } else if (toks[0] == "index_params") {
                    if (toks.size() > 2) {
                         ParseU32(toks[1], &spec->index_params.nlist);
                         ParseU32(toks[2], &spec->index_params.nprobe);
                    }
                }
            }
            return pomai::Status::Ok();
        }

    } // namespace

    pomai::Status Manifest::EnsureInitialized(std::string_view root_path)
    {
        std::error_code ec;
        fs::create_directories(std::string(root_path), ec);
        if (ec)
            return pomai::Status::IOError("create_directories root failed");

        fs::create_directories(fs::path(std::string(root_path)) / "membranes", ec);
        if (ec)
            return pomai::Status::IOError("create_directories membranes failed");

        const auto mp = RootManifestPath(root_path);
        if (fs::exists(mp, ec))
            return pomai::Status::Ok();

        // Write empty root v3
        return AtomicWriteFile(mp, "pomai.manifest.v3\nversion 1\n");
    }

    pomai::Status Manifest::CreateMembrane(std::string_view root_path, const pomai::MembraneSpec &spec)
    {
        if (!IsValidName(spec.name))
            return pomai::Status::InvalidArgument("invalid membrane name");
        if (spec.dim == 0)
            return pomai::Status::InvalidArgument("dim must be > 0");
        if (spec.shard_count == 0)
            return pomai::Status::InvalidArgument("shard_count must be > 0");

        auto st = EnsureInitialized(root_path);
        if (!st.ok())
            return st;

        std::vector<RootEntry> entries;
        st = LoadRoot(root_path, &entries);
        if (!st.ok())
            return st;

        auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const RootEntry &e)
                               { return e.name == spec.name; });
        if (it != entries.end())
            return pomai::Status::AlreadyExists("membrane already exists");

        std::error_code ec;
        fs::create_directories(MembraneDir(root_path, spec.name), ec);
        if (ec)
            return pomai::Status::IOError("create_directories membrane failed");

        st = WriteMembraneManifest(root_path, spec);
        if (!st.ok())
            return st;

        entries.push_back({spec.name});
        std::sort(entries.begin(), entries.end(),
                  [](const RootEntry &a, const RootEntry &b)
                  { return a.name < b.name; });

        return AtomicWriteFile(RootManifestPath(root_path), SerializeRoot(entries));
    }

    pomai::Status Manifest::DropMembrane(std::string_view root_path, std::string_view name)
    {
        if (!IsValidName(name))
            return pomai::Status::InvalidArgument("invalid membrane name");

        auto st = EnsureInitialized(root_path);
        if (!st.ok())
            return st;

        std::vector<RootEntry> entries;
        st = LoadRoot(root_path, &entries);
        if (!st.ok())
            return st;

        auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const RootEntry &e)
                               { return e.name == name; });
        if (it == entries.end())
            return pomai::Status::NotFound("membrane not found");

        entries.erase(it);
        st = AtomicWriteFile(RootManifestPath(root_path), SerializeRoot(entries));
        if (!st.ok())
            return st;

        std::error_code ec;
        fs::remove_all(MembraneDir(root_path, name), ec);
        return pomai::Status::Ok();
    }

    pomai::Status Manifest::ListMembranes(std::string_view root_path, std::vector<std::string> *out)
    {
        if (!out)
            return pomai::Status::InvalidArgument("out=null");

        auto st = EnsureInitialized(root_path);
        if (!st.ok())
            return st;

        std::vector<RootEntry> entries;
        st = LoadRoot(root_path, &entries);
        if (!st.ok())
            return st;

        out->clear();
        out->reserve(entries.size());
        for (const auto &e : entries)
            out->push_back(e.name);

        return pomai::Status::Ok();
    }

    pomai::Status Manifest::GetMembrane(std::string_view root_path, std::string_view name, pomai::MembraneSpec *out)
    {
        if (!out)
            return pomai::Status::InvalidArgument("out=null");
        if (!IsValidName(name))
            return pomai::Status::InvalidArgument("invalid membrane name");

        // First check if it exists in root
        std::vector<RootEntry> entries;
        auto st = LoadRoot(root_path, &entries);
        if (!st.ok()) return st;

        auto it = std::find_if(entries.begin(), entries.end(),
                               [&](const RootEntry &e)
                               { return e.name == name; });
        if (it == entries.end())
            return pomai::Status::NotFound("membrane not found in root");

        // Now load detailed spec from membrane specific manifest
        return LoadMembraneManifest(root_path, name, out);
    }

} // namespace pomai::storage
