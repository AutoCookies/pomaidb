#include "storage/manifest/manifest.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
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
                return pomai::Status::IOError("Manifest: failed to open file for reading: " + path);

            in.seekg(0, std::ios::end);
            std::streamoff n = in.tellg();
            in.seekg(0, std::ios::beg);

            std::string buf;
            buf.resize(static_cast<std::size_t>(n));
            if (n > 0)
                in.read(buf.data(), n);

            if (!in.good() && n > 0)
                return pomai::Status::IOError("Manifest: failed to read file: " + path);

            *out = std::move(buf);
            return pomai::Status::Ok();
        }

        static pomai::Status AtomicWriteFile(const std::string &final_path, std::string_view content)
        {
            const std::string tmp = final_path + ".tmp";
            {
                std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                if (!out.is_open())
                    return pomai::Status::IOError("Manifest: failed to open tmp file for writing: " + tmp);
                out.write(content.data(), static_cast<std::streamsize>(content.size()));
                if (!out.good())
                    return pomai::Status::IOError("Manifest: failed to write to tmp file: " + tmp);
            }

            std::error_code ec;
            fs::rename(tmp, final_path, ec);
            if (ec)
                return pomai::Status::IOError("Manifest: failed to rename " + tmp + " to " + final_path + ": " + ec.message());
            return pomai::Status::Ok();
        }

        struct RootEntry
        {
            std::string name;
            std::uint32_t shard_count = 0;
            std::uint32_t dim = 0;
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
            if (header != "pomai.manifest.v2")
                return pomai::Status::Corruption("Manifest: invalid header, expected 'pomai.manifest.v2', got: " + std::string(header));
            sv = (p == std::string_view::npos) ? std::string_view{} : sv.substr(p + 1);

            while (!sv.empty())
            {
                std::size_t eol = sv.find('\n');
                std::string_view line = (eol == std::string_view::npos) ? sv : sv.substr(0, eol);
                sv = (eol == std::string_view::npos) ? std::string_view{} : sv.substr(eol + 1);

                if (line.empty())
                    continue;

                auto toks = SplitWs(line);
                if (toks.size() != 6)
                    return pomai::Status::Corruption("Manifest: invalid line format, expected 6 tokens, got " + std::to_string(toks.size()));

                if (toks[0] != "membrane" || toks[2] != "shards" || toks[4] != "dim")
                    return pomai::Status::Corruption("Manifest: invalid line format, expected 'membrane {name} shards {N} dim {D}'");

                RootEntry e;
                e.name = std::string(toks[1]);
                if (!IsValidName(e.name))
                    return pomai::Status::Corruption("Manifest: invalid membrane name: " + e.name);

                st = ParseU32(toks[3], &e.shard_count);
                if (!st.ok())
                    return st;
                st = ParseU32(toks[5], &e.dim);
                if (!st.ok())
                    return st;

                out_entries->push_back(std::move(e));
            }

            std::sort(out_entries->begin(), out_entries->end(),
                      [](const RootEntry &a, const RootEntry &b)
                      { return a.name < b.name; });

            return pomai::Status::Ok();
        }

        static std::string SerializeRoot(const std::vector<RootEntry> &entries)
        {
            std::string out;
            out += "pomai.manifest.v2\n";
            for (const auto &e : entries)
            {
                out += "membrane " + e.name +
                       " shards " + std::to_string(e.shard_count) +
                       " dim " + std::to_string(e.dim) + "\n";
            }
            return out;
        }

        static pomai::Status WriteMembraneManifest(std::string_view root_path, const pomai::MembraneSpec &spec)
        {
            std::string content;
            content += "pomai.membrane.v1\n";
            content += "name " + spec.name + "\n";
            content += "shards " + std::to_string(spec.shard_count) + "\n";
            content += "dim " + std::to_string(spec.dim) + "\n";
            return AtomicWriteFile(MembraneManifestPath(root_path, spec.name), content);
        }

    } // namespace

    pomai::Status Manifest::EnsureInitialized(std::string_view root_path)
    {
        std::error_code ec;
        fs::create_directories(std::string(root_path), ec);
        if (ec)
            return pomai::Status::IOError("Manifest: failed to create root directory: " + std::string(root_path) + ": " + ec.message());

        fs::create_directories(fs::path(std::string(root_path)) / "membranes", ec);
        if (ec)
            return pomai::Status::IOError("Manifest: failed to create membranes directory: " + ec.message());

        const auto mp = RootManifestPath(root_path);
        if (fs::exists(mp, ec))
            return pomai::Status::Ok();

        return AtomicWriteFile(mp, "pomai.manifest.v2\n");
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
            return pomai::Status::IOError("Manifest: failed to create membrane directory for '" + spec.name + "': " + ec.message());

        st = WriteMembraneManifest(root_path, spec);
        if (!st.ok())
            return st;

        entries.push_back({spec.name, spec.shard_count, spec.dim});
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

        out->name = it->name;
        out->dim = it->dim;
        out->shard_count = it->shard_count;
        return pomai::Status::Ok();
    }

} // namespace pomai::storage
