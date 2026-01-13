/*
 * src/core/pomai_db.cc
 *
 * Implementation for PomaiDB and Membrance persistence (manifest + per-membrance schema).
 *
 * No third-party deps. Uses std::filesystem and plain text manifest and WalManager.
 */

#include "src/core/pomai_db.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib> // getenv
#include <system_error>
#include <cstring>

namespace pomai::core
{

    // small trim helper (local)
    static inline std::string trim_str(const std::string &s)
    {
        const char *ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos)
            return "";
        size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }

    // ---------------------------- Membrance ------------------------------------

    Membrance::Membrance(const std::string &nm, const MembranceConfig &cfg, const std::string &data_root)
        : name(nm), dim(cfg.dim), ram_mb(cfg.ram_mb)
    {
        // Build data path: data_root/name
        try
        {
            std::filesystem::path base(data_root);
            std::filesystem::path p = base / name;
            data_path = p.string();
            std::error_code ec;
            std::filesystem::create_directories(p, ec);
            if (ec)
            {
                std::clog << "[PomaiDB] Warning: failed to create membrance directory " << data_path << ": " << ec.message() << "\n";
                // proceed — Orbit will use path anyway (may fail to write)
            }
        }
        catch (const std::exception &e)
        {
            std::clog << "[PomaiDB] Warning: exception creating membrance directory: " << e.what() << "\n";
            data_path = data_root + "/" + name;
        }

        // Create ShardArena and Orbit
        // For per-membrance arenas we use ShardArena; shard id 0 is fine here (single arena per membrance)
        arena = std::make_unique<pomai::memory::ShardArena>(0, ram_mb * 1024 * 1024);

        pomai::ai::orbit::PomaiOrbit::Config cfg_orbit;
        cfg_orbit.dim = dim;
        cfg_orbit.data_path = data_path;
        // Let first membrance optionally run cortex; keep default behavior.
        orbit = std::make_unique<pomai::ai::orbit::PomaiOrbit>(cfg_orbit, arena.get());

        // If there's an existing schema file under data_path, PomaiOrbit constructor tries to load it.
    }

    // ---------------------------- PomaiDB -------------------------------------

    static inline std::string get_env_or_default(const char *env, const char *def)
    {
        const char *v = std::getenv(env);
        return v ? std::string(v) : std::string(def);
    }

    PomaiDB::PomaiDB(const std::string &data_root)
    {
        // Determine data root: prefer constructor arg, else env var, else default
        if (!data_root.empty())
        {
            data_root_ = data_root;
        }
        else
        {
            data_root_ = get_env_or_default("POMAI_DB_DIR", "./data/pomai_db");
        }

        // Ensure directory exists
        std::error_code ec;
        std::filesystem::create_directories(data_root_, ec);
        if (ec)
        {
            std::clog << "[PomaiDB] Warning: could not create data root " << data_root_ << ": " << ec.message() << "\n";
        }

        manifest_path_ = (std::filesystem::path(data_root_) / "membrances.manifest").string();
        std::string wal_path = (std::filesystem::path(data_root_) / "wal.log").string();

        // Open WAL early
        pomai::memory::WalManager::WalConfig wcfg;
        wcfg.sync_on_append = true;
        if (!wal_.open(wal_path, true, wcfg))
        {
            std::clog << "[PomaiDB] Warning: failed to open WAL at " << wal_path << " (will continue without WAL)\n";
        }

        // Load manifest (if any)
        if (!load_manifest())
        {
            std::clog << "[PomaiDB] No existing manifest or load failed; starting empty DB\n";
        }

        // Replay WAL (apply create/drop records to bring DB to latest)
        auto apply_cb = [this](uint16_t type, const void *payload, uint32_t len, uint64_t /*seq*/) -> bool
        {
            try
            {
                if (type == pomai::memory::WAL_REC_CREATE_MEMBRANCE)
                {
                    if (!payload || len == 0)
                        return true;
                    std::string s(reinterpret_cast<const char *>(payload), len);
                    // parse name|dim|ram_mb
                    std::istringstream iss(s);
                    std::string t;
                    std::vector<std::string> parts;
                    while (std::getline(iss, t, '|'))
                        parts.push_back(t);
                    if (parts.size() >= 2)
                    {
                        std::string name = trim_str(parts[0]);
                        size_t dim = 0;
                        size_t ram_mb = 256;
                        try
                        {
                            dim = static_cast<size_t>(std::stoul(parts[1]));
                        }
                        catch (...)
                        {
                            dim = 0;
                        }
                        if (parts.size() >= 3)
                        {
                            try
                            {
                                ram_mb = static_cast<size_t>(std::stoul(parts[2]));
                            }
                            catch (...)
                            {
                                ram_mb = 256;
                            }
                        }
                        if (!name.empty() && dim != 0)
                        {
                            MembranceConfig cfg;
                            cfg.dim = dim;
                            cfg.ram_mb = ram_mb;
                            // Create internal (no WAL append)
                            std::unique_lock<std::shared_mutex> lk(mu_);
                            if (membrances_.count(name) == 0)
                            {
                                create_membrance_internal(name, cfg);
                            }
                        }
                    }
                }
                else if (type == pomai::memory::WAL_REC_DROP_MEMBRANCE)
                {
                    if (!payload || len == 0)
                        return true;
                    std::string name(reinterpret_cast<const char *>(payload), len);
                    name = trim_str(name);
                    if (!name.empty())
                    {
                        std::unique_lock<std::shared_mutex> lk(mu_);
                        auto it = membrances_.find(name);
                        if (it != membrances_.end())
                        {
                            membrances_.erase(it);
                            // Try removing data dir (best-effort)
                            try
                            {
                                std::filesystem::path p = std::filesystem::path(data_root_) / name;
                                std::error_code ec;
                                std::filesystem::remove_all(p, ec);
                            }
                            catch (...)
                            { /* ignore */
                            }
                        }
                    }
                }
                else
                {
                    // ignore other types (including checkpoint)
                }
            }
            catch (...)
            {
                // keep replaying but log if necessary
            }
            return true;
        };

        if (!wal_.replay(apply_cb))
        {
            std::clog << "[PomaiDB] Warning: WAL replay failed or truncated (continuing)\n";
        }
    }

    PomaiDB::~PomaiDB()
    {
        // Save ephemeral state: persist all membrance schemas and manifest
        bool ok = save_all_membrances();
        if (!ok)
        {
            std::clog << "[PomaiDB] Warning: failed to save some membrance schemas on shutdown\n";
        }
        if (!save_manifest())
        {
            std::clog << "[PomaiDB] Warning: failed to persist manifest on shutdown\n";
        }

        // Close WAL
        wal_.close();
    }

    bool PomaiDB::create_membrance_internal(const std::string &name, const MembranceConfig &cfg)
    {
        // Assumes mu_ locked by caller
        if (membrances_.count(name))
            return false;
        if (cfg.dim == 0)
            return false;

        try
        {
            auto m = std::make_unique<Membrance>(name, cfg, data_root_);
            // Let orbit save its schema immediately so on crash we have schema on disk
            try
            {
                m->orbit->save_schema();
            }
            catch (...)
            {
                std::clog << "[PomaiDB] Warning: failed to save schema for membrance " << name << " (non-fatal)\n";
            }
            membrances_.emplace(name, std::move(m));
            std::clog << "[PomaiDB] Membrance created (in-memory): " << name << " dim=" << cfg.dim << " ram=" << cfg.ram_mb << "MB\n";
            return true;
        }
        catch (const std::exception &e)
        {
            std::clog << "[PomaiDB] create_membrance_internal exception: " << e.what() << "\n";
            return false;
        }
        catch (...)
        {
            std::clog << "[PomaiDB] create_membrance_internal unknown exception\n";
            return false;
        }
    }

    bool PomaiDB::create_membrance(const std::string &name, const MembranceConfig &cfg)
    {
        // Write-ahead: append WAL first
        bool wal_ok = false;
        try
        {
            std::string payload = name + "|" + std::to_string(cfg.dim) + "|" + std::to_string(cfg.ram_mb);
            auto seq = wal_.append_record(static_cast<uint16_t>(pomai::memory::WAL_REC_CREATE_MEMBRANCE), payload.data(), static_cast<uint32_t>(payload.size()));
            wal_ok = seq.has_value();
            if (!wal_ok)
            {
                std::clog << "[PomaiDB] create_membrance: WAL append failed for " << name << "\n";
                return false;
            }
        }
        catch (...)
        {
            std::clog << "[PomaiDB] create_membrance: WAL append threw\n";
            return false;
        }

        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            if (!create_membrance_internal(name, cfg))
                return false;
        }

        // Persist manifest (atomic write). If successful, checkpoint (truncate WAL)
        if (!save_manifest())
        {
            std::clog << "[PomaiDB] Warning: failed to persist manifest after create_membrance\n";
        }
        else
        {
            if (!wal_.truncate_to_zero())
            {
                std::clog << "[PomaiDB] Warning: failed to truncate WAL after manifest checkpoint\n";
            }
        }
        return true;
    }

    bool PomaiDB::drop_membrance(const std::string &name)
    {
        // Write-ahead: append WAL drop record
        bool wal_ok = false;
        try
        {
            std::string payload = name;
            auto seq = wal_.append_record(static_cast<uint16_t>(pomai::memory::WAL_REC_DROP_MEMBRANCE), payload.data(), static_cast<uint32_t>(payload.size()));
            wal_ok = seq.has_value();
            if (!wal_ok)
            {
                std::clog << "[PomaiDB] drop_membrance: WAL append failed for " << name << "\n";
                return false;
            }
        }
        catch (...)
        {
            std::clog << "[PomaiDB] drop_membrance: WAL append threw\n";
            return false;
        }

        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            auto it = membrances_.find(name);
            if (it == membrances_.end())
                return false;

            // Attempt to remove on-disk directory for the membrance
            try
            {
                std::filesystem::path p = std::filesystem::path(data_root_) / name;
                std::error_code ec;
                std::filesystem::remove_all(p, ec);
                if (ec)
                {
                    std::clog << "[PomaiDB] Warning: failed to remove membrance data dir " << p.string() << ": " << ec.message() << "\n";
                }
            }
            catch (...)
            {
                std::clog << "[PomaiDB] Warning: exception while removing membrance directory\n";
            }

            membrances_.erase(it);
        }

        // Persist manifest (atomic write). If successful, checkpoint (truncate WAL)
        if (!save_manifest())
        {
            std::clog << "[PomaiDB] Warning: failed to persist manifest after drop_membrance\n";
        }
        else
        {
            if (!wal_.truncate_to_zero())
            {
                std::clog << "[PomaiDB] Warning: failed to truncate WAL after manifest checkpoint\n";
            }
        }
        std::clog << "[PomaiDB] Membrance dropped: " << name << "\n";
        return true;
    }

    Membrance *PomaiDB::get_membrance(const std::string &name)
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = membrances_.find(name);
        return (it != membrances_.end()) ? it->second.get() : nullptr;
    }

    const Membrance *PomaiDB::get_membrance(const std::string &name) const
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        auto it = membrances_.find(name);
        return (it != membrances_.end()) ? it->second.get() : nullptr;
    }

    std::vector<std::string> PomaiDB::list_membrances() const
    {
        std::vector<std::string> res;
        std::shared_lock<std::shared_mutex> lock(mu_);
        for (const auto &kv : membrances_)
            res.push_back(kv.first);
        return res;
    }

    // Vector API forwarding
    bool PomaiDB::insert(const std::string &membr, const float *vec, uint64_t label)
    {
        Membrance *m = get_membrance(membr);
        if (!m || !vec)
            return false;
        return m->orbit->insert(vec, label);
    }

    std::vector<std::pair<uint64_t, float>> PomaiDB::search(const std::string &membr, const float *query, size_t k)
    {
        Membrance *m = get_membrance(membr);
        if (!m || !query)
            return {};
        return m->orbit->search(query, k);
    }

    bool PomaiDB::get(const std::string &membr, uint64_t label, std::vector<float> &out)
    {
        Membrance *m = get_membrance(membr);
        if (!m)
            return false;
        return m->orbit->get(label, out);
    }

    bool PomaiDB::remove(const std::string &membr, uint64_t label)
    {
        Membrance *m = get_membrance(membr);
        if (!m)
            return false;
        return m->orbit->remove(label);
    }

    size_t PomaiDB::get_membrance_dim(const std::string &name) const
    {
        const Membrance *m = get_membrance(name);
        return m ? m->dim : 0;
    }

    // ---------------------------- Persistence ---------------------------------
    // Manifest format (text): one line per membrance:
    // name|dim|ram_mb
    // Lines starting with '#' ignored. File written atomically via temp rename.

    bool PomaiDB::save_manifest()
    {
        std::shared_lock<std::shared_mutex> lock(mu_); // snapshot under shared lock
        std::string tmp = manifest_path_ + ".tmp";

        std::ofstream ofs(tmp, std::ios::trunc);
        if (!ofs.is_open())
        {
            std::clog << "[PomaiDB] save_manifest: failed to open temp file " << tmp << "\n";
            return false;
        }

        ofs << "# PomaiDB membrances manifest\n";
        for (const auto &kv : membrances_)
        {
            const Membrance *m = kv.second.get();
            ofs << m->name << '|' << m->dim << '|' << m->ram_mb << '\n';
        }
        ofs.close();
        // Atomic rename
        std::error_code ec;
        std::filesystem::rename(tmp, manifest_path_, ec);
        if (ec)
        {
            std::clog << "[PomaiDB] save_manifest: rename failed: " << ec.message() << "\n";
            // attempt fallback: copy then remove
            try
            {
                std::filesystem::copy_file(tmp, manifest_path_, std::filesystem::copy_options::overwrite_existing);
                std::filesystem::remove(tmp);
            }
            catch (const std::exception &e)
            {
                std::clog << "[PomaiDB] save_manifest fallback: " << e.what() << "\n";
                return false;
            }
        }
        return true;
    }

    bool PomaiDB::load_manifest()
    {
        std::unique_lock<std::shared_mutex> lock(mu_); // loading modifies membrances_
        std::ifstream ifs(manifest_path_);
        if (!ifs.is_open())
        {
            return false; // no manifest => nothing to load
        }

        std::string line;
        size_t loaded = 0;
        while (std::getline(ifs, line))
        {
            line = trim_str(line);
            if (line.empty())
                continue;
            if (line[0] == '#')
                continue;
            // parse name|dim|ram_mb
            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> parts;
            while (std::getline(iss, token, '|'))
                parts.push_back(token);
            if (parts.size() < 2)
                continue;
            std::string name = trim_str(parts[0]);
            size_t dim = 0;
            size_t ram_mb = 256;
            try
            {
                dim = static_cast<size_t>(std::stoul(parts[1]));
            }
            catch (...)
            {
                dim = 0;
            }
            if (parts.size() >= 3)
            {
                try
                {
                    ram_mb = static_cast<size_t>(std::stoul(parts[2]));
                }
                catch (...)
                {
                    ram_mb = 256;
                }
            }
            if (name.empty() || dim == 0)
                continue;
            MembranceConfig cfg;
            cfg.dim = dim;
            cfg.ram_mb = ram_mb;
            if (create_membrance_internal(name, cfg))
                ++loaded;
        }
        return loaded > 0;
    }

    bool PomaiDB::save_all_membrances()
    {
        std::shared_lock<std::shared_mutex> lock(mu_); // snapshot, allow concurrent readers but block writers
        bool ok = true;
        for (const auto &kv : membrances_)
        {
            Membrance *m = kv.second.get();
            try
            {
                m->orbit->save_schema();
            }
            catch (const std::exception &e)
            {
                std::clog << "[PomaiDB] save_all_membrances: exception saving schema for " << m->name << ": " << e.what() << "\n";
                ok = false;
            }
            catch (...)
            {
                std::clog << "[PomaiDB] save_all_membrances: unknown exception saving schema for " << m->name << "\n";
                ok = false;
            }
        }
        return ok;
    }

} // namespace pomai::core