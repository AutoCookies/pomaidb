/*
 * src/core/pomai_db.cc
 *
 * Implementation for PomaiDB and Membrance persistence.
 */

#include "src/core/pomai_db.h"
#include "src/core/metadata_index.h"
#include "src/core/split_manager.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <system_error>
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>

namespace pomai::core
{

    static inline std::string trim_str(const std::string &s)
    {
        const char *ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos)
            return "";
        size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }

    // [FIXED] Constructor signature matches header (4 args)
    Membrance::Membrance(const std::string &nm, const MembranceConfig &cfg,
                         const std::string &root_path, const pomai::config::PomaiConfig &global_cfg)
        : name(nm), dim(cfg.dim), ram_mb(cfg.ram_mb)
    {
        // Build data path: data_root/name
        try
        {
            std::filesystem::path base(root_path);
            std::filesystem::path p = base / name;
            
            if (!std::filesystem::exists(p))
            {
                std::error_code ec;
                std::filesystem::create_directories(p, ec);
            }
            data_path = p.string();
        }
        catch (...)
        {
            data_path = "./data/" + name;
        }

        // 1. Create ShardArena
        arena = std::make_unique<pomai::memory::ShardArena>(0, ram_mb * 1024 * 1024, global_cfg);

        // 2. Create Orbit
        pomai::ai::orbit::PomaiOrbit::Config ocfg;
        ocfg.dim = dim;
        ocfg.data_path = data_path;
        ocfg.algo = global_cfg.orbit;
        ocfg.cortex_cfg = global_cfg.network;
        
        // Use default EEQ config
        ocfg.eeq_cfg = pomai::config::EternalEchoConfig();

        orbit = std::make_unique<pomai::ai::orbit::PomaiOrbit>(ocfg, arena.get());

        // 3. Initialize Hot Tier
        hot_tier = std::make_unique<HotTier>(dim, global_cfg.hot_tier);

        // 4. Initialize Metadata Index
        try
        {
            if (cfg.enable_metadata_index)
            {
                std::string meta_path = data_path + "/metadata";
                if (!std::filesystem::exists(meta_path))
                    std::filesystem::create_directories(meta_path);

                // [FIXED] Use make_shared instead of make_unique
                meta_index = std::make_shared<MetadataIndex>(global_cfg.metadata);

                // Link to Orbit
                orbit->set_metadata_index(meta_index);
            }
        }
        catch (const std::exception &e)
        {
            std::clog << "[PomaiDB] MetadataIndex init failed: " << e.what() << "\n";
        }

        // 5. Initialize Split Manager
        try
        {
            split_mgr = std::make_unique<SplitManager>(global_cfg);
            split_mgr->load(data_path);
        }
        catch (const std::exception &e)
        {
            std::clog << "[PomaiDB] SplitManager init failed: " << e.what() << "\n";
        }
    }

    PomaiDB::PomaiDB(const pomai::config::PomaiConfig &config)
        : config_(config), bg_running_(false)
    {
        std::filesystem::path root(config_.res.data_root);
        try
        {
            if (!std::filesystem::exists(root))
                std::filesystem::create_directories(root);
        }
        catch (...) {}

        std::string manifest_path = (root / "membrances.manifest").string();
        std::string wal_path = (root / "wal.log").string();

        if (!wal_.open(wal_path, true, config_.wal))
        {
            std::clog << "[PomaiDB] Warning: failed to open WAL\n";
        }

        if (!load_manifest())
        {
            std::clog << "[PomaiDB] Clean start.\n";
        }

        auto apply_cb = [this](uint16_t type, const void *payload, uint32_t len, uint64_t /*seq*/) -> bool
        {
            try
            {
                if (type == pomai::memory::WAL_REC_CREATE_MEMBRANCE)
                {
                    if (!payload || len == 0) return true;
                    std::string s(reinterpret_cast<const char *>(payload), len);
                    std::istringstream iss(s);
                    std::string t;
                    std::vector<std::string> parts;
                    while (std::getline(iss, t, '|')) parts.push_back(t);

                    if (parts.size() >= 2)
                    {
                        std::string name = trim_str(parts[0]);
                        size_t dim = 0, ram_mb = 256;
                        try { dim = static_cast<size_t>(std::stoul(parts[1])); } catch (...) {}
                        if (parts.size() >= 3) { try { ram_mb = static_cast<size_t>(std::stoul(parts[2])); } catch (...) {} }

                        if (!name.empty() && dim != 0)
                        {
                            MembranceConfig c;
                            c.dim = dim;
                            c.ram_mb = ram_mb;
                            std::unique_lock<std::shared_mutex> lk(mu_);
                            if (membrances_.count(name) == 0)
                                create_membrance_internal(name, c);
                        }
                    }
                }
                else if (type == pomai::memory::WAL_REC_DROP_MEMBRANCE)
                {
                    if (!payload || len == 0) return true;
                    std::string name(reinterpret_cast<const char *>(payload), len);
                    name = trim_str(name);
                    if (!name.empty())
                    {
                        std::unique_lock<std::shared_mutex> lk(mu_);
                        auto it = membrances_.find(name);
                        if (it != membrances_.end())
                        {
                            membrances_.erase(it);
                            try { std::filesystem::remove_all(std::filesystem::path(config_.res.data_root) / name); } catch (...) {}
                        }
                    }
                }
            }
            catch (...) {}
            return true;
        };

        wal_.replay(apply_cb);

        bg_running_ = true;
        bg_thread_ = std::thread(&PomaiDB::background_worker, this);
    }

    PomaiDB::~PomaiDB()
    {
        bg_running_ = false;
        if (bg_thread_.joinable())
            bg_thread_.join();

        save_all_membrances();
        save_manifest();
        wal_.close();
    }

    bool PomaiDB::create_membrance_internal(const std::string &name, const MembranceConfig &cfg)
    {
        if (membrances_.count(name)) return false;
        if (cfg.dim == 0) return false;

        try
        {
            // [FIXED] Pass global config_ correctly here
            auto m = std::make_unique<Membrance>(name, cfg, config_.res.data_root, config_);

            try { m->orbit->save_schema(); } catch (...) {}

            membrances_.emplace(name, std::move(m));
            return true;
        }
        catch (const std::exception &e)
        {
            std::clog << "[PomaiDB] Error creating membrance " << name << ": " << e.what() << "\n";
            return false;
        }
    }

    bool PomaiDB::create_membrance(const std::string &name, const MembranceConfig &cfg)
    {
        MembranceConfig final_cfg = cfg;
        if (final_cfg.engine.empty()) final_cfg.engine = config_.db.engine_type;
        if (final_cfg.ram_mb == 0) final_cfg.ram_mb = config_.db.default_membrance_ram_mb;

        try
        {
            std::string payload = name + "|" + std::to_string(final_cfg.dim) + "|" + std::to_string(final_cfg.ram_mb);
            wal_.append_record(static_cast<uint16_t>(pomai::memory::WAL_REC_CREATE_MEMBRANCE),
                               payload.data(), static_cast<uint32_t>(payload.size()));
        }
        catch (...) { return false; }

        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            if (!create_membrance_internal(name, final_cfg)) return false;
        }

        save_manifest();
        wal_.truncate_to_zero();
        return true;
    }

    bool PomaiDB::drop_membrance(const std::string &name)
    {
        try
        {
            wal_.append_record(static_cast<uint16_t>(pomai::memory::WAL_REC_DROP_MEMBRANCE),
                               name.data(), static_cast<uint32_t>(name.size()));
        }
        catch (...) { return false; }

        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            auto it = membrances_.find(name);
            if (it == membrances_.end()) return false;

            try { std::filesystem::remove_all(std::filesystem::path(config_.res.data_root) / name); } catch (...) {}
            membrances_.erase(it);
        }

        save_manifest();
        wal_.truncate_to_zero();
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
        std::shared_lock<std::shared_mutex> lock(mu_);
        std::vector<std::string> names;
        for (const auto &kv : membrances_)
            names.push_back(kv.first);
        return names;
    }

    bool PomaiDB::insert(const std::string &membr, const float *vec, uint64_t label)
    {
        Membrance *m = get_membrance(membr);
        if (!m) return false;
        if (m->hot_tier) { m->hot_tier->push(label, vec); return true; }
        return m->orbit->insert(vec, label);
    }

    bool PomaiDB::insert_batch(const std::string &membr, const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        Membrance *m = get_membrance(membr);
        if (!m) return false;
        return m->orbit->insert_batch(batch);
    }

    std::vector<std::pair<uint64_t, float>> PomaiDB::search(const std::string &membr, const float *query, size_t k)
    {
        Membrance *m = get_membrance(membr);
        if (!m) return {};

        auto hot = m->hot_tier ? m->hot_tier->search(query, k) : std::vector<std::pair<uint64_t, float>>{};
        auto warm = m->orbit->search(query, k);

        if (hot.empty()) return warm;
        if (warm.empty()) return hot;

        std::vector<std::pair<uint64_t, float>> merged;
        merged.reserve(k);
        size_t i = 0, j = 0;
        while (merged.size() < k && (i < hot.size() || j < warm.size()))
        {
            if (i < hot.size() && (j >= warm.size() || hot[i].second < warm[j].second))
                merged.push_back(hot[i++]);
            else
                merged.push_back(warm[j++]);
        }
        return merged;
    }

    bool PomaiDB::get(const std::string &membr, uint64_t label, std::vector<float> &out)
    {
        Membrance *m = get_membrance(membr);
        if (!m) return false;
        return m->orbit->get(label, out);
    }

    bool PomaiDB::remove(const std::string &membr, uint64_t label)
    {
        Membrance *m = get_membrance(membr);
        if (!m) return false;
        return m->orbit->remove(label);
    }

    size_t PomaiDB::get_membrance_dim(const std::string &name) const
    {
        const Membrance *m = get_membrance(name);
        return m ? m->dim : 0;
    }

    bool PomaiDB::save_manifest()
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        std::string path = config_.res.data_root + "/" + config_.db.manifest_file;
        std::string tmp = path + ".tmp";

        std::ofstream ofs(tmp, std::ios::trunc);
        if (!ofs.is_open()) return false;

        ofs << "# PomaiDB Manifest\n";
        for (const auto &kv : membrances_)
        {
            const auto *m = kv.second.get();
            ofs << m->name << '|' << m->dim << '|' << m->ram_mb << '\n';
        }
        ofs.close();
        std::filesystem::rename(tmp, path);
        return true;
    }

    bool PomaiDB::load_manifest()
    {
        std::unique_lock<std::shared_mutex> lock(mu_);
        std::string path = config_.res.data_root + "/" + config_.db.manifest_file;
        std::ifstream ifs(path);
        if (!ifs.is_open()) return false;

        std::string line;
        size_t loaded = 0;
        while (std::getline(ifs, line))
        {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string t;
            std::vector<std::string> parts;
            while (std::getline(iss, t, '|')) parts.push_back(trim_str(t));

            if (parts.size() >= 2)
            {
                std::string name = parts[0];
                size_t dim = 0, ram = 256;
                try { dim = std::stoul(parts[1]); } catch (...) {}
                if (parts.size() >= 3) try { ram = std::stoul(parts[2]); } catch (...) {}

                if (!name.empty() && dim > 0)
                {
                    MembranceConfig c;
                    c.dim = dim;
                    c.ram_mb = ram;
                    if (create_membrance_internal(name, c)) loaded++;
                }
            }
        }
        return loaded > 0;
    }

    bool PomaiDB::save_all_membrances()
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        bool ok = true;
        for (const auto &kv : membrances_)
        {
            try
            {
                auto* m = kv.second.get();
                m->orbit->save_schema();
                
                if (m->split_mgr && m->split_mgr->has_split()) {
                    m->split_mgr->save(m->data_path);
                }
            }
            catch (...) { ok = false; }
        }
        return ok;
    }

    void PomaiDB::background_worker()
    {
        while (bg_running_)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.db.bg_worker_interval_ms));
            std::shared_lock<std::shared_mutex> lock(mu_);

            for (auto &kv : membrances_)
            {
                Membrance *m = kv.second.get();
                if (!m || !m->hot_tier) continue;

                auto batch = m->hot_tier->swap_and_flush();
                if (batch.empty()) continue;

                std::vector<std::pair<uint64_t, std::vector<float>>> vec_batch;
                vec_batch.reserve(batch.count());
                const size_t dim = batch.dim;

                for (size_t i = 0; i < batch.labels.size(); ++i)
                {
                    std::vector<float> v(dim);
                    std::memcpy(v.data(), batch.data.data() + i * dim, dim * sizeof(float));
                    vec_batch.emplace_back(batch.labels[i], std::move(v));
                }
                m->orbit->insert_batch(vec_batch);
            }
        }
    }

} // namespace pomai::core