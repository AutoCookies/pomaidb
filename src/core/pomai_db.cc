/*
 * src/core/pomai_db.cc
 *
 * Implementation for PomaiDB and Membrance persistence.
 *
 * Updates:
 * - Refactored to use centralized PomaiConfig injection.
 * - Membrance initialization now pulls algorithm/network settings from global config.
 * - [Fix] WAL replay now correctly handles config propagation.
 */

#include "src/core/pomai_db.h"
#include "src/core/metadata_index.h"

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

    // [CHANGED] Constructor now takes Global Config
    Membrance::Membrance(const std::string &nm, const MembranceConfig &cfg,
                         const std::string &data_root, const pomai::config::PomaiConfig &global_cfg)
        : name(nm), dim(cfg.dim), ram_mb(cfg.ram_mb)
    {
        // Build data path: data_root/name
        try
        {
            std::filesystem::path base(data_root);
            std::filesystem::path p = base / name;
            // Note: data_root is guaranteed to exist by PomaiDB ctor, but subfolder might not.
            if (!std::filesystem::exists(p))
            {
                std::error_code ec;
                std::filesystem::create_directories(p, ec);
                if (ec)
                    std::clog << "[PomaiDB] Warning: create_directories failed: " << ec.message() << "\n";
            }
            data_path = p.string();
        }
        catch (...)
        {
            data_path = "./data/" + name;
        }

        // 1. Create ShardArena (Memory Backend)
        // shard_id=0 because currently one arena per membrance
        arena = std::make_unique<pomai::memory::ShardArena>(0, ram_mb * 1024 * 1024);

        // 2. Create Orbit (Indexing Engine)
        // [FIXED] Construct Orbit Config from Global Config + Local Params
        pomai::ai::orbit::PomaiOrbit::Config ocfg;
        ocfg.dim = dim;
        ocfg.data_path = data_path;

        // Inject Algorithm Params (Centroids, Neighbors, BucketCap)
        ocfg.algo = global_cfg.orbit;

        // Inject Network Params (Cortex Port/P2P)
        // Note: Cortex usually shares port/config with main server or uses offset.
        // Here we pass the network config directly.
        ocfg.cortex_cfg = global_cfg.network;

        // Inject EEQ Params (Optional: if we add eeq to global config later, copy here)
        // ocfg.eeq_cfg = global_cfg.eeq; // (Future proofing)

        orbit = std::make_unique<pomai::ai::orbit::PomaiOrbit>(ocfg, arena.get());

        // 3. Initialize Hot Tier (Shock Absorber / Write Buffer)
        hot_tier = std::make_unique<HotTier>(dim, global_cfg.hot_tier);

        // 4. Initialize Metadata Index (Optional)
        try
        {
            if (cfg.enable_metadata_index)
            {
                // Vẫn tạo thư mục nếu cần cho tương lai, nhưng không truyền vào constructor
                std::string meta_path = data_path + "/metadata";
                if (!std::filesystem::exists(meta_path))
                    std::filesystem::create_directories(meta_path);

                // THAY ĐỔI TẠI ĐÂY: Truyền global_cfg.metadata thay vì meta_path
                meta_index = std::make_unique<MetadataIndex>(global_cfg.metadata);

                // Link to Orbit (View only)
                orbit->set_metadata_index(std::shared_ptr<MetadataIndex>(meta_index.get(), [](MetadataIndex *) {}));
            }
        }
        catch (const std::exception &e)
        {
            std::clog << "[PomaiDB] MetadataIndex init failed: " << e.what() << "\n";
        }
    }

    // ---------------------------- PomaiDB -------------------------------------

    // [CHANGED] Constructor takes Global Config
    PomaiDB::PomaiDB(const pomai::config::PomaiConfig &config)
        : config_(config), bg_running_(false)
    {
        // 1. Ensure Data Root Exists
        std::filesystem::path root(config_.res.data_root);
        try
        {
            if (!std::filesystem::exists(root))
                std::filesystem::create_directories(root);
        }
        catch (...)
        {
            std::clog << "[PomaiDB] Critical Warning: Could not create data root " << config_.res.data_root << "\n";
        }

        // 2. Setup Paths
        std::string manifest_path = (root / "membrances.manifest").string();
        std::string wal_path = (root / "wal.log").string();

        // 3. Open WAL
        pomai::memory::WalManager::WalConfig wcfg;
        wcfg.sync_on_append = true;
        if (!wal_.open(wal_path, true, wcfg))
        {
            std::clog << "[PomaiDB] Warning: failed to open WAL (running without persistence safety)\n";
        }

        // 4. Load Snapshot (Manifest)
        if (!load_manifest())
        {
            std::clog << "[PomaiDB] Clean start (no manifest found).\n";
        }

        // 5. Replay WAL (Crash Recovery)
        auto apply_cb = [this](uint16_t type, const void *payload, uint32_t len, uint64_t /*seq*/) -> bool
        {
            try
            {
                if (type == pomai::memory::WAL_REC_CREATE_MEMBRANCE)
                {
                    if (!payload || len == 0)
                        return true;
                    std::string s(reinterpret_cast<const char *>(payload), len);
                    std::istringstream iss(s);
                    std::string t;
                    std::vector<std::string> parts;
                    while (std::getline(iss, t, '|'))
                        parts.push_back(t);

                    if (parts.size() >= 2)
                    {
                        std::string name = trim_str(parts[0]);
                        size_t dim = 0, ram_mb = 256;
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
                            // Best-effort cleanup
                            try
                            {
                                std::filesystem::remove_all(std::filesystem::path(config_.res.data_root) / name);
                            }
                            catch (...)
                            {
                            }
                        }
                    }
                }
            }
            catch (...)
            {
            }
            return true;
        };

        wal_.replay(apply_cb);

        // 6. Start Background Worker (Hot->Warm Merger)
        bg_running_ = true;
        bg_thread_ = std::thread(&PomaiDB::background_worker, this);
    }

    PomaiDB::~PomaiDB()
    {
        // Stop worker
        bg_running_ = false;
        if (bg_thread_.joinable())
            bg_thread_.join();

        // Save State
        save_all_membrances();
        save_manifest();
        wal_.close();
    }

    bool PomaiDB::create_membrance_internal(const std::string &name, const MembranceConfig &cfg)
    {
        // Assumes mu_ is locked
        if (membrances_.count(name))
            return false;
        if (cfg.dim == 0)
            return false;

        try
        {
            // [FIXED] Pass global config_ to Membrance constructor
            auto m = std::make_unique<Membrance>(name, cfg, config_.res.data_root, config_);

            // Initial save to disk
            try
            {
                m->orbit->save_schema();
            }
            catch (...)
            {
            }

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
        // [QUAN TRỌNG] Tạo một bản sao config cục bộ để áp dụng Defaults
        MembranceConfig final_cfg = cfg;

        // 1. Áp dụng engine mặc định nếu người dùng không nhập
        if (final_cfg.engine.empty())
        {
            final_cfg.engine = config_.db.engine_type; // Sử dụng "orbit" từ config
        }

        // 2. Áp dụng RAM mặc định nếu giá trị truyền vào là 0
        if (final_cfg.ram_mb == 0)
        {
            final_cfg.ram_mb = config_.db.default_membrance_ram_mb; // Sử dụng 256 từ config
        }

        // --- Bắt đầu logic WAL với final_cfg đã được cập nhật ---
        try
        {
            // Ghi vào WAL các giá trị ĐÃ áp dụng default để khi replay không bị sai dim/ram
            std::string payload = name + "|" + std::to_string(final_cfg.dim) + "|" + std::to_string(final_cfg.ram_mb);
            wal_.append_record(static_cast<uint16_t>(pomai::memory::WAL_REC_CREATE_MEMBRANCE),
                               payload.data(), static_cast<uint32_t>(payload.size()));
        }
        catch (...)
        {
            return false;
        }

        // 2. Apply In-Memory (Truyền final_cfg thay vì cfg gốc)
        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            if (!create_membrance_internal(name, final_cfg))
                return false;
        }

        // 3. Snapshot Manifest & Truncate WAL
        save_manifest(); // Hàm này bên trong cũng phải dùng config_.db.manifest_file
        wal_.truncate_to_zero();
        return true;
    }

    bool PomaiDB::drop_membrance(const std::string &name)
    {
        // 1. WAL Log
        try
        {
            wal_.append_record(static_cast<uint16_t>(pomai::memory::WAL_REC_DROP_MEMBRANCE),
                               name.data(), static_cast<uint32_t>(name.size()));
        }
        catch (...)
        {
            return false;
        }

        // 2. Apply In-Memory
        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            auto it = membrances_.find(name);
            if (it == membrances_.end())
                return false;

            try
            {
                std::filesystem::remove_all(std::filesystem::path(config_.res.data_root) / name);
            }
            catch (...)
            {
            }
            membrances_.erase(it);
        }

        // 3. Checkpoint
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

    // --- Vector Operations Forwarding ---

    bool PomaiDB::insert(const std::string &membr, const float *vec, uint64_t label)
    {
        Membrance *m = get_membrance(membr);
        if (!m)
            return false;

        // Hot Tier First
        if (m->hot_tier)
        {
            m->hot_tier->push(label, vec);
            return true;
        }
        return m->orbit->insert(vec, label);
    }

    bool PomaiDB::insert_batch(const std::string &membr, const std::vector<std::pair<uint64_t, std::vector<float>>> &batch)
    {
        Membrance *m = get_membrance(membr);
        if (!m)
            return false;
        return m->orbit->insert_batch(batch);
    }

    std::vector<std::pair<uint64_t, float>> PomaiDB::search(const std::string &membr, const float *query, size_t k)
    {
        Membrance *m = get_membrance(membr);
        if (!m)
            return {};

        // 1. Hot Search
        auto hot = m->hot_tier ? m->hot_tier->search(query, k) : std::vector<std::pair<uint64_t, float>>{};

        // 2. Warm Search (Orbit)
        auto warm = m->orbit->search(query, k);

        // 3. Merge (Simplified merge-sort top-k)
        if (hot.empty())
            return warm;
        if (warm.empty())
            return hot;

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
        if (!m)
            return false;
        // Check Hot Tier? Not implemented yet for GET, assumed ephemeral.
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

    // --- Persistence IO ---

    bool PomaiDB::save_manifest()
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        std::string path = config_.res.data_root + "/" + config_.db.manifest_file;
        std::string tmp = path + ".tmp";

        std::ofstream ofs(tmp, std::ios::trunc);
        if (!ofs.is_open())
            return false;

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
        if (!ifs.is_open())
            return false;

        std::string line;
        size_t loaded = 0;
        while (std::getline(ifs, line))
        {
            if (line.empty() || line[0] == '#')
                continue;
            std::istringstream iss(line);
            std::string t;
            std::vector<std::string> parts;
            while (std::getline(iss, t, '|'))
                parts.push_back(trim_str(t));

            if (parts.size() >= 2)
            {
                std::string name = parts[0];
                size_t dim = 0, ram = 256;
                try
                {
                    dim = std::stoul(parts[1]);
                }
                catch (...)
                {
                }
                if (parts.size() >= 3)
                    try
                    {
                        ram = std::stoul(parts[2]);
                    }
                    catch (...)
                    {
                    }

                if (!name.empty() && dim > 0)
                {
                    MembranceConfig c;
                    c.dim = dim;
                    c.ram_mb = ram;
                    // Internal create bypassing WAL
                    if (create_membrance_internal(name, c))
                        loaded++;
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
                kv.second->orbit->save_schema();
            }
            catch (...)
            {
                ok = false;
            }
        }
        return ok;
    }

    void PomaiDB::background_worker()
    {
        while (bg_running_)
        {
            // Flush interval (shock absorber)
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.db.bg_worker_interval_ms));

            // Snapshot lock for reading membrance list
            std::shared_lock<std::shared_mutex> lock(mu_);

            for (auto &kv : membrances_)
            {
                Membrance *m = kv.second.get();
                if (!m || !m->hot_tier)
                    continue;

                // 1. Drain Hot Tier
                auto batch = m->hot_tier->swap_and_flush();
                if (batch.empty())
                    continue;

                // 2. Convert to Orbit Batch
                std::vector<std::pair<uint64_t, std::vector<float>>> vec_batch;
                vec_batch.reserve(batch.count());
                const size_t dim = batch.dim;

                for (size_t i = 0; i < batch.labels.size(); ++i)
                {
                    std::vector<float> v(dim);
                    std::memcpy(v.data(), batch.data.data() + i * dim, dim * sizeof(float));
                    vec_batch.emplace_back(batch.labels[i], std::move(v));
                }

                // 3. Insert to Warm (Orbit)
                m->orbit->insert_batch(vec_batch);
            }
        }
    }

} // namespace pomai::core