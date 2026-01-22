#include "src/core/pomai_db.h"
#include "src/core/metadata_index.h"
#include "src/core/split_manager.h"
#include "src/core/types.h"
#include "src/core/cpu_kernels.h"
#include "src/facade/data_supplier.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <system_error>
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>
#include <functional>
#include <cctype>

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

    static inline std::string upcase(const std::string &s)
    {
        std::string r = s;
        for (char &c : r)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return r;
    }

    Membrance::Membrance(const std::string &nm, const MembranceConfig &cfg,
                         const std::string &root_path, const pomai::config::PomaiConfig &global_cfg)
        : name(nm), dim(cfg.dim), ram_mb(cfg.ram_mb), data_path(), data_type(cfg.data_type)
    {
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

        arena = std::make_unique<pomai::memory::ShardArena>(0, ram_mb * 1024 * 1024, global_cfg);

        pomai::ai::orbit::PomaiOrbit::Config ocfg;
        ocfg.dim = dim;
        ocfg.data_path = data_path;
        ocfg.algo = global_cfg.orbit;
        ocfg.cortex_cfg = global_cfg.network;
        // Updated: PomaiOrbit::Config now uses ZeroHarmonyConfig (named zero_harmony_cfg).
        // Construct a default ZeroHarmonyConfig (alias EternalEchoConfig still available),
        // but prefer explicit ZeroHarmonyConfig to avoid ambiguity.
        ocfg.zero_harmony_cfg = pomai::config::ZeroHarmonyConfig();

        orbit = std::make_unique<pomai::ai::orbit::PomaiOrbit>(ocfg, arena.get());

        hot_tier = std::make_unique<HotTier>(dim, global_cfg.hot_tier, data_type);

        try
        {
            if (cfg.enable_metadata_index)
            {
                std::string meta_path = data_path + "/metadata";
                if (!std::filesystem::exists(meta_path))
                    std::filesystem::create_directories(meta_path);
                meta_index = std::make_shared<MetadataIndex>(global_cfg.metadata);
                orbit->set_metadata_index(meta_index);
            }
        }
        catch (...)
        {
        }

        try
        {
            split_mgr = std::make_unique<SplitManager>(global_cfg);
            split_mgr->load(data_path);
        }
        catch (...)
        {
        }
    }

    PomaiDB::PomaiDB(const pomai::config::PomaiConfig &config)
        : config_(config), bg_running_(false)
    {
        try
        {
            std::filesystem::path root(config_.res.data_root);
            if (!std::filesystem::exists(root))
                std::filesystem::create_directories(root);
        }
        catch (...)
        {
        }

        std::string manifest_path = (std::filesystem::path(config_.res.data_root) / config_.db.manifest_file).string();
        std::string wal_path = (std::filesystem::path(config_.res.data_root) / "wal.log").string();

        wal_.open(wal_path, true, config_.wal);

        load_manifest();

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
                        pomai::core::DataType dt = pomai::core::DataType::FLOAT32;
                        try
                        {
                            dim = static_cast<size_t>(std::stoul(parts[1]));
                        }
                        catch (...)
                        {
                        }
                        if (parts.size() >= 3)
                            try
                            {
                                ram_mb = static_cast<size_t>(std::stoul(parts[2]));
                            }
                            catch (...)
                            {
                            }
                        if (parts.size() >= 4)
                            try
                            {
                                dt = pomai::core::parse_dtype(trim_str(parts[3]));
                            }
                            catch (...)
                            {
                            }
                        if (!name.empty() && dim != 0)
                        {
                            MembranceConfig c;
                            c.dim = dim;
                            c.ram_mb = ram_mb;
                            c.data_type = dt;
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
        if (membrances_.count(name))
            return false;
        if (cfg.dim == 0)
            return false;
        try
        {
            auto m = std::make_unique<Membrance>(name, cfg, config_.res.data_root, config_);
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
        catch (...)
        {
            return false;
        }
    }

    bool PomaiDB::create_membrance(const std::string &name, const MembranceConfig &cfg)
    {
        MembranceConfig final_cfg = cfg;
        if (final_cfg.engine.empty())
            final_cfg.engine = config_.db.engine_type;
        if (final_cfg.ram_mb == 0)
            final_cfg.ram_mb = config_.db.default_membrance_ram_mb;

        std::string payload = name + "|" + std::to_string(final_cfg.dim) + "|" + std::to_string(final_cfg.ram_mb) + "|" + pomai::core::dtype_name(final_cfg.data_type);
        wal_.append_record(static_cast<uint16_t>(pomai::memory::WAL_REC_CREATE_MEMBRANCE), payload.data(), static_cast<uint32_t>(payload.size()));

        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            if (!create_membrance_internal(name, final_cfg))
                return false;
        }

        save_manifest();
        wal_.truncate_to_zero();
        return true;
    }

    bool PomaiDB::drop_membrance(const std::string &name)
    {
        wal_.append_record(static_cast<uint16_t>(pomai::memory::WAL_REC_DROP_MEMBRANCE), name.data(), static_cast<uint32_t>(name.size()));

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
        names.reserve(membrances_.size());
        for (const auto &kv : membrances_)
            names.push_back(kv.first);
        return names;
    }

    bool PomaiDB::insert(const std::string &membr, const float *vec, uint64_t label)
    {
        Membrance *m = get_membrance(membr);
        if (!m)
            return false;
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

        if (m->hot_tier)
        {
            for (const auto &item : batch)
            {
                m->hot_tier->push(item.first, item.second.data());
            }
            return true;
        }

        std::unique_lock<std::shared_mutex> lock(mu_);
        return m->orbit->insert_batch(batch);
    }

    std::vector<std::pair<uint64_t, float>> PomaiDB::search(const std::string &membr, const float *query, size_t k)
    {
        Membrance *m = get_membrance(membr);
        if (!m)
            return {};
        auto hot = m->hot_tier ? m->hot_tier->search(query, k) : std::vector<std::pair<uint64_t, float>>{};
        auto warm = m->orbit->search(query, k);
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

    bool PomaiDB::save_manifest_internal()
    {
        std::string path = config_.res.data_root + "/" + config_.db.manifest_file;
        std::string tmp = path + ".tmp";
        std::ofstream ofs(tmp, std::ios::trunc);
        if (!ofs.is_open())
            return false;
        ofs << "# PomaiDB Manifest\n";
        for (const auto &kv : membrances_)
        {
            const auto *m = kv.second.get();
            std::string dt = pomai::core::dtype_name(m->data_type);
            ofs << m->name << '|' << m->dim << '|' << m->ram_mb << '|' << dt << '\n';
        }
        ofs.close();
        try
        {
            std::filesystem::rename(tmp, path);
        }
        catch (...)
        {
            return false;
        }
        return true;
    }

    bool PomaiDB::save_manifest()
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        return save_manifest_internal();
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
                pomai::core::DataType dt = pomai::core::DataType::FLOAT32;
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
                if (parts.size() >= 4)
                    try
                    {
                        dt = pomai::core::parse_dtype(parts[3]);
                    }
                    catch (...)
                    {
                    }
                if (!name.empty() && dim > 0)
                {
                    MembranceConfig c;
                    c.dim = dim;
                    c.ram_mb = ram;
                    c.data_type = dt;
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
                auto *m = kv.second.get();
                m->orbit->save_schema();
                if (m->split_mgr && m->split_mgr->has_split())
                    m->split_mgr->save(m->data_path);
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
            std::this_thread::sleep_for(std::chrono::milliseconds(config_.db.bg_worker_interval_ms));
            std::shared_lock<std::shared_mutex> lock(mu_);
            for (auto &kv : membrances_)
            {
                Membrance *m = kv.second.get();
                if (!m || !m->hot_tier)
                    continue;
                auto batch = m->hot_tier->swap_and_flush();
                if (batch.empty())
                    continue;
                const size_t dim = batch.dim;
                const uint32_t elem_size = batch.element_size;
                const pomai::core::DataType bdt = batch.data_type;
                const size_t count = batch.count();
                const uint8_t *flat = batch.data.data();
                std::vector<float> tmp;
                tmp.resize(dim);
                for (size_t i = 0; i < count; ++i)
                {
                    const uint8_t *slot = flat + i * dim * elem_size;
                    switch (bdt)
                    {
                    case DataType::FLOAT32:
                        std::memcpy(tmp.data(), slot, dim * sizeof(float));
                        break;
                    case DataType::FLOAT64:
                    {
                        const double *dp = reinterpret_cast<const double *>(slot);
                        for (size_t d = 0; d < dim; ++d)
                            tmp[d] = static_cast<float>(dp[d]);
                        break;
                    }
                    case DataType::INT32:
                    {
                        const int32_t *ip = reinterpret_cast<const int32_t *>(slot);
                        for (size_t d = 0; d < dim; ++d)
                            tmp[d] = static_cast<float>(ip[d]);
                        break;
                    }
                    case DataType::INT8:
                    {
                        const int8_t *ip = reinterpret_cast<const int8_t *>(slot);
                        for (size_t d = 0; d < dim; ++d)
                            tmp[d] = static_cast<float>(ip[d]);
                        break;
                    }
                    case DataType::FLOAT16:
                    {
                        const uint16_t *hp = reinterpret_cast<const uint16_t *>(slot);
                        for (size_t d = 0; d < dim; ++d)
                            tmp[d] = fp16_to_fp32(hp[d]);
                        break;
                    }
                    default:
                    {
                        size_t copy_bytes = std::min<size_t>(dim * sizeof(float), dim * elem_size);
                        std::memcpy(tmp.data(), slot, copy_bytes);
                        if (copy_bytes < dim * sizeof(float))
                            for (size_t d = (copy_bytes / sizeof(float)); d < dim; ++d)
                                tmp[d] = 0.0f;
                        break;
                    }
                    }
                    try
                    {
                        m->orbit->insert(tmp.data(), batch.labels[i]);
                    }
                    catch (...)
                    {
                    }
                }
            }
        }
    }

    bool PomaiDB::iterate_batch(const std::string &membr,
                                const std::string &mode,
                                size_t off,
                                size_t lim,
                                size_t batch_size,
                                const std::function<void(const std::vector<uint64_t> &ids, const std::string &concat_buf, uint32_t per_vec_bytes)> &consumer)
    {
        if (!consumer)
            return false;
        Membrance *m = get_membrance(membr);
        if (!m)
            return false;
        const std::vector<uint64_t> *idxs_ptr = nullptr;
        std::vector<uint64_t> tmp_all;
        std::string up = upcase(mode);
        if (up == "TRAIN" && m->split_mgr)
            idxs_ptr = &m->split_mgr->train_indices;
        else if (up == "VAL" && m->split_mgr)
            idxs_ptr = &m->split_mgr->val_indices;
        else if (up == "TEST" && m->split_mgr)
            idxs_ptr = &m->split_mgr->test_indices;
        else
        {
            tmp_all = m->orbit->get_all_labels();
            idxs_ptr = &tmp_all;
        }
        if (!idxs_ptr || idxs_ptr->empty())
            return true;
        size_t start = off;
        size_t to_process = std::min(lim == 0 ? idxs_ptr->size() - start : lim, idxs_ptr->size() - start);
        size_t processed = 0;
        thread_local std::string tl_concat_buf;
        thread_local std::vector<uint64_t> tl_ids;
        uint32_t elem_size = static_cast<uint32_t>(pomai::core::dtype_size(m->data_type));
        size_t per_vec_bytes = elem_size * m->dim;
        while (processed < to_process)
        {
            size_t chunk = batch_size == 0 ? to_process - processed : std::min(batch_size, to_process - processed);
            tl_ids.clear();
            tl_ids.reserve(chunk);
            tl_concat_buf.clear();
            tl_concat_buf.reserve(chunk * per_vec_bytes);
            for (size_t i = 0; i < chunk; ++i)
            {
                uint64_t id = (*idxs_ptr)[start + processed + i];
                tl_ids.push_back(id);
                pomai::core::DataType got_dt;
                uint32_t got_elem_size = 0;
                std::string raw;
                bool ok = pomai::server::data_supplier::fetch_vector_raw(m, id, raw, m->dim, got_dt, got_elem_size);
                (void)ok;
                if (raw.size() < per_vec_bytes)
                {
                    if (!raw.empty())
                        tl_concat_buf.append(raw.data(), raw.size());
                    tl_concat_buf.append(per_vec_bytes - raw.size(), '\0');
                }
                else
                    tl_concat_buf.append(raw.data(), per_vec_bytes);
            }
            consumer(tl_ids, tl_concat_buf, static_cast<uint32_t>(per_vec_bytes));
            processed += chunk;
        }
        return true;
    }

    bool PomaiDB::fetch_batch_raw(const std::string &membr, const std::vector<uint64_t> &ids, std::vector<std::string> &outs)
    {
        outs.clear();
        if (ids.empty())
            return true;
        Membrance *m = get_membrance(membr);
        if (!m)
            return false;
        try
        {
            if (m->orbit)
            {
                if (m->orbit->get_vectors_raw(ids, outs))
                {
                    if (outs.size() != ids.size())
                        outs.resize(ids.size());
                    return true;
                }
            }
        }
        catch (...)
        {
        }
        outs.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i)
        {
            pomai::core::DataType got_dt;
            uint32_t got_elem = 0;
            try
            {
                std::string raw;
                bool ok = pomai::server::data_supplier::fetch_vector_raw(m, ids[i], raw, m->dim, got_dt, got_elem);
                if (ok || !raw.empty())
                    outs[i] = std::move(raw);
                else
                    outs[i].clear();
            }
            catch (...)
            {
                outs[i].clear();
            }
        }
        return true;
    }

    bool PomaiDB::checkpoint_all()
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        bool ok = true;
        for (const auto &kv : membrances_)
        {
            try
            {
                auto *m = kv.second.get();
                m->orbit->save_schema();
                if (m->split_mgr && m->split_mgr->has_split())
                    m->split_mgr->save(m->data_path);
            }
            catch (...)
            {
                ok = false;
            }
        }
        for (const auto &kv : membrances_)
        {
            try
            {
                auto *m = kv.second.get();
                if (m && m->arena)
                    m->arena->persist_range(0, m->arena->capacity(), true);
            }
            catch (...)
            {
                ok = false;
            }
        }
        if (!wal_.truncate_to_zero())
            ok = false;
        return ok;
    }

} // namespace pomai::core