/*
 * src/facade/sql_executor.cc
 *
 * SQL/text protocol executor for Pomai server (refactored & optimized).
 *
 * Goals achieved:
 *  - Batch-first IO: use PomaiDB::fetch_batch_raw for bulk retrievals to minimize
 *    atomic ops / mmap lookups / filesystem reads and avoid per-id overhead.
 *  - Minimize allocations: pre-reserve output buffers using known sizes.
 *  - Clear, small helper functions to reduce duplication (fetch-with-fallback,
 *    append-and-pad).
 *  - Defensive input parsing and early error returns.
 *
 * Notes:
 *  - This file returns std::string responses (may contain binary payload).
 *    The caller (server) is responsible for sending the response to the client.
 *  - For extremely large responses consider switching to a streaming API (socket
 *    write from server side) to avoid keeping the whole payload in memory. The
 *    current design keeps compatibility with the existing server code path.
 *
 * Performance tips:
 *  - Ensure PomaiDB::fetch_batch_raw is implemented to group ids by bucket/arena
 *    and to reuse thread-local buffers. That is the most important lever for
 *    end-to-end throughput.
 */

#include "src/facade/sql_executor.h"

#include "src/facade/server_utils.h"
#include "src/facade/data_supplier.h"
#include "src/core/pomai_db.h"
#include "src/core/metadata_index.h"
#include "src/ai/whispergrain.h"
#include "src/core/metrics.h"
#include "src/core/types.h"

#include <sstream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <algorithm>
#include <chrono>

namespace pomai::server
{

    namespace utils = pomai::server::utils;
    namespace supplier = pomai::server::data_supplier;

    SqlExecutor::SqlExecutor() {}

    // Helper: attempt a single batch fetch via PomaiDB and fall back to per-id supplier
    static inline bool fetch_raws_with_fallback(
        pomai::core::PomaiDB *db,
        pomai::core::Membrance *m,
        const std::vector<uint64_t> &ids,
        size_t per_vec_bytes,
        std::vector<std::string> &out_raws)
    {
        out_raws.clear();
        if (ids.empty())
            return true;

        // Prefer DB-level batch fetch (fast path)
        try
        {
            if (db->fetch_batch_raw(m->name, ids, out_raws))
            {
                // Ensure returned elements are padded to per_vec_bytes
                for (auto &s : out_raws)
                {
                    if (s.size() < per_vec_bytes)
                        s.resize(per_vec_bytes, 0);
                    else if (s.size() > per_vec_bytes)
                        s.resize(per_vec_bytes);
                }
                return true;
            }
        }
        catch (...)
        {
            // fall through to per-id fallback
        }

        // Fallback: per-id supplier calls (slower, used rarely)
        out_raws.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i)
        {
            pomai::core::DataType dt;
            uint32_t elem_size = 0;
            try
            {
                supplier::fetch_vector_raw(m, ids[i], out_raws[i], m->dim, dt, elem_size);
            }
            catch (...)
            {
                out_raws[i].clear();
            }
            if (out_raws[i].size() < per_vec_bytes)
                out_raws[i].resize(per_vec_bytes, 0);
            else if (out_raws[i].size() > per_vec_bytes)
                out_raws[i].resize(per_vec_bytes);
        }
        return true;
    }

    // Append a vector of raw strings into out string (already padded to per_vec_bytes)
    static inline void append_raws_into(std::string &out, const std::vector<std::string> &raws, size_t per_vec_bytes)
    {
        // Reserve a bit to reduce reallocations if possible
        size_t needed = raws.size() * per_vec_bytes;
        if (needed > 0)
            out.reserve(out.size() + needed);

        for (const auto &r : raws)
        {
            if (r.size() >= per_vec_bytes)
            {
                out.append(r.data(), per_vec_bytes);
            }
            else
            {
                out.append(r.data(), r.size());
                out.append(per_vec_bytes - r.size(), '\0');
            }
        }
    }

    std::string SqlExecutor::execute(pomai::core::PomaiDB *db,
                                     pomai::ai::WhisperGrain &whisper,
                                     ClientState &state,
                                     const std::string &raw_cmd)
    {
        std::string cmd = utils::trim(raw_cmd);
        if (cmd.empty())
            return "ERR empty command\n";
        if (cmd.back() == ';')
            cmd.pop_back();

        std::string up = utils::to_upper(cmd);
        auto parts = utils::split_ws(cmd);

        // --- 1. SHOW MEMBRANCES ---
        if (up == "SHOW MEMBRANCES")
        {
            auto list = db->list_membrances();
            std::ostringstream ss;
            ss << "MEMBRANCES: " << list.size() << "\n";
            for (auto &m : list)
                ss << " - " << m << "\n";
            return ss.str();
        }

        // --- 2. USE <name> ---
        if (up.rfind("USE ", 0) == 0)
        {
            auto parts2 = utils::split_ws(cmd);
            if (parts2.size() >= 2)
            {
                std::string name = parts2[1];
                state.current_membrance = name;
                return std::string("OK: switched to membrance ") + name + "\n";
            }
            return "ERR: USE <name>;\n";
        }

        // --- 3. EXEC SPLIT ---
        if (up.rfind("EXEC SPLIT", 0) == 0)
        {
            auto parts2 = utils::split_ws(cmd);
            if (parts2.size() < 6)
                return "ERR: Usage: EXEC SPLIT <name> <tr%> <val%> <te%> [STRATIFIED <key> | CLUSTER]\n";

            std::string name = parts2[2];
            float tr = 0, val = 0, te = 0;
            try
            {
                tr = std::stof(parts2[3]);
                val = std::stof(parts2[4]);
                te = std::stof(parts2[5]);
            }
            catch (...)
            {
                return "ERR: Invalid split percentages\n";
            }

            if (tr + val + te > 1.001f)
                return "ERR: Ratios sum must be <= 1.0\n";

            auto *m = db->get_membrance(name);
            if (!m)
                return "ERR: Membrance not found\n";
            if (!m->split_mgr)
                return "ERR: Split manager not initialized\n";

            // detect strategy
            std::string strategy = "RANDOM";
            std::string strat_key;
            if (parts2.size() >= 7)
            {
                std::string t = utils::to_upper(parts2[6]);
                if (t == "STRATIFIED")
                {
                    strategy = "STRATIFIED";
                    if (parts2.size() >= 8)
                        strat_key = parts2[7];
                    else
                        return "ERR: STRATIFIED requires a key\n";
                }
                else if (t == "CLUSTER")
                    strategy = "CLUSTER";
            }

            size_t total_vectors = 0;
            try { total_vectors = m->orbit->get_info().num_vectors; }
            catch (...) { total_vectors = 0; }

            if (strategy == "STRATIFIED")
            {
                if (!m->meta_index) return "ERR: Metadata Index not enabled\n";
                auto groups = m->meta_index->get_groups(strat_key);
                if (groups.empty())
                    return std::string("ERR: No metadata for key '") + strat_key + "'\n";

                std::vector<uint64_t> items;
                std::vector<uint64_t> labels;
                items.reserve(1024);
                for (const auto &kv : groups)
                {
                    uint64_t label_hash = utils::hash_label(kv.first);
                    for (uint64_t id : kv.second)
                    {
                        items.push_back(id);
                        labels.push_back(label_hash);
                    }
                }
                m->split_mgr->execute_stratified_split(items, labels, tr, val, te);
                total_vectors = items.size();
            }
            else if (strategy == "CLUSTER")
            {
                if (!m->orbit) return "ERR: Orbit required for CLUSTER\n";
                size_t num_c = m->orbit->num_centroids();
                if (num_c == 0) return "ERR: No centroids (train first)\n";

                std::vector<uint32_t> cids(num_c);
                std::iota(cids.begin(), cids.end(), 0);
                std::mt19937 g(std::random_device{}());
                std::shuffle(cids.begin(), cids.end(), g);

                size_t n_train_c = static_cast<size_t>(num_c * tr);
                size_t n_val_c = static_cast<size_t>(num_c * val);

                std::vector<uint64_t> train_items, val_items, test_items;
                for (size_t i = 0; i < n_train_c; ++i)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[i]);
                    train_items.insert(train_items.end(), vec_ids.begin(), vec_ids.end());
                }
                for (size_t i = n_train_c; i < n_train_c + n_val_c; ++i)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[i]);
                    val_items.insert(val_items.end(), vec_ids.begin(), vec_ids.end());
                }
                for (size_t i = n_train_c + n_val_c; i < num_c; ++i)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[i]);
                    test_items.insert(test_items.end(), vec_ids.begin(), vec_ids.end());
                }
                m->split_mgr->reset();
                m->split_mgr->train_indices = std::move(train_items);
                m->split_mgr->val_indices = std::move(val_items);
                m->split_mgr->test_indices = std::move(test_items);
                total_vectors = m->split_mgr->train_indices.size() +
                                m->split_mgr->val_indices.size() +
                                m->split_mgr->test_indices.size();
            }
            else if (strategy == "TEMPORAL")
            {
                if (!m->meta_index) return "ERR: Metadata Index not enabled\n";
                if (strat_key.empty()) return "ERR: TEMPORAL requires a key\n";
                auto groups = m->meta_index->get_groups(strat_key);
                if (groups.empty()) return "ERR: No metadata found\n";
                std::vector<uint64_t> all_ordered;
                for (const auto &kv : groups)
                    all_ordered.insert(all_ordered.end(), kv.second.begin(), kv.second.end());
                size_t n = all_ordered.size();
                size_t n_train = static_cast<size_t>(n * tr);
                size_t n_val = static_cast<size_t>(n * val);
                std::vector<uint64_t> train_items, val_items, test_items;
                size_t idx = 0;
                if (n_train > 0)
                {
                    train_items.insert(train_items.end(), all_ordered.begin(), all_ordered.begin() + n_train);
                    idx += n_train;
                }
                if (n_val > 0 && idx < n)
                {
                    val_items.insert(val_items.end(), all_ordered.begin() + idx, all_ordered.begin() + idx + n_val);
                    idx += n_val;
                }
                if (idx < n)
                    test_items.insert(test_items.end(), all_ordered.begin() + idx, all_ordered.end());
                m->split_mgr->reset();
                m->split_mgr->train_indices = std::move(train_items);
                m->split_mgr->val_indices = std::move(val_items);
                m->split_mgr->test_indices = std::move(test_items);
                total_vectors = n;
            }

            if (strategy == "RANDOM")
            {
                std::vector<uint64_t> all_labels;
                try { if (m->orbit) all_labels = m->orbit->get_all_labels(); }
                catch (...) { all_labels.clear(); }

                if (!all_labels.empty())
                    m->split_mgr->execute_split_with_items(all_labels, tr, val, te);
                else
                {
                    if (total_vectors == 0) return "ERR: Membrance empty\n";
                    m->split_mgr->execute_random_split(total_vectors, tr, val, te);
                }
            }

            if (m->split_mgr->save(m->data_path))
            {
                std::ostringstream ss;
                ss << "OK: Split " << total_vectors << " vectors into "
                   << m->split_mgr->train_indices.size() << " train, "
                   << m->split_mgr->val_indices.size() << " val, "
                   << m->split_mgr->test_indices.size() << " test";
                return ss.str() + "\n";
            }
            return "ERR: Failed to save split file\n";
        }

        // --- 4. ITERATE ---
        if (up.rfind("ITERATE", 0) == 0)
        {
            auto parts2 = utils::split_ws(cmd);
            if (parts2.size() < 3)
                return "ERR: Usage: ITERATE <name> <mode> [split] [off] [lim] [BATCH <n>]\n";

            std::string name = parts2[1];
            std::string mode = utils::to_upper(parts2[2]);
            auto *m = db->get_membrance(name);
            if (!m) return "ERR: Membrance not found\n";

            size_t dim = m->dim;

            // parse optional split token and off/lim
            size_t off = 0;
            size_t lim = 1000000;
            const std::vector<uint64_t> *idxs = nullptr;
            auto choose_split = [&](const std::string &s) -> const std::vector<uint64_t> *
            {
                if (!m->split_mgr) return nullptr;
                if (s == "TRAIN") return &m->split_mgr->train_indices;
                if (s == "VAL") return &m->split_mgr->val_indices;
                if (s == "TEST") return &m->split_mgr->test_indices;
                return nullptr;
            };

            size_t cur_tok = 3;
            if (parts2.size() > 3)
            {
                std::string maybe = utils::to_upper(parts2[3]);
                const std::vector<uint64_t> *cand = choose_split(maybe);
                if (cand) { idxs = cand; cur_tok = 4; }
            }
            if (!idxs && m->split_mgr) idxs = &m->split_mgr->train_indices;

            if (parts2.size() > cur_tok)
            {
                try { off = std::stoul(parts2[cur_tok]); } catch(...) { off = 0; }
            }
            if (parts2.size() > cur_tok + 1)
            {
                try { lim = std::stoul(parts2[cur_tok + 1]); } catch(...) { lim = 1000000; }
            }

            // optional BATCH param
            size_t batch_size_param = 0;
            for (size_t t = 3; t < parts2.size(); ++t)
            {
                if (utils::to_upper(parts2[t]) == "BATCH" && t + 1 < parts2.size())
                {
                    try { batch_size_param = std::stoul(parts2[t + 1]); } catch(...) { batch_size_param = 0; }
                    break;
                }
            }

            // helper: dtype & elem_size
            auto get_membrance_dtype = [&](std::string &out_dtype_str, uint32_t &out_elem_size)
            {
                if (m->hot_tier) { out_elem_size = m->hot_tier->element_size(); out_dtype_str = m->hot_tier->data_type_string(); }
                else { out_elem_size = static_cast<uint32_t>(pomai::core::dtype_size(m->data_type)); out_dtype_str = pomai::core::dtype_name(m->data_type); }
            };

            // ---- TRIPLET ----
            if (mode == "TRIPLET")
            {
                if (!m->meta_index || !m->orbit) return "OK BINARY float32 0 " + std::to_string(dim) + " 0\n";
                if (parts2.size() < 4) return "ERR: ITERATE <name> TRIPLET <key> [limit]\n";

                std::string key = parts2[3];
                size_t limit = 100;
                if (parts2.size() >= 6) { try { limit = std::stoul(parts2[5]); } catch(...) {} }
                else if (parts2.size() == 5) { try { limit = std::stoul(parts2[4]); } catch(...) {} }
                if (limit == 0) limit = 1;

                auto groups = m->meta_index->get_groups(key);
                std::vector<std::string> cls;
                for (const auto &kv : groups) if (kv.second.size() >= 2) cls.push_back(kv.first);
                if (cls.size() < 2) return "OK BINARY float32 0 " + std::to_string(dim) + " 0\n";

                std::string dtype_str; uint32_t elem_size; get_membrance_dtype(dtype_str, elem_size);
                size_t per_vec = static_cast<size_t>(elem_size) * dim;
                size_t total_bytes = limit * 3 * per_vec;

                std::string out = supplier::make_header("OK BINARY", dtype_str, limit, dim, total_bytes);
                out.reserve(out.size() + total_bytes);

                std::mt19937 rng(std::random_device{}());
                std::vector<uint64_t> trip_ids(3);
                std::vector<std::string> trip_raw;

                for (size_t i = 0; i < limit; ++i)
                {
                    const auto &c = cls[std::uniform_int_distribution<size_t>(0, cls.size() - 1)(rng)];
                    std::string c2 = c;
                    while (c2 == c) c2 = cls[std::uniform_int_distribution<size_t>(0, cls.size() - 1)(rng)];
                    const auto &ids = groups.at(c);
                    const auto &ids2 = groups.at(c2);

                    trip_ids[0] = ids[std::uniform_int_distribution<size_t>(0, ids.size() - 1)(rng)];
                    trip_ids[1] = trip_ids[0];
                    while (trip_ids[1] == trip_ids[0]) trip_ids[1] = ids[std::uniform_int_distribution<size_t>(0, ids.size() - 1)(rng)];
                    trip_ids[2] = ids2[std::uniform_int_distribution<size_t>(0, ids2.size() - 1)(rng)];

                    // Batch fetch 3 ids
                    fetch_raws_with_fallback(db, m, trip_ids, per_vec, trip_raw);
                    append_raws_into(out, trip_raw, per_vec);
                }
                return out;
            }

            // ---- PAIR ----
            if (mode == "PAIR")
            {
                if (!idxs) return "ERR: No split indices available for PAIR\n";
                if (off >= idxs->size()) return "OK BINARY_PAIR float32 0 " + std::to_string(dim) + " 0\n";

                std::string dtype_str; uint32_t elem_size; get_membrance_dtype(dtype_str, elem_size);
                size_t per_vec = static_cast<size_t>(elem_size) * dim;
                size_t cnt = std::min(lim, idxs->size() - off);
                size_t per = sizeof(uint64_t) + per_vec;

                // Batched streaming path
                if (batch_size_param > 0)
                {
                    std::string out;
                    size_t processed = 0;
                    std::vector<uint64_t> ids;
                    std::vector<std::string> raws;

                    while (processed < cnt)
                    {
                        size_t this_batch = std::min<size_t>(batch_size_param, cnt - processed);
                        ids.clear(); ids.reserve(this_batch);
                        for (size_t i = 0; i < this_batch; ++i) ids.push_back((*idxs)[off + processed + i]);

                        fetch_raws_with_fallback(db, m, ids, per_vec, raws);

                        out += supplier::make_header("OK BINARY_PAIR", dtype_str, this_batch, dim, this_batch * per);
                        out.reserve(out.size() + this_batch * per);

                        for (size_t i = 0; i < this_batch; ++i)
                        {
                            uint64_t id = ids[i];
                            out.append(reinterpret_cast<const char *>(&id), sizeof(id));
                            const std::string &raw = raws[i];
                            if (raw.size() >= per_vec) out.append(raw.data(), per_vec);
                            else { out.append(raw.data(), raw.size()); out.append(per_vec - raw.size(), '\0'); }
                        }
                        processed += this_batch;
                    }
                    return out;
                }
                else
                {
                    // Single-response path: fetch all ids at once
                    std::vector<uint64_t> ids;
                    ids.reserve(cnt);
                    for (size_t i = 0; i < cnt; ++i) ids.push_back((*idxs)[off + i]);

                    std::vector<std::string> raws;
                    fetch_raws_with_fallback(db, m, ids, per_vec, raws);

                    size_t total_bytes = cnt * per;
                    std::string header = "OK BINARY_PAIR " + dtype_str + " " + std::to_string(cnt) + " " + std::to_string(dim) + " " + std::to_string(total_bytes) + "\n";
                    std::string out;
                    out.reserve(header.size() + total_bytes);
                    out += header;

                    for (size_t i = 0; i < cnt; ++i)
                    {
                        uint64_t id = ids[i];
                        out.append(reinterpret_cast<const char *>(&id), sizeof(id));
                        const std::string &raw = raws[i];
                        if (raw.size() >= per_vec) out.append(raw.data(), per_vec);
                        else { out.append(raw.data(), raw.size()); out.append(per_vec - raw.size(), '\0'); }
                    }
                    return out;
                }
            }

            // ---- TRAIN/VAL/TEST ----
            if (mode == "TRAIN" || mode == "VAL" || mode == "TEST")
            {
                if (!idxs) return "ERR: No split indices available for ITERATE\n";
                if (off >= idxs->size()) return "OK BINARY float32 0 " + std::to_string(dim) + " 0\n";

                std::string dtype_str; uint32_t elem_size; get_membrance_dtype(dtype_str, elem_size);
                size_t per_vec = static_cast<size_t>(elem_size) * dim;
                size_t cnt = std::min(lim, idxs->size() - off);

                if (batch_size_param > 0)
                {
                    std::string out;
                    size_t processed = 0;
                    std::vector<uint64_t> ids;
                    std::vector<std::string> raws;

                    while (processed < cnt)
                    {
                        size_t this_batch = std::min<size_t>(batch_size_param, cnt - processed);
                        ids.clear(); ids.reserve(this_batch);
                        for (size_t i = 0; i < this_batch; ++i) ids.push_back((*idxs)[off + processed + i]);

                        fetch_raws_with_fallback(db, m, ids, per_vec, raws);

                        size_t total_bytes = this_batch * per_vec;
                        out += supplier::make_header("OK BINARY", dtype_str, this_batch, dim, total_bytes);
                        out.reserve(out.size() + total_bytes);
                        append_raws_into(out, raws, per_vec);

                        processed += this_batch;
                    }
                    return out;
                }
                else
                {
                    std::vector<uint64_t> ids;
                    ids.reserve(cnt);
                    for (size_t i = 0; i < cnt; ++i) ids.push_back((*idxs)[off + i]);

                    std::vector<std::string> raws;
                    fetch_raws_with_fallback(db, m, ids, per_vec, raws);

                    size_t total_bytes = cnt * per_vec;
                    std::string header = supplier::make_header("OK BINARY", dtype_str, cnt, dim, total_bytes);
                    std::string out;
                    out.reserve(header.size() + total_bytes);
                    out += header;
                    append_raws_into(out, raws, per_vec);
                    return out;
                }
            }

            return "ERR: Invalid ITERATE mode (TRAIN/VAL/TEST/TRIPLET/PAIR)\n";
        }

        // CREATE MEMBRANCE ...
        if (up.rfind("CREATE MEMBRANCE", 0) == 0)
        {
            size_t pos_dim = up.find(" DIM ");
            if (pos_dim == std::string::npos) return "ERR: CREATE MEMBRANCE missing DIM\n";

            std::string name = utils::trim(cmd.substr(std::string("CREATE MEMBRANCE").size(), pos_dim - std::string("CREATE MEMBRANCE").size()));
            std::string tail = utils::trim(cmd.substr(pos_dim + 5));

            std::istringstream iss(tail);
            size_t dim = 0;
            std::string token;
            size_t ram_mb = 256;
            std::string data_type = "float32";

            if (!(iss >> token)) return "ERR: invalid DIM\n";
            try { dim = static_cast<size_t>(std::stoul(token)); } catch(...) { return "ERR: invalid DIM\n"; }

            while (iss >> token)
            {
                std::string up_tok = utils::to_upper(token);
                if (up_tok == "DATA_TYPE" || up_tok == "DATA-TYPE")
                {
                    std::string dt;
                    if (!(iss >> dt)) return "ERR: DATA_TYPE requires a value\n";
                    std::transform(dt.begin(), dt.end(), dt.begin(), ::tolower);
                    if (dt != "float32" && dt != "float64" && dt != "int32" && dt != "int8" && dt != "float16")
                        return std::string("ERR: unsupported DATA_TYPE '") + dt + "'\n";
                    data_type = dt;
                }
                else if (up_tok == "RAM")
                {
                    std::string r;
                    if (!(iss >> r)) return "ERR: RAM requires a numeric value\n";
                    try { ram_mb = static_cast<size_t>(std::stoul(r)); } catch(...) { return "ERR: invalid RAM value\n"; }
                }
            }

            if (dim == 0) return "ERR: invalid DIM\n";

            pomai::core::MembranceConfig cfg;
            cfg.dim = dim; cfg.ram_mb = ram_mb;
            try { cfg.data_type = pomai::core::parse_dtype(data_type); } catch(...) { return std::string("ERR: unsupported DATA_TYPE '") + data_type + "'\n"; }

            bool ok = db->create_membrance(name, cfg);
            if (ok) {
                std::ostringstream ss;
                ss << "OK: created membrance " << name << " dim=" << dim << " data_type=" << data_type << " ram=" << ram_mb << "MB\n";
                return ss.str();
            }
            return "ERR: create failed (exists or invalid)\n";
        }

        // SEARCH ...
        if (utils::to_upper(cmd).rfind("SEARCH ", 0) == 0)
        {
            auto parts2 = utils::split_ws(cmd);
            if (parts2.size() < 5) return "ERR: usage SEARCH <name> QUERY <vec> TOP <k>\n";

            std::string name = parts2[1];
            size_t query_idx = 0;
            for (size_t i = 2; i < parts2.size(); ++i)
            {
                if (utils::to_upper(parts2[i]) == "QUERY") { query_idx = i; break; }
            }
            if (query_idx == 0 || query_idx + 1 >= parts2.size()) return "ERR: missing QUERY <vec>\n";

            std::string vec_str = parts2[query_idx + 1];
            if (vec_str.size() >= 2 && vec_str.front() == '(' && vec_str.back() == ')') vec_str = vec_str.substr(1, vec_str.size() - 2);

            std::vector<float> query_vec;
            if (!utils::parse_vector(vec_str, query_vec)) return "ERR: invalid vector format\n";

            size_t k = 10;
            for (size_t i = query_idx + 2; i < parts2.size(); ++i)
            {
                if (utils::to_upper(parts2[i]) == "TOP" && i + 1 < parts2.size()) {
                    try { k = std::stoul(parts2[i + 1]); } catch(...) {}
                    break;
                }
            }

            auto *m = db->get_membrance(name);
            if (!m) return "ERR: membrance not found\n";
            if (!m->orbit) return "ERR: engine not ready\n";
            if (query_vec.size() != m->dim) return "ERR: query dimension mismatch\n";

            auto results = m->orbit->search(query_vec.data(), k);
            std::ostringstream ss;
            ss << "OK " << results.size() << "\n";
            for (const auto &p : results) ss << p.first << " " << std::fixed << std::setprecision(6) << p.second << "\n";
            return ss.str();
        }

        // DROP MEMBRANCE
        if (up.rfind("DROP MEMBRANCE", 0) == 0)
        {
            auto parts2 = utils::split_ws(cmd);
            if (parts2.size() < 3) return "ERR: DROP MEMBRANCE <name>;\n";
            std::string name = parts2[2];
            bool ok = db->drop_membrance(name);
            return ok ? std::string("OK: dropped ") + name + "\n" : std::string("ERR: drop failed\n");
        }

        if (up == "EXEC CHECKPOINT")
        {
            if (db->checkpoint_all()) return "OK: Checkpoint completed. WAL truncated.\n";
            return "ERR: Checkpoint failed partially.\n";
        }

        // GET MEMBRANCE INFO ...
        if (up.rfind("GET MEMBRANCE", 0) == 0)
        {
            auto parts2 = utils::split_ws(cmd);
            std::string name;
            bool ok_parse = false;
            if (parts2.size() >= 4 && utils::to_upper(parts2[2]) == "INFO") { name = parts2[3]; ok_parse = true; }
            else if (parts2.size() >= 4 && utils::to_upper(parts2[3]) == "INFO") { name = parts2[2]; ok_parse = true; }
            else if (parts2.size() >= 3 && utils::to_upper(parts2[2]) == "INFO") {
                if (state.current_membrance.empty()) return "ERR: no current membrance (USE <name>)\n";
                name = state.current_membrance; ok_parse = true;
            }
            if (!ok_parse) return "ERR: expected 'GET MEMBRANCE INFO ...'\n";

            auto *m = db->get_membrance(name);
            if (!m) return std::string("ERR: membrance not found: ") + name + "\n";

            pomai::ai::orbit::MembranceInfo info;
            try { info = m->orbit->get_info(); } catch(...) { info.dim = m->dim; info.num_vectors = 0; info.disk_bytes = 0; }

            if (info.disk_bytes == 0)
            {
                try
                {
                    std::filesystem::path dp(m->data_path);
                    if (std::filesystem::exists(dp))
                    {
                        for (auto const &entry : std::filesystem::recursive_directory_iterator(dp))
                        {
                            if (!entry.is_regular_file()) continue;
                            std::error_code ec;
                            uint64_t fsz = static_cast<uint64_t>(entry.file_size(ec));
                            if (!ec) info.disk_bytes += fsz;
                        }
                    }
                }
                catch (...) {}
            }

            size_t n_train = 0, n_val = 0, n_test = 0;
            if (m->split_mgr) { n_train = m->split_mgr->train_indices.size(); n_val = m->split_mgr->val_indices.size(); n_test = m->split_mgr->test_indices.size(); }

            size_t feature_dim = (m->dim > 0) ? m->dim : info.dim;
            size_t total_vectors = info.num_vectors;
            if (total_vectors == 0) total_vectors = n_train + n_val + n_test;

            std::ostringstream ss;
            ss << "MEMBRANCE: " << name << "\n";
            ss << "--- AI Contract ---\n";
            ss << " feature_dim: " << feature_dim << "\n";
            ss << " metric: L2\n";
            ss << " data_type: " << pomai::core::dtype_name(m->data_type) << "\n";
            ss << " total_vectors: " << total_vectors << "\n";
            ss << " split_train: " << n_train << "\n";
            ss << " split_val: " << n_val << "\n";
            ss << " split_test: " << n_test << "\n";
            ss << "--- Storage Stats ---\n";
            ss << " disk_bytes: " << utils::bytes_human(info.disk_bytes) << "\n";
            ss << " ram_mb_configured: " << m->ram_mb << "\n";
            return ss.str();
        }

        // INSERT handling (unchanged behaviour: batch insert)
        {
            std::string upstart = utils::to_upper(cmd.substr(0, std::min<size_t>(cmd.size(), 16)));
            if (upstart.rfind("INSERT INTO", 0) == 0 || utils::to_upper(cmd).rfind("INSERT VALUES", 0) == 0)
            {
                std::string body = cmd;
                std::string name;
                bool has_explicit_into = (utils::to_upper(cmd).find("INTO") != std::string::npos);
                if (!has_explicit_into && utils::to_upper(cmd).rfind("INSERT VALUES", 0) == 0)
                {
                    if (state.current_membrance.empty()) return "ERR: no current membrance (USE <name>)\n";
                    name = state.current_membrance;
                    body = std::string("INSERT INTO ") + name + " " + cmd.substr(std::string("INSERT VALUES").size());
                }
                size_t pos_into = utils::to_upper(body).find("INTO");
                size_t pos_values = utils::to_upper(body).find("VALUES");
                if (pos_values == std::string::npos) return "ERR: INSERT missing VALUES\n";
                if (pos_into == std::string::npos) return "ERR: INSERT missing INTO\n";
                size_t name_start = pos_into + 4;
                name = utils::trim(body.substr(name_start, pos_values - name_start));
                if (name.empty()) return "ERR: missing membrance name\n";

                auto *m = db->get_membrance(name);
                if (!m) return "ERR: membrance not found\n";

                size_t pos_after_values = pos_values + std::string("VALUES").size();
                size_t cur = body.find_first_not_of(" \t\r\n", pos_after_values);
                if (cur == std::string::npos || body[cur] != '(') return "ERR: VALUES syntax\n";

                std::vector<std::pair<std::string, std::vector<float>>> tuples;
                tuples.reserve(8);

                while (true)
                {
                    if (cur >= body.size() || body[cur] != '(') break;
                    size_t depth = 0;
                    size_t start = cur;
                    size_t end = std::string::npos;
                    for (size_t p = cur; p < body.size(); ++p)
                    {
                        if (body[p] == '(') ++depth;
                        else if (body[p] == ')') { --depth; if (depth == 0) { end = p; cur = p + 1; break; } }
                    }
                    if (end == std::string::npos) return "ERR: unmatched parentheses in VALUES\n";

                    std::string tuple_text = body.substr(start + 1, end - start - 1);
                    size_t vec_lb = tuple_text.find('[');
                    size_t vec_rb = tuple_text.rfind(']');
                    if (vec_lb == std::string::npos || vec_rb == std::string::npos || vec_rb <= vec_lb) return "ERR: tuple vector syntax\n";
                    size_t comma_before_vec = tuple_text.rfind(',', vec_lb);
                    if (comma_before_vec == std::string::npos) return "ERR: tuple syntax (label, [vec])\n";
                    std::string label_tok = utils::trim(tuple_text.substr(0, comma_before_vec));
                    std::string_view vec_view(tuple_text.data() + vec_lb + 1, vec_rb - vec_lb - 1);

                    auto vec_vals = utils::parse_float_list_sv(utils::trim_sv(vec_view));
                    if (vec_vals.size() != m->dim)
                    {
                        std::ostringstream ss;
                        ss << "ERR: dim mismatch expected=" << m->dim << " got=" << vec_vals.size() << "\n";
                        return ss.str();
                    }
                    tuples.emplace_back(label_tok, std::move(vec_vals));

                    cur = body.find_first_not_of(" \t\r\n", cur);
                    if (cur == std::string::npos) break;
                    if (body[cur] == ',') { ++cur; cur = body.find_first_not_of(" \t\r\n", cur); continue; }
                    break;
                }

                // parse optional TAGS
                std::vector<pomai::core::Tag> global_tags;
                size_t pos_tags = utils::to_upper(body).find(" TAGS ", pos_after_values);
                if (pos_tags != std::string::npos)
                {
                    size_t t_l = body.find('(', pos_tags);
                    size_t t_r = body.find(')', t_l);
                    if (t_l != std::string::npos && t_r != std::string::npos && t_r > t_l)
                    {
                        std::string tags_inside = body.substr(t_l + 1, t_r - t_l - 1);
                        global_tags = utils::parse_tags_list(tags_inside);
                    }
                }

                std::vector<std::pair<uint64_t, std::vector<float>>> batch_data;
                batch_data.reserve(tuples.size());
                std::vector<uint64_t> inserted_hashes;
                if (!global_tags.empty()) inserted_hashes.reserve(tuples.size());

                for (auto &tp : tuples)
                {
                    uint64_t label_hash = utils::hash_key(tp.first);
                    batch_data.emplace_back(label_hash, std::move(tp.second));
                    if (!global_tags.empty()) inserted_hashes.push_back(label_hash);
                }

                bool ok_batch = false;
                try { ok_batch = db->insert_batch(name, batch_data); } catch(...) { ok_batch = false; }

                if (ok_batch && !global_tags.empty() && m->meta_index)
                {
                    for (uint64_t h : inserted_hashes)
                    {
                        try { m->meta_index->add_tags(h, global_tags); } catch(...) {}
                    }
                }

                std::ostringstream ss;
                ss << "OK: inserted " << (ok_batch ? batch_data.size() : 0) << " / " << tuples.size() << (ok_batch ? " (batch)" : " (failed)") << "\n";
                return ss.str();
            }
        }

        return "ERR: unknown command\n";
    }

} // namespace pomai::server