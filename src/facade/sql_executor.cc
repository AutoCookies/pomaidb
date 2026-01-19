// src/facade/sql_executor.cc
//
// Implementation of SqlExecutor extracted from server.h so the SQL/text protocol
// logic can be reused outside of the server class. The code preserves original
// behavior (no logic changes), but uses the injected PomaiDB/WhisperGrain/ClientState
// instead of server members.
//
// Enhancement:
// - ITERATE now supports an optional "BATCH <n>" token to stream results in
//   batches of at most n vectors. If BATCH is not provided, behavior is
//   unchanged (single response containing up to lim entries).
// - TRIPLET no longer requires an explicit <limit> token: if missing we compute
//   a sensible default from available data.
#include "src/facade/sql_executor.h"

#include "src/facade/server_utils.h"
#include "src/facade/data_supplier.h"
#include "src/core/pomai_db.h"
#include "src/core/metadata_index.h"
#include "src/ai/whispergrain.h"
#include "src/core/metrics.h"
#include "src/core/types.h" // added to report/parse data_type names

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
            auto parts = utils::split_ws(cmd);
            if (parts.size() >= 2)
            {
                std::string name = parts[1];
                if (!name.empty() && name.back() == ';')
                    name.pop_back();
                state.current_membrance = name;
                return std::string("OK: switched to membrance ") + name + "\n";
            }
            return "ERR: USE <name>;\n";
        }

        // --- 3. EXEC SPLIT ---
        if (up.rfind("EXEC SPLIT", 0) == 0)
        {
            auto parts = utils::split_ws(cmd);
            if (parts.size() < 6)
                return "ERR: Usage: EXEC SPLIT <name> <tr%> <val%> <te%> [STRATIFIED <key> | CLUSTER]\n";

            std::string name = parts[2];
            float tr = 0, val = 0, te = 0;
            try
            {
                tr = std::stof(parts[3]);
                val = std::stof(parts[4]);
                te = std::stof(parts[5]);
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

            // --- Detect Strategy ---
            std::string strategy = "RANDOM";
            std::string strat_key = "";

            if (parts.size() >= 7)
            {
                std::string type = utils::to_upper(parts[6]);
                if (type == "STRATIFIED")
                {
                    strategy = "STRATIFIED";
                    if (parts.size() >= 8)
                        strat_key = parts[7];
                    else
                        return "ERR: STRATIFIED requires a key (e.g. STRATIFIED class)\n";
                }
                else if (type == "CLUSTER")
                {
                    strategy = "CLUSTER";
                }
            }

            // Check Empty (fallback to 0 if info fails)
            size_t total_vectors = 0;
            try
            {
                total_vectors = m->orbit->get_info().num_vectors;
            }
            catch (...)
            {
            }

            // --- Strategy Dispatch ---

            // 1. STRATIFIED SPLIT
            if (strategy == "STRATIFIED")
            {
                if (!m->meta_index)
                    return "ERR: Metadata Index not enabled for this membrance\n";

                auto groups = m->meta_index->get_groups(strat_key);
                if (groups.empty())
                    return "ERR: No metadata found for key '" + strat_key + "'\n";

                // Flatten map -> vector for manager
                std::vector<uint64_t> items;
                std::vector<uint64_t> labels;
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
            // 2. CLUSTER SPLIT
            else if (strategy == "CLUSTER")
            {
                if (!m->orbit)
                    return "ERR: Orbit engine required for Cluster split\n";

                size_t num_c = m->orbit->num_centroids();
                if (num_c == 0)
                    return "ERR: No centroids found (Train model first)\n";

                std::vector<uint32_t> cids(num_c);
                std::iota(cids.begin(), cids.end(), 0);

                std::mt19937 g(std::random_device{}());
                std::shuffle(cids.begin(), cids.end(), g);

                size_t n_train_c = static_cast<size_t>(num_c * tr);
                size_t n_val_c = static_cast<size_t>(num_c * val);

                std::vector<uint64_t> train_items, val_items, test_items;
                size_t idx = 0;

                for (; idx < n_train_c; ++idx)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[idx]);
                    train_items.insert(train_items.end(), vec_ids.begin(), vec_ids.end());
                }
                for (; idx < n_train_c + n_val_c; ++idx)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[idx]);
                    val_items.insert(val_items.end(), vec_ids.begin(), vec_ids.end());
                }
                for (; idx < num_c; ++idx)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[idx]);
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
                if (!m->meta_index)
                    return "ERR: Metadata Index not enabled\n";
                if (strat_key.empty())
                    return "ERR: TEMPORAL requires a key (e.g. TEMPORAL date)\n";

                auto groups = m->meta_index->get_groups(strat_key);
                if (groups.empty())
                    return "ERR: No metadata found for key\n";

                std::vector<uint64_t> all_ordered_items;
                for (const auto &kv : groups)
                {
                    all_ordered_items.insert(all_ordered_items.end(), kv.second.begin(), kv.second.end());
                }

                if (all_ordered_items.empty())
                    return "ERR: No items found\n";

                size_t n = all_ordered_items.size();
                size_t n_train = static_cast<size_t>(n * tr);
                size_t n_val = static_cast<size_t>(n * val);

                std::vector<uint64_t> train_items, val_items, test_items;

                auto it = all_ordered_items.begin();

                if (n_train > 0)
                {
                    train_items.assign(it, it + n_train);
                    it += n_train;
                }

                if (n_val > 0)
                {
                    val_items.assign(it, it + n_val);
                    it += n_val;
                }

                if (it != all_ordered_items.end())
                {
                    test_items.assign(it, all_ordered_items.end());
                }

                m->split_mgr->reset();
                m->split_mgr->train_indices = std::move(train_items);
                m->split_mgr->val_indices = std::move(val_items);
                m->split_mgr->test_indices = std::move(test_items);

                total_vectors = n;
            }

            if (strategy == "RANDOM")
            {
                std::vector<uint64_t> all_labels;
                try
                {
                    if (m->orbit)
                    {
                        all_labels = m->orbit->get_all_labels();
                    }
                }
                catch (...)
                {
                    all_labels.clear();
                }

                if (!all_labels.empty())
                {
                    m->split_mgr->execute_split_with_items(all_labels, tr, val, te);
                }
                else
                {
                    if (total_vectors == 0)
                        return "ERR: Membrance is empty (0 vectors)\n";
                    m->split_mgr->execute_random_split(total_vectors, tr, val, te);
                }
            }

            if (m->split_mgr->save(m->data_path))
            {
                std::stringstream ss;
                ss << "OK: Split " << total_vectors << " vectors into "
                   << m->split_mgr->train_indices.size() << " train, "
                   << m->split_mgr->val_indices.size() << " val, "
                   << m->split_mgr->test_indices.size() << " test";
                return ss.str() + "\n";
            }
            else
            {
                return "ERR: Failed to save split file to disk\n";
            }
        }

        // --- 4. ITERATE ---
        if (up.rfind("ITERATE", 0) == 0)
        {
            auto parts = utils::split_ws(cmd);
            if (parts.size() < 3)
                return "ERR: Usage: ITERATE <name> <mode> [split] [off] [lim] [BATCH <n>]\n";

            std::string name = parts[1];
            std::string mode = utils::to_upper(parts[2]);
            auto *m = db->get_membrance(name);
            if (!m)
                return "ERR: Membrance not found\n";

            size_t dim = m->dim;

            // parse optional split token and off/lim (unchanged)
            size_t off = 0;
            size_t lim = 1000000;
            const std::vector<uint64_t> *idxs = nullptr;
            auto choose_split = [&](const std::string &s) -> const std::vector<uint64_t> *
            {
                if (!m->split_mgr)
                    return nullptr;
                if (s == "TRAIN")
                    return &m->split_mgr->train_indices;
                if (s == "VAL")
                    return &m->split_mgr->val_indices;
                if (s == "TEST")
                    return &m->split_mgr->test_indices;
                return nullptr;
            };

            size_t cur_tok = 3;
            if (parts.size() > 3)
            {
                std::string maybe = utils::to_upper(parts[3]);
                const std::vector<uint64_t> *cand = choose_split(maybe);
                if (cand)
                {
                    idxs = cand;
                    cur_tok = 4;
                }
            }
            if (!idxs && m->split_mgr)
                idxs = &m->split_mgr->train_indices;

            if (parts.size() > cur_tok)
            {
                try
                {
                    off = std::stoul(parts[cur_tok]);
                }
                catch (...)
                {
                    off = 0;
                }
            }
            if (parts.size() > cur_tok + 1)
            {
                try
                {
                    lim = std::stoul(parts[cur_tok + 1]);
                }
                catch (...)
                {
                    lim = 1000000;
                }
            }

            // New: parse optional BATCH token anywhere in parts
            size_t batch_size_param = 0;
            for (size_t t = 3; t < parts.size(); ++t)
            {
                if (utils::to_upper(parts[t]) == "BATCH" && t + 1 < parts.size())
                {
                    try
                    {
                        batch_size_param = std::stoul(parts[t + 1]);
                        if (batch_size_param == 0) batch_size_param = 0; // treat 0 as disabled
                    }
                    catch (...)
                    {
                        batch_size_param = 0;
                    }
                    break;
                }
            }

            // Helper: get dtype string & elem_size for this membrance/hot_tier
            auto get_membrance_dtype = [&](std::string &out_dtype_str, uint32_t &out_elem_size)
            {
                if (m->hot_tier)
                {
                    out_elem_size = m->hot_tier->element_size();
                    out_dtype_str = m->hot_tier->data_type_string();
                }
                else
                {
                    out_elem_size = static_cast<uint32_t>(pomai::core::dtype_size(m->data_type));
                    out_dtype_str = pomai::core::dtype_name(m->data_type);
                }
            };

            // MODE: TRIPLET (produce 3 vectors per record) - now returns raw dtype bytes
            if (mode == "TRIPLET")
            {
                if (!m->meta_index || !m->orbit)
                    return "OK BINARY float32 0 " + std::to_string(dim) + " 0\n";

                // key can be parts[3] or parts[cur_tok] depending on whether a split token was present.
                std::string key;
                size_t key_tok_index = 3;
                if (cur_tok > 3)
                    key_tok_index = cur_tok; // split token consumed token at 3, but TRIPLET syntax expects key at 3 originally
                if (parts.size() > key_tok_index)
                    key = parts[key_tok_index];
                else
                    return "ERR: ITERATE <name> TRIPLET <key> [<limit>] [BATCH <n>]\n";

                // optional limit: if provided (e.g. parts[4] exists and is numeric) use it; otherwise compute default.
                size_t limit = 0;
                bool parsed_limit = false;
                // find next token after key token to try parse limit
                size_t maybe_limit_idx = key_tok_index + 1;
                if (maybe_limit_idx < parts.size())
                {
                    // skip tokens that are "BATCH" or other flags
                    if (utils::to_upper(parts[maybe_limit_idx]) != "BATCH")
                    {
                        try
                        {
                            limit = std::stoul(parts[maybe_limit_idx]);
                            parsed_limit = true;
                        }
                        catch (...)
                        {
                            parsed_limit = false;
                        }
                    }
                }

                auto groups = m->meta_index->get_groups(key);
                // collect valid classes with >=2 elements
                std::vector<std::string> cls;
                size_t total_items = 0;
                for (const auto &kv : groups)
                {
                    if (kv.second.size() >= 2)
                    {
                        cls.push_back(kv.first);
                        total_items += kv.second.size();
                    }
                }

                if (cls.size() < 2 || total_items < 2)
                    return "OK BINARY float32 0 " + std::to_string(dim) + " 0\n";

                // compute default limit if not provided: number of available triplets approximated by total_items/3
                if (!parsed_limit)
                {
                    size_t default_limit = total_items / 3;
                    if (default_limit == 0)
                        default_limit = 1;
                    limit = default_limit;
                }

                // determine dtype & element size
                std::string dtype_str;
                uint32_t elem_size;
                get_membrance_dtype(dtype_str, elem_size);

                size_t per_vec = static_cast<size_t>(elem_size) * dim;

                // If batching requested - stream in chunks of batch_size_param triplets
                if (batch_size_param > 0)
                {
                    std::string out;
                    std::mt19937 rng(std::random_device{}());
                    size_t remaining = limit;
                    while (remaining > 0)
                    {
                        size_t this_batch = std::min<size_t>(batch_size_param, remaining);
                        size_t total_bytes = this_batch * 3 * per_vec;
                        out += supplier::make_header("OK BINARY", dtype_str, this_batch, dim, total_bytes);
                        for (size_t i = 0; i < this_batch; ++i)
                        {
                            const auto &c = cls[std::uniform_int_distribution<size_t>(0, cls.size() - 1)(rng)];
                            const auto &ids = groups[c];
                            uint64_t ida = ids[std::uniform_int_distribution<size_t>(0, ids.size() - 1)(rng)];
                            uint64_t idp = ida;
                            int tries = 0;
                            while (idp == ida && ++tries < 20)
                                idp = ids[std::uniform_int_distribution<size_t>(0, ids.size() - 1)(rng)];
                            uint64_t idn = ida;
                            while (idn == ida)
                            {
                                const auto &c2 = cls[std::uniform_int_distribution<size_t>(0, cls.size() - 1)(rng)];
                                if (c2 == c)
                                    continue;
                                const auto &ids2 = groups[c2];
                                idn = ids2[std::uniform_int_distribution<size_t>(0, ids2.size() - 1)(rng)];
                            }

                            // fetch raw bytes for ida,idp,idn and append
                            pomai::core::DataType got_dt;
                            uint32_t got_elem;
                            std::string raw;
                            supplier::fetch_vector_raw(m, ida, raw, dim, got_dt, got_elem);
                            if (raw.size() < per_vec) raw.resize(per_vec, 0);
                            out.append(raw.data(), per_vec);

                            supplier::fetch_vector_raw(m, idp, raw, dim, got_dt, got_elem);
                            if (raw.size() < per_vec) raw.resize(per_vec, 0);
                            out.append(raw.data(), per_vec);

                            supplier::fetch_vector_raw(m, idn, raw, dim, got_dt, got_elem);
                            if (raw.size() < per_vec) raw.resize(per_vec, 0);
                            out.append(raw.data(), per_vec);
                        }
                        remaining -= this_batch;
                    }
                    return out;
                }
                else
                {
                    // previous single-response behavior
                    size_t total_bytes = limit * 3 * per_vec;
                    std::string out = supplier::make_header("OK BINARY", dtype_str, limit, dim, total_bytes);

                    std::mt19937 rng(std::random_device{}());
                    for (size_t i = 0; i < limit; ++i)
                    {
                        const auto &c = cls[std::uniform_int_distribution<size_t>(0, cls.size() - 1)(rng)];
                        const auto &ids = groups[c];
                        uint64_t ida = ids[std::uniform_int_distribution<size_t>(0, ids.size() - 1)(rng)];
                        uint64_t idp = ida;
                        int tries = 0;
                        while (idp == ida && ++tries < 20)
                            idp = ids[std::uniform_int_distribution<size_t>(0, ids.size() - 1)(rng)];
                        uint64_t idn = ida;
                        while (idn == ida)
                        {
                            const auto &c2 = cls[std::uniform_int_distribution<size_t>(0, cls.size() - 1)(rng)];
                            if (c2 == c)
                                continue;
                            const auto &ids2 = groups[c2];
                            idn = ids2[std::uniform_int_distribution<size_t>(0, ids2.size() - 1)(rng)];
                        }

                        // fetch raw bytes for ida,idp,idn and append
                        pomai::core::DataType got_dt;
                        uint32_t got_elem;
                        std::string raw;
                        supplier::fetch_vector_raw(m, ida, raw, dim, got_dt, got_elem);
                        if (raw.size() < per_vec) raw.resize(per_vec, 0);
                        out.append(raw.data(), per_vec);

                        supplier::fetch_vector_raw(m, idp, raw, dim, got_dt, got_elem);
                        if (raw.size() < per_vec) raw.resize(per_vec, 0);
                        out.append(raw.data(), per_vec);

                        supplier::fetch_vector_raw(m, idn, raw, dim, got_dt, got_elem);
                        if (raw.size() < per_vec) raw.resize(per_vec, 0);
                        out.append(raw.data(), per_vec);
                    }
                    return out;
                }
            }

            // MODE: PAIR -> emit (uint64 label) + raw vector (element size = membrance elem)
            if (mode == "PAIR")
            {
                if (!idxs)
                    return "ERR: No split indices available for PAIR\n";
                if (off >= idxs->size())
                    return "OK BINARY_PAIR float32 0 " + std::to_string(dim) + " 0\n";

                std::string dtype_str;
                uint32_t elem_size;
                get_membrance_dtype(dtype_str, elem_size);
                size_t per_vec = static_cast<size_t>(elem_size) * dim;

                size_t cnt = std::min(lim, idxs->size() - off);
                size_t per = sizeof(uint64_t) + per_vec;

                // Batching support: stream chunks of batch_size_param pairs
                if (batch_size_param > 0)
                {
                    std::string out;
                    size_t processed = 0;
                    while (processed < cnt)
                    {
                        size_t this_batch = std::min<size_t>(batch_size_param, cnt - processed);
                        std::string header = "OK BINARY_PAIR " + dtype_str + " " + std::to_string(this_batch) + " " + std::to_string(dim) + " " + std::to_string(this_batch * per) + "\n";
                        out += header;
                        for (size_t i = 0; i < this_batch; ++i)
                        {
                            uint64_t id = (*idxs)[off + processed + i];
                            out.append(reinterpret_cast<const char *>(&id), sizeof(id));
                            pomai::core::DataType got_dt;
                            uint32_t got_elem;
                            std::string raw;
                            supplier::fetch_vector_raw(m, id, raw, dim, got_dt, got_elem);
                            if (raw.size() < per_vec)
                                raw.resize(per_vec, 0);
                            out.append(raw.data(), per_vec);
                        }
                        processed += this_batch;
                    }
                    return out;
                }
                else
                {
                    // single response (existing behavior)
                    size_t cnt_bytes = cnt * per;
                    std::string header = "OK BINARY_PAIR " + dtype_str + " " + std::to_string(cnt) + " " + std::to_string(dim) + " " + std::to_string(cnt_bytes) + "\n";
                    std::string out;
                    out.reserve(header.size() + cnt * per);
                    out += header;

                    for (size_t i = 0; i < cnt; ++i)
                    {
                        uint64_t id = (*idxs)[off + i];
                        out.append(reinterpret_cast<const char *>(&id), sizeof(id));

                        pomai::core::DataType got_dt;
                        uint32_t got_elem;
                        std::string raw;
                        supplier::fetch_vector_raw(m, id, raw, dim, got_dt, got_elem);
                        if (raw.size() < per_vec)
                            raw.resize(per_vec, 0);
                        out.append(raw.data(), per_vec);
                    }
                    return out;
                }
            }

            // MODE: TRAIN/VAL/TEST -> emit vectors only (raw storage bytes)
            if (mode == "TRAIN" || mode == "VAL" || mode == "TEST")
            {
                if (!idxs)
                    return "ERR: No split indices available for ITERATE\n";
                if (off >= idxs->size())
                    return "OK BINARY float32 0 " + std::to_string(dim) + " 0\n";

                std::string dtype_str;
                uint32_t elem_size;
                get_membrance_dtype(dtype_str, elem_size);
                size_t per_vec = static_cast<size_t>(elem_size) * dim;

                size_t cnt = std::min(lim, idxs->size() - off);

                // Batching support: stream sequential chunks
                if (batch_size_param > 0)
                {
                    std::string out;
                    size_t processed = 0;
                    while (processed < cnt)
                    {
                        size_t this_batch = std::min<size_t>(batch_size_param, cnt - processed);
                        size_t total_bytes = this_batch * per_vec;
                        out += supplier::make_header("OK BINARY", dtype_str, this_batch, dim, total_bytes);
                        for (size_t i = 0; i < this_batch; ++i)
                        {
                            uint64_t id = (*idxs)[off + processed + i];
                            pomai::core::DataType got_dt;
                            uint32_t got_elem;
                            std::string raw;
                            supplier::fetch_vector_raw(m, id, raw, dim, got_dt, got_elem);
                            if (raw.size() < per_vec)
                                raw.resize(per_vec, 0);
                            out.append(raw.data(), per_vec);
                        }
                        processed += this_batch;
                    }
                    return out;
                }
                else
                {
                    // single response (existing behavior)
                    size_t total_bytes = cnt * per_vec;
                    std::string header = supplier::make_header("OK BINARY", dtype_str, cnt, dim, total_bytes);
                    std::string out;
                    out.reserve(header.size() + std::min<size_t>(total_bytes, 1 << 20));
                    out += header;

                    for (size_t i = 0; i < cnt; ++i)
                    {
                        uint64_t id = (*idxs)[off + i];
                        pomai::core::DataType got_dt;
                        uint32_t got_elem;
                        std::string raw;
                        supplier::fetch_vector_raw(m, id, raw, dim, got_dt, got_elem);
                        if (raw.size() < per_vec)
                            raw.resize(per_vec, 0);
                        out.append(raw.data(), per_vec);
                    }
                    return out;
                }
            }

            return "ERR: Invalid ITERATE mode (TRAIN/VAL/TEST/TRIPLET/PAIR)\n";
        }

        // CREATE MEMBRANCE ...
        if (up.rfind("CREATE MEMBRANCE", 0) == 0)
        {
            // Expected syntax:
            // CREATE MEMBRANCE <name> DIM <n> DATA_TYPE <float32|float64|int32> RAM <mb>
            // DATA_TYPE is optional (default float32). RAM is optional (default 256).
            size_t pos_dim = up.find(" DIM ");
            if (pos_dim == std::string::npos)
                return "ERR: CREATE MEMBRANCE missing DIM\n";

            std::string name = utils::trim(cmd.substr(std::string("CREATE MEMBRANCE").size(), pos_dim - std::string("CREATE MEMBRANCE").size()));
            std::string tail = utils::trim(cmd.substr(pos_dim + 5)); // after "DIM "

            std::istringstream iss(tail);
            size_t dim = 0;
            std::string token;
            size_t ram_mb = 256;
            std::string data_type = "float32"; // default

            // First token must be dim number
            if (!(iss >> token))
                return "ERR: invalid DIM\n";
            try
            {
                dim = static_cast<size_t>(std::stoul(token));
            }
            catch (...)
            {
                return "ERR: invalid DIM\n";
            }

            // Parse remaining tokens in flexible order: DATA_TYPE <val>, RAM <mb>
            while (iss >> token)
            {
                std::string up_tok = utils::to_upper(token);
                if (up_tok == "DATA_TYPE" || up_tok == "DATA-TYPE")
                {
                    std::string dt;
                    if (!(iss >> dt))
                        return "ERR: DATA_TYPE requires a value (float32|float64|int32)\n";
                    // normalize to lowercase
                    std::transform(dt.begin(), dt.end(), dt.begin(), ::tolower);
                    if (dt != "float32" && dt != "float64" && dt != "int32" && dt != "int8" && dt != "float16")
                        return std::string("ERR: unsupported DATA_TYPE '") + dt + "' (supported: float32, float64, int32, int8, float16)\n";
                    data_type = dt;
                }
                else if (up_tok == "RAM")
                {
                    std::string r;
                    if (!(iss >> r))
                        return "ERR: RAM requires a numeric value\n";
                    try
                    {
                        ram_mb = static_cast<size_t>(std::stoul(r));
                    }
                    catch (...)
                    {
                        return "ERR: invalid RAM value\n";
                    }
                }
                else
                {
                    // Unknown token: skip or treat as error. We'll skip unknown tokens to be lenient.
                    // Consume one argument if looks like a value (already consumed token), continue.
                    continue;
                }
            }

            if (dim == 0)
                return "ERR: invalid DIM\n";

            // Build config and pass data_type through
            pomai::core::MembranceConfig cfg;
            cfg.dim = dim;
            cfg.ram_mb = ram_mb;
            // convert textual data_type into enum
            try
            {
                cfg.data_type = pomai::core::parse_dtype(data_type);
            }
            catch (...)
            {
                return std::string("ERR: unsupported DATA_TYPE '") + data_type + "'\n";
            }

            bool ok = db->create_membrance(name, cfg);
            if (ok)
            {
                std::ostringstream ss;
                ss << "OK: created membrance " << name << " dim=" << dim << " data_type=" << data_type << " ram=" << ram_mb << "MB\n";
                return ss.str();
            }
            return "ERR: create failed (exists or invalid)\n";
        }

        // DROP MEMBRANCE
        if (up.rfind("DROP MEMBRANCE", 0) == 0)
        {
            auto parts = utils::split_ws(cmd);
            if (parts.size() < 3)
                return "ERR: DROP MEMBRANCE <name>;\n";
            std::string name = parts[2];
            if (!name.empty() && name.back() == ';')
                name.pop_back();
            bool ok = db->drop_membrance(name);
            return ok ? std::string("OK: dropped ") + name + "\n" : std::string("ERR: drop failed\n");
        }

        // GET MEMBRANCE INFO ...
        if (up.rfind("GET MEMBRANCE", 0) == 0)
        {
            auto parts = utils::split_ws(cmd);
            if (!parts.empty() && !parts.back().empty() && parts.back().back() == ';')
            {
                parts.back() = parts.back().substr(0, parts.back().size() - 1);
            }

            std::string name;
            bool ok_parse = false;

            if (parts.size() >= 4 && utils::to_upper(parts[2]) == "INFO")
            {
                name = parts[3];
                ok_parse = true;
            }
            else if (parts.size() >= 4 && utils::to_upper(parts[3]) == "INFO")
            {
                name = parts[2];
                ok_parse = true;
            }
            else if (parts.size() >= 3 && utils::to_upper(parts[2]) == "INFO")
            {
                if (state.current_membrance.empty())
                    return "ERR: no current membrance (USE <name>)\n";
                name = state.current_membrance;
                ok_parse = true;
            }

            if (!ok_parse)
                return "ERR: expected 'GET MEMBRANCE INFO ...'\n";

            auto *m = db->get_membrance(name);
            if (!m)
                return std::string("ERR: membrance not found: ") + name + "\n";

            // 1. Gather Storage Info (Physical)
            pomai::ai::orbit::MembranceInfo info;
            try
            {
                info = m->orbit->get_info();
            }
            catch (...)
            {
                info.dim = m->dim;
                info.num_vectors = 0;
                info.disk_bytes = 0;
            }

            if (info.disk_bytes == 0)
            {
                try
                {
                    std::filesystem::path dp(m->data_path);
                    if (std::filesystem::exists(dp))
                    {
                        for (auto const &entry : std::filesystem::recursive_directory_iterator(dp))
                        {
                            if (!entry.is_regular_file())
                                continue;
                            std::error_code ec;
                            uint64_t fsz = static_cast<uint64_t>(entry.file_size(ec));
                            if (!ec)
                                info.disk_bytes += fsz;
                        }
                    }
                }
                catch (...)
                {
                }
            }

            size_t n_train = 0, n_val = 0, n_test = 0;
            if (m->split_mgr)
            {
                n_train = m->split_mgr->train_indices.size();
                n_val = m->split_mgr->val_indices.size();
                n_test = m->split_mgr->test_indices.size();
            }

            size_t feature_dim = (m->dim > 0) ? m->dim : info.dim;

            size_t total_vectors = info.num_vectors;
            if (total_vectors == 0)
            {
                total_vectors = n_train + n_val + n_test;
            }

            std::ostringstream ss;
            ss << "MEMBRANCE: " << name << "\n";
            ss << "--- AI Contract ---\n";
            ss << " feature_dim: " << feature_dim << "\n";
            ss << " metric: L2\n";
            // Report actual data_type configured for this membrance
            try
            {
                ss << " data_type: " << pomai::core::dtype_name(m->data_type) << "\n";
            }
            catch (...)
            {
                ss << " data_type: float32\n";
            }
            ss << " total_vectors: " << total_vectors << "\n";
            ss << " split_train: " << n_train << "\n";
            ss << " split_val: " << n_val << "\n";
            ss << " split_test: " << n_test << "\n";
            ss << "--- Storage Stats ---\n";
            ss << " disk_bytes: " << utils::bytes_human(info.disk_bytes) << "\n";
            ss << " ram_mb_configured: " << m->ram_mb << "\n";

            return ss.str();
        }

        // INSERT INTO ... VALUES ...
        {
            std::string upstart = utils::to_upper(cmd.substr(0, std::min<size_t>(cmd.size(), 16)));
            if (upstart.rfind("INSERT INTO", 0) == 0 || utils::to_upper(cmd).rfind("INSERT VALUES", 0) == 0)
            {
                std::string body = cmd;
                std::string name;
                bool has_explicit_into = (utils::to_upper(cmd).find("INTO") != std::string::npos);
                if (!has_explicit_into && utils::to_upper(cmd).rfind("INSERT VALUES", 0) == 0)
                {
                    if (state.current_membrance.empty())
                        return "ERR: no current membrance (USE <name>)\n";
                    name = state.current_membrance;
                    body = std::string("INSERT INTO ") + name + " " + cmd.substr(std::string("INSERT VALUES").size());
                }
                size_t pos_into = utils::to_upper(body).find("INTO");
                size_t pos_values = utils::to_upper(body).find("VALUES");
                if (pos_values == std::string::npos)
                    return "ERR: INSERT missing VALUES\n";
                if (pos_into == std::string::npos)
                    return "ERR: INSERT missing INTO\n";
                size_t name_start = pos_into + 4;
                name = utils::trim(body.substr(name_start, pos_values - name_start));
                if (name.empty())
                    return "ERR: missing membrance name\n";

                auto *m = db->get_membrance(name);
                if (!m)
                    return "ERR: membrance not found\n";

                size_t pos_after_values = pos_values + std::string("VALUES").size();
                size_t cur = body.find_first_not_of(" \t\r\n", pos_after_values);
                if (cur == std::string::npos || body[cur] != '(')
                    return "ERR: VALUES syntax\n";

                std::vector<std::pair<std::string, std::vector<float>>> tuples;
                tuples.reserve(8);

                while (true)
                {
                    if (cur >= body.size() || body[cur] != '(')
                        break;
                    size_t depth = 0;
                    size_t start = cur;
                    size_t end = std::string::npos;
                    for (size_t p = cur; p < body.size(); ++p)
                    {
                        if (body[p] == '(')
                            ++depth;
                        else if (body[p] == ')')
                        {
                            --depth;
                            if (depth == 0)
                            {
                                end = p;
                                cur = p + 1;
                                break;
                            }
                        }
                    }
                    if (end == std::string::npos)
                        return "ERR: unmatched parentheses in VALUES\n";

                    std::string tuple_text = body.substr(start + 1, end - start - 1);
                    size_t vec_lb = tuple_text.find('[');
                    size_t vec_rb = tuple_text.rfind(']');
                    if (vec_lb == std::string::npos || vec_rb == std::string::npos || vec_rb <= vec_lb)
                        return "ERR: tuple vector syntax\n";
                    size_t comma_before_vec = tuple_text.rfind(',', vec_lb);
                    if (comma_before_vec == std::string::npos)
                        return "ERR: tuple syntax (label, [vec])\n";
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
                    if (cur == std::string::npos)
                        break;
                    if (body[cur] == ',')
                    {
                        ++cur;
                        cur = body.find_first_not_of(" \t\r\n", cur);
                        continue;
                    }
                    break;
                }

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
                if (!global_tags.empty())
                    inserted_hashes.reserve(tuples.size());

                for (auto &tp : tuples)
                {
                    uint64_t label_hash = utils::hash_key(tp.first);
                    batch_data.emplace_back(label_hash, std::move(tp.second));
                    if (!global_tags.empty())
                        inserted_hashes.push_back(label_hash);
                }

                bool ok_batch = false;
                try
                {
                    ok_batch = db->insert_batch(name, batch_data);
                }
                catch (...)
                {
                    ok_batch = false;
                }

                if (ok_batch && !global_tags.empty() && m->meta_index)
                {
                    for (uint64_t h : inserted_hashes)
                    {
                        try
                        {
                            m->meta_index->add_tags(h, global_tags);
                        }
                        catch (...)
                        {
                            std::clog << "[SqlExecutor] Warning: failed to add tags for label " << h << "\n";
                        }
                    }
                }

                std::ostringstream ss;
                ss << "OK: inserted " << (ok_batch ? batch_data.size() : 0) << " / " << tuples.size() << (ok_batch ? " (batch)" : " (failed)") << "\n";
                return ss.str();
            }
        }

        // SEARCH ...
        if (utils::to_upper(cmd).rfind("SEARCH ", 0) == 0)
        {
            size_t pos_q = utils::to_upper(cmd).find(" QUERY ");
            std::string name;
            size_t vec_lb = std::string::npos;
            size_t vec_rb = std::string::npos;
            if (pos_q != std::string::npos)
            {
                name = utils::trim(cmd.substr(7, pos_q - 7));
                vec_lb = cmd.find('[', pos_q);
                vec_rb = cmd.find(']', vec_lb);
            }
            else
            {
                size_t pos_q2 = utils::to_upper(cmd).find("SEARCH QUERY");
                if (pos_q2 == std::string::npos)
                    return "ERR: SEARCH syntax\n";
                if (state.current_membrance.empty())
                    return "ERR: no current membrance (USE <name>)\n";
                name = state.current_membrance;
                vec_lb = cmd.find('[', pos_q2);
                vec_rb = cmd.find(']', vec_lb);
            }
            if (vec_lb == std::string::npos || vec_rb == std::string::npos)
                return "ERR: SEARCH missing vector\n";
            std::string veccsv = cmd.substr(vec_lb + 1, vec_rb - vec_lb - 1);
            auto vec = utils::parse_float_list_sv(utils::trim_sv(std::string_view(veccsv)));
            size_t pos_top = utils::to_upper(cmd).find(" TOP ", vec_rb);
            int topk = 10;
            if (pos_top != std::string::npos)
            {
                size_t start = pos_top + 5;
                size_t end = cmd.find(';', start);
                std::string kn = (end == std::string::npos) ? utils::trim(cmd.substr(start)) : utils::trim(cmd.substr(start, end - start));
                try
                {
                    topk = std::stoi(kn);
                }
                catch (...)
                {
                    topk = 10;
                }
            }

            auto *m = db->get_membrance(name);
            if (!m)
                return "ERR: membrance not found\n";
            if (vec.size() != m->dim)
            {
                std::ostringstream ss;
                ss << "ERR: dim mismatch expected=" << m->dim << " got=" << vec.size() << "\n";
                return ss.str();
            }

            std::string hot_key_for_freq;
            std::vector<std::pair<uint64_t, float>> res;

            std::string upcmd = utils::to_upper(cmd);
            size_t pos_where = upcmd.find(" WHERE ", vec_rb);

            if (pos_where != std::string::npos && m->meta_index)
            {
                size_t cond_start = pos_where + 7;
                size_t cond_end = std::string::npos;
                size_t pos_top_after = upcmd.find(" TOP ", cond_start);
                if (pos_top_after != std::string::npos)
                    cond_end = pos_top_after;
                else
                {
                    size_t semi = cmd.find(';', cond_start);
                    cond_end = (semi == std::string::npos) ? cmd.size() : semi;
                }
                std::string cond = utils::trim(cmd.substr(cond_start, cond_end - cond_start));
                size_t eq = cond.find('=');
                if (eq != std::string::npos)
                {
                    std::string key = utils::trim(cond.substr(0, eq));
                    std::string val = utils::trim(cond.substr(eq + 1));
                    if (val.size() >= 2 && ((val.front() == '\'' && val.back() == '\'') || (val.front() == '\"' && val.back() == '\"')))
                        val = val.substr(1, val.size() - 2);

                    std::vector<uint64_t> candidates = m->meta_index->filter(key, val);

                    hot_key_for_freq = key + "=" + val;

                    bool is_hot = false;
                    {
                        std::lock_guard<std::mutex> qlk(freq_mu_);
                        uint32_t &cnt = query_freq_[hot_key_for_freq];
                        cnt++;
                        if (cnt >= 5)
                            is_hot = true;
                    }

                    auto budget = whisper.compute_budget(is_hot);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    try
                    {
                        res = m->orbit->search_filtered_with_budget(vec.data(), static_cast<size_t>(topk), candidates, budget);
                    }
                    catch (...)
                    {
                        res = m->orbit->search(vec.data(), static_cast<size_t>(topk));
                    }
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    whisper.observe_latency(static_cast<float>(ms));
                }
                else
                {
                    auto budget = whisper.compute_budget(false);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    res = m->orbit->search_with_budget(vec.data(), static_cast<size_t>(topk), budget);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    whisper.observe_latency(static_cast<float>(ms));
                }
            }
            else
            {
                hot_key_for_freq = name;
                bool is_hot = false;
                {
                    std::lock_guard<std::mutex> qlk(freq_mu_);
                    uint32_t &cnt = query_freq_[hot_key_for_freq];
                    cnt++;
                    if (cnt >= 5)
                        is_hot = true;
                }
                auto budget = whisper.compute_budget(is_hot);
                auto t0 = std::chrono::high_resolution_clock::now();
                try
                {
                    res = m->orbit->search_with_budget(vec.data(), static_cast<size_t>(topk), budget);
                }
                catch (...)
                {
                    res = m->orbit->search(vec.data(), static_cast<size_t>(topk));
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                whisper.observe_latency(static_cast<float>(ms));
            }

            std::ostringstream ss;
            ss << "RESULTS " << res.size() << "\n";
            for (auto &p : res)
                ss << p.first << " " << p.second << "\n";
            return ss.str();
        }

        // GET <name> LABEL <label>;
        if (utils::to_upper(cmd).rfind("GET ", 0) == 0)
        {
            auto parts = utils::split_ws(cmd);
            if (parts.size() < 4)
                return "ERR: GET <name> LABEL <label>\n";
            std::string name = parts[1];
            std::string label = parts[3];
            if (!label.empty() && label.back() == ';')
                label.pop_back();
            auto *m = db->get_membrance(name);
            if (!m)
                return "ERR: membrance not found\n";
            std::vector<float> out;
            bool ok = m->orbit->get(utils::hash_key(label), out);
            if (!ok)
                return "ERR: not found\n";
            std::ostringstream ss;
            ss << "VECTOR " << out.size() << " ";
            for (float v : out)
                ss << v << " ";
            ss << "\n";
            return ss.str();
        }

        // DELETE <name> LABEL <label>;
        if (utils::to_upper(cmd).rfind("DELETE ", 0) == 0 || utils::to_upper(cmd).rfind("VDEL ", 0) == 0)
        {
            auto parts = utils::split_ws(cmd);
            if (parts.size() < 4)
                return "ERR: DELETE <name> LABEL <label>\n";
            std::string name = parts[1];
            std::string label = parts[3];
            if (!label.empty() && label.back() == ';')
                label.pop_back();
            auto *m = db->get_membrance(name);
            if (!m)
                return "ERR: membrance not found\n";
            bool ok = m->orbit->remove(utils::hash_key(label));
            return ok ? std::string("OK\n") : std::string("ERR: remove failed\n");
        }

        return "ERR: unknown command\n";
    }

} // namespace pomai::server