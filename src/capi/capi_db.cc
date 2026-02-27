#include "pomai/c_api.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "capi_utils.h"
#include "core/memory/pin_manager.h"
#include "pomai/version.h"

namespace {

constexpr const char* kDefaultMembrane = "__default__";

struct RecordWrapper {
    pomai_record_t pub{};
    std::vector<float> vec_data;
    std::vector<uint8_t> meta_data;
};

struct SearchResultsWrapper {
    pomai_search_results_t pub{};
    std::vector<uint64_t> ids;
    std::vector<float> scores;
    std::vector<uint32_t> shard_ids;
};

constexpr uint32_t MinOptionsStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_options_t, hnsw_ef_search) + sizeof(uint32_t));
}

constexpr uint32_t MinUpsertStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_upsert_t, metadata_len) + sizeof(uint32_t));
}

constexpr uint32_t MinQueryStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_query_t, alpha) + sizeof(float));
}


bool DeadlineExceeded(uint32_t deadline_ms) {
    if (deadline_ms == 0) {
        return false;
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return now_ms.count() >= deadline_ms;
}

bool ParseTenantFilter(const char* expr, pomai::SearchOptions* out_opts) {
    if (expr == nullptr || *expr == '\0') {
        return true;
    }

    std::string s(expr);
    const auto eq = s.find('=');
    if (eq == std::string::npos) {
        return false;
    }
    auto field = s.substr(0, eq);
    auto value = s.substr(eq + 1);
    auto trim = [](std::string* v) {
        while (!v->empty() && std::isspace(static_cast<unsigned char>(v->front()))) v->erase(v->begin());
        while (!v->empty() && std::isspace(static_cast<unsigned char>(v->back()))) v->pop_back();
    };
    trim(&field);
    trim(&value);
    if (field != "tenant") {
        return false;
    }
    out_opts->filters.push_back(pomai::Filter(field, value));
    return true;
}

pomai::Metadata ToMetadata(const pomai_upsert_t& item) {
    if (item.metadata == nullptr || item.metadata_len == 0) {
        return pomai::Metadata();
    }
    return pomai::Metadata(std::string(reinterpret_cast<const char*>(item.metadata), item.metadata_len));
}

}  // namespace

extern "C" {

const char* pomai_version_string(void) {
    static const std::string kVersion =
        std::to_string(POMAI_VERSION_MAJOR) + "." + std::to_string(POMAI_VERSION_MINOR) + "." + std::to_string(POMAI_VERSION_PATCH);
    return kVersion.c_str();
}

uint32_t pomai_abi_version(void) {
    return POMAI_ABI_VERSION;
}

void pomai_options_init(pomai_options_t* opts) {
    if (opts == nullptr) {
        return;
    }
    opts->struct_size = static_cast<uint32_t>(sizeof(pomai_options_t));
    opts->path = nullptr;
    opts->shards = 4;
    opts->dim = 512;
    opts->search_threads = 0;
    opts->fsync_policy = POMAI_FSYNC_POLICY_NEVER;
    opts->memory_budget_bytes = 0;
    opts->deadline_ms = 0;
    opts->index_type = 0; // IVF
    opts->hnsw_m = 32;
    opts->hnsw_ef_construction = 200;
    opts->hnsw_ef_search = 64;
    opts->adaptive_threshold = 5000;
    opts->metric = 0; // L2
}

void pomai_scan_options_init(pomai_scan_options_t* opts) {
    if (opts == nullptr) {
        return;
    }
    opts->struct_size = static_cast<uint32_t>(sizeof(pomai_scan_options_t));
    opts->start_id = 0;
    opts->has_start_id = false;
    opts->deadline_ms = 0;
}

pomai_status_t* pomai_open(const pomai_options_t* opts, pomai_db_t** out_db) {
    if (opts == nullptr || out_db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts/out_db must be non-null");
    }
    if (opts->struct_size < MinOptionsStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "options.struct_size is too small");
    }
    if (DeadlineExceeded(opts->deadline_ms)) {
        return MakeStatus(POMAI_STATUS_DEADLINE_EXCEEDED, "deadline exceeded before open");
    }
    if (opts->path == nullptr || opts->path[0] == '\0') {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "options.path must be non-empty");
    }

    pomai::DBOptions cpp_opts;
    cpp_opts.path = opts->path;
    cpp_opts.shard_count = opts->shards;
    cpp_opts.dim = opts->dim;
    cpp_opts.search_threads = opts->search_threads;
    cpp_opts.fsync = (opts->fsync_policy == POMAI_FSYNC_POLICY_ALWAYS)
                         ? pomai::FsyncPolicy::kAlways
                         : pomai::FsyncPolicy::kNever;
    cpp_opts.metric = (opts->metric == 1) ? pomai::MetricType::kInnerProduct : pomai::MetricType::kL2;
    cpp_opts.index_params.adaptive_threshold = opts->adaptive_threshold;
    
    // Default to IVF, overwrite if HNSW.
    if (opts->index_type == 1) {
        cpp_opts.index_params.type = pomai::IndexType::kHnsw;
        cpp_opts.index_params.hnsw_m = opts->hnsw_m;
        cpp_opts.index_params.hnsw_ef_construction = opts->hnsw_ef_construction;
        cpp_opts.index_params.hnsw_ef_search = opts->hnsw_ef_search;
    } else {
        cpp_opts.index_params.type = pomai::IndexType::kIvfFlat;
    }

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(cpp_opts, &db);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    *out_db = new pomai_db_t{std::move(db)};
    return nullptr;
}

pomai_status_t* pomai_close(pomai_db_t* db) {
    if (db == nullptr) {
        return nullptr;
    }
    auto st = db->db->Close();
    delete db;
    return ToCStatus(st);
}

pomai_status_t* pomai_put(pomai_db_t* db, const pomai_upsert_t* item) {
    if (db == nullptr || item == nullptr || item->vector == nullptr || item->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid put arguments");
    }
    if (item->struct_size < MinUpsertStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "upsert.struct_size is too small");
    }
    std::span<const float> vec(item->vector, item->dim);
    return ToCStatus(db->db->Put(item->id, vec, ToMetadata(*item)));
}

pomai_status_t* pomai_put_batch(pomai_db_t* db, const pomai_upsert_t* items, size_t n) {
    if (db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null");
    }
    if (n == 0) {
        return nullptr;
    }
    if (items == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "items must be non-null");
    }

    std::vector<pomai::VectorId> ids;
    std::vector<std::span<const float>> vecs;
    ids.reserve(n);
    vecs.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (items[i].struct_size < MinUpsertStructSize()) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "all batch items require valid struct_size");
        }
        if (items[i].vector == nullptr || items[i].dim == 0) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "all batch items require vector and dim");
        }
        ids.push_back(items[i].id);
        vecs.emplace_back(items[i].vector, items[i].dim);
    }
    return ToCStatus(db->db->PutBatch(ids, vecs));
}

pomai_status_t* pomai_delete(pomai_db_t* db, uint64_t id) {
    if (db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null");
    }
    return ToCStatus(db->db->Delete(id));
}

pomai_status_t* pomai_freeze(pomai_db_t* db) {
    if (db == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db must be non-null");
    }
    return ToCStatus(db->db->Freeze(kDefaultMembrane));
}

pomai_status_t* pomai_get(pomai_db_t* db, uint64_t id, pomai_record_t** out_record) {
    if (db == nullptr || out_record == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/out_record must be non-null");
    }

    std::vector<float> vec;
    pomai::Metadata meta;
    auto st = db->db->Get(id, &vec, &meta);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    auto* w = new RecordWrapper();
    w->vec_data = std::move(vec);
    w->meta_data.assign(meta.tenant.begin(), meta.tenant.end());

    w->pub.struct_size = static_cast<uint32_t>(sizeof(pomai_record_t));
    w->pub.id = id;
    w->pub.dim = static_cast<uint32_t>(w->vec_data.size());
    w->pub.vector = w->vec_data.data();
    w->pub.metadata = w->meta_data.empty() ? nullptr : w->meta_data.data();
    w->pub.metadata_len = static_cast<uint32_t>(w->meta_data.size());
    w->pub.is_deleted = false;

    *out_record = &w->pub;
    return nullptr;
}

void pomai_record_free(pomai_record_t* record) {
    delete reinterpret_cast<RecordWrapper*>(record);
}

pomai_status_t* pomai_exists(pomai_db_t* db, uint64_t id, bool* out_exists) {
    if (db == nullptr || out_exists == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/out_exists must be non-null");
    }
    return ToCStatus(db->db->Exists(id, out_exists));
}

pomai_status_t* pomai_search(pomai_db_t* db, const pomai_query_t* query, pomai_search_results_t** out) {
    if (db == nullptr || query == nullptr || out == nullptr || query->vector == nullptr || query->dim == 0 || query->topk == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid search args");
    }
    if (query->struct_size < MinQueryStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "query.struct_size is too small");
    }
    if (DeadlineExceeded(query->deadline_ms)) {
        return MakeStatus(POMAI_STATUS_DEADLINE_EXCEEDED, "deadline exceeded before search");
    }

    pomai::SearchResult res;
    pomai::SearchOptions opts;
    if (!ParseTenantFilter(query->filter_expression, &opts)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "filter_expression must use tenant=<value>");
    }
    if (query->flags & POMAI_QUERY_FLAG_ZERO_COPY) {
        opts.zero_copy = true;
    }

    auto st = db->db->Search(std::span<const float>(query->vector, query->dim), query->topk, opts, &res);
    if (!st.ok() && st.code() != pomai::ErrorCode::kPartial) {
        return ToCStatus(st);
    }

    if (DeadlineExceeded(query->deadline_ms)) {
        return MakeStatus(POMAI_STATUS_DEADLINE_EXCEEDED, "deadline exceeded after search");
    }

    auto* w = new SearchResultsWrapper();
    w->ids.reserve(res.hits.size());
    w->scores.reserve(res.hits.size());
    w->shard_ids.reserve(res.hits.size());
    for (const auto& hit : res.hits) {
        w->ids.push_back(hit.id);
        w->scores.push_back(hit.score);
        w->shard_ids.push_back(UINT32_MAX);
    }

    w->pub.struct_size = static_cast<uint32_t>(sizeof(pomai_search_results_t));
    w->pub.count = w->ids.size();
    w->pub.ids = w->ids.data();
    w->pub.scores = w->scores.data();
    w->pub.shard_ids = w->shard_ids.data();
    if (opts.zero_copy && !res.zero_copy_pointers.empty()) {
        w->pub.zero_copy_pointers = new pomai_semantic_pointer_t[res.zero_copy_pointers.size()];
        for (size_t i = 0; i < res.zero_copy_pointers.size(); ++i) {
            w->pub.zero_copy_pointers[i].struct_size = sizeof(pomai_semantic_pointer_t);
            w->pub.zero_copy_pointers[i].raw_data_ptr = res.zero_copy_pointers[i].raw_data_ptr;
            w->pub.zero_copy_pointers[i].dim = res.zero_copy_pointers[i].dim;
            w->pub.zero_copy_pointers[i].quant_min = res.zero_copy_pointers[i].quant_min;
            w->pub.zero_copy_pointers[i].quant_inv_scale = res.zero_copy_pointers[i].quant_inv_scale;
            w->pub.zero_copy_pointers[i].session_id = res.zero_copy_pointers[i].session_id;
        }
    } else {
        w->pub.zero_copy_pointers = nullptr;
    }
    *out = &w->pub;

    if (st.code() == pomai::ErrorCode::kPartial) {
        return MakeStatus(POMAI_STATUS_PARTIAL_FAILURE, st.message());
    }
    if (!res.errors.empty()) {
        return MakeStatus(POMAI_STATUS_PARTIAL_FAILURE, "partial shard failures");
    }
    return nullptr;
}

pomai_status_t* pomai_search_batch(pomai_db_t* db, const pomai_query_t* queries, size_t num_queries, pomai_search_results_t** out) {
    if (db == nullptr || queries == nullptr || out == nullptr || num_queries == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid batch search args");
    }
    if (queries[0].struct_size < MinQueryStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "query.struct_size is too small");
    }
    
    // We assume all queries in the batch have the same dimensions and options.
    const uint32_t dim = queries[0].dim;
    const uint32_t topk = queries[0].topk;
    
    pomai::SearchOptions opts;
    if (!ParseTenantFilter(queries[0].filter_expression, &opts)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "filter_expression must use tenant=<value>");
    }
    if (queries[0].flags & POMAI_QUERY_FLAG_ZERO_COPY) {
        opts.zero_copy = true;
    }

    std::vector<float> flat_queries;
    flat_queries.reserve(num_queries * dim);
    for (size_t i = 0; i < num_queries; ++i) {
        if (queries[i].vector == nullptr || queries[i].dim != dim || queries[i].topk != topk) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "batch queries must have identical dim and topk");
        }
        flat_queries.insert(flat_queries.end(), queries[i].vector, queries[i].vector + dim);
    }

    std::vector<pomai::SearchResult> batch_res;
    auto st = db->db->SearchBatch(std::span<const float>(flat_queries.data(), flat_queries.size()), num_queries, topk, opts, &batch_res);
    
    if (!st.ok() && st.code() != pomai::ErrorCode::kPartial) {
        return ToCStatus(st);
    }

    // Allocate array of results
    *out = new pomai_search_results_t[num_queries];
    
    for (size_t q = 0; q < num_queries; ++q) {
        const auto& res = batch_res[q];
        pomai_search_results_t& pub = (*out)[q];

        pub.struct_size = static_cast<uint32_t>(sizeof(pomai_search_results_t));
        pub.count = res.hits.size();
        
        pub.ids = new uint64_t[pub.count];
        pub.scores = new float[pub.count];
        pub.shard_ids = new uint32_t[pub.count];
        
        for (size_t i = 0; i < pub.count; ++i) {
            pub.ids[i] = res.hits[i].id;
            pub.scores[i] = res.hits[i].score;
            pub.shard_ids[i] = UINT32_MAX;
        }

        if (opts.zero_copy && !res.zero_copy_pointers.empty()) {
            pub.zero_copy_pointers = new pomai_semantic_pointer_t[res.zero_copy_pointers.size()];
            for (size_t i = 0; i < res.zero_copy_pointers.size(); ++i) {
                pub.zero_copy_pointers[i].struct_size = sizeof(pomai_semantic_pointer_t);
                pub.zero_copy_pointers[i].raw_data_ptr = res.zero_copy_pointers[i].raw_data_ptr;
                pub.zero_copy_pointers[i].dim = res.zero_copy_pointers[i].dim;
                pub.zero_copy_pointers[i].quant_min = res.zero_copy_pointers[i].quant_min;
                pub.zero_copy_pointers[i].quant_inv_scale = res.zero_copy_pointers[i].quant_inv_scale;
                pub.zero_copy_pointers[i].session_id = res.zero_copy_pointers[i].session_id;
            }
        } else {
            pub.zero_copy_pointers = nullptr;
        }
    }

    if (st.code() == pomai::ErrorCode::kPartial) {
        return MakeStatus(POMAI_STATUS_PARTIAL_FAILURE, st.message());
    }
    return nullptr;
}

void pomai_search_results_free(pomai_search_results_t* results) {
    if (results && results->zero_copy_pointers) {
        delete[] results->zero_copy_pointers;
    }
    delete reinterpret_cast<SearchResultsWrapper*>(results);
}

void pomai_search_batch_free(pomai_search_results_t* results, size_t num_queries) {
    if (!results) return;
    for (size_t i = 0; i < num_queries; ++i) {
        if (results[i].ids) {
            delete[] results[i].ids;
        }
        if (results[i].scores) {
            delete[] results[i].scores;
        }
        if (results[i].shard_ids) {
            delete[] results[i].shard_ids;
        }
        if (results[i].zero_copy_pointers) {
            delete[] results[i].zero_copy_pointers;
        }
    }
    delete[] results;
}

void pomai_release_pointer(uint64_t session_id) {
    pomai::core::MemoryPinManager::Instance().Unpin(session_id);
}

void pomai_free(void* p) {
    std::free(p);
}

}  // extern "C"
