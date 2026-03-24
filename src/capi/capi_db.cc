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

#include "palloc_compat.h"
#include "capi_utils.h"
#include "core/memory/pin_manager.h"
#include "pomai/options.h"
#include "pomai/pomai.h"
#include "pomai/rag.h"
#include "pomai/rag/embedding_provider.h"
#include "pomai/rag/pipeline.h"
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
    return static_cast<uint32_t>(offsetof(pomai_query_t, filter_expression) + sizeof(const char*));
}

double ComputeAggregateValue(uint32_t op, const std::vector<pomai::SearchHit>& hits) {
    if (hits.empty()) return 0.0;
    if (op == 5u) return static_cast<double>(hits.size()); // count
    double sum = 0.0;
    double mn = hits[0].score;
    double mx = hits[0].score;
    for (const auto& h : hits) {
        const double v = static_cast<double>(h.score);
        sum += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    if (op == 1u) return sum;
    if (op == 2u) return sum / static_cast<double>(hits.size());
    if (op == 3u) return mn;
    if (op == 4u) return mx;
    if (op == 6u) return hits.front().score; // top1 score proxy
    return 0.0;
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
    if (field != "tenant" && field != "device_id" && field != "location_id") {
        return false;
    }
    out_opts->filters.push_back(pomai::Filter(field, value));
    return true;
}

std::string JsonEscape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        const unsigned char uc = static_cast<unsigned char>(c);
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (uc < 0x20) {
                    out += "\\u00";
                    const char* hex = "0123456789abcdef";
                    out.push_back(hex[(uc >> 4) & 0x0f]);
                    out.push_back(hex[uc & 0x0f]);
                } else {
                    out.push_back(c);
                }
                break;
        }
    }
    return out;
}

pomai::Metadata ToMetadata(const pomai_upsert_t& item) {
    if (item.metadata == nullptr || item.metadata_len == 0) {
        return pomai::Metadata();
    }
    pomai::Metadata m(std::string(reinterpret_cast<const char*>(item.metadata), item.metadata_len));
    m.device_id = m.tenant;
    m.location_id = m.tenant;
    return m;
}

}  // namespace

struct pomai_rag_pipeline_t {
    std::unique_ptr<pomai::MockEmbeddingProvider> mock_embed;
    std::unique_ptr<pomai::RagPipeline> pipeline;
};

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
    opts->edge_profile = 0; // user-defined
    opts->gateway_rate_limit_per_sec = 0;
    opts->gateway_idempotency_ttl_sec = 0;
    opts->gateway_token_file = nullptr;
    opts->gateway_upstream_sync_url = nullptr;
    opts->gateway_upstream_sync_enabled = false;
    opts->gateway_require_mtls_proxy_header = false;
    opts->gateway_mtls_proxy_header = nullptr;
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

pomai_status_t* pomai_options_resolve_json(const pomai_options_t* opts, char** out_json, size_t* out_len) {
    if (opts == nullptr || out_json == nullptr || out_len == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts/out_json/out_len must be non-null");
    }
    if (opts->struct_size < MinOptionsStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "options.struct_size is too small");
    }

    pomai::DBOptions db_opts;
    db_opts.path = (opts->path != nullptr) ? opts->path : "";
    db_opts.dim = opts->dim;
    db_opts.shard_count = opts->shards > 0 ? opts->shards : 4u;
    db_opts.fsync = (opts->fsync_policy == POMAI_FSYNC_POLICY_ALWAYS) ? pomai::FsyncPolicy::kAlways : pomai::FsyncPolicy::kNever;
    db_opts.metric = (opts->metric == 1) ? pomai::MetricType::kInnerProduct : pomai::MetricType::kL2;
    db_opts.index_params.adaptive_threshold = opts->adaptive_threshold;
    if (opts->index_type == 1) {
        db_opts.index_params.type = pomai::IndexType::kHnsw;
        db_opts.index_params.hnsw_m = opts->hnsw_m;
        db_opts.index_params.hnsw_ef_construction = opts->hnsw_ef_construction;
        db_opts.index_params.hnsw_ef_search = opts->hnsw_ef_search;
    } else {
        db_opts.index_params.type = pomai::IndexType::kIvfFlat;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, edge_profile) + sizeof(uint8_t))) {
        db_opts.edge_profile = static_cast<pomai::EdgeProfile>(opts->edge_profile);
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_rate_limit_per_sec) + sizeof(uint32_t)) &&
        opts->gateway_rate_limit_per_sec > 0) {
        db_opts.gateway_rate_limit_per_sec = opts->gateway_rate_limit_per_sec;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_idempotency_ttl_sec) + sizeof(uint32_t)) &&
        opts->gateway_idempotency_ttl_sec > 0) {
        db_opts.gateway_idempotency_ttl_sec = opts->gateway_idempotency_ttl_sec;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_token_file) + sizeof(const char*)) &&
        opts->gateway_token_file != nullptr) {
        db_opts.gateway_token_file = opts->gateway_token_file;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_upstream_sync_url) + sizeof(const char*)) &&
        opts->gateway_upstream_sync_url != nullptr) {
        db_opts.gateway_upstream_sync_url = opts->gateway_upstream_sync_url;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_upstream_sync_enabled) + sizeof(bool))) {
        db_opts.gateway_upstream_sync_enabled = opts->gateway_upstream_sync_enabled;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_require_mtls_proxy_header) + sizeof(bool))) {
        db_opts.gateway_require_mtls_proxy_header = opts->gateway_require_mtls_proxy_header;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_mtls_proxy_header) + sizeof(const char*)) &&
        opts->gateway_mtls_proxy_header != nullptr) {
        db_opts.gateway_mtls_proxy_header = opts->gateway_mtls_proxy_header;
    }
    db_opts.ApplyEdgeProfile();

    const char* profile_name = "user_defined";
    if (db_opts.edge_profile == pomai::EdgeProfile::kLowRam) profile_name = "edge-low-ram";
    if (db_opts.edge_profile == pomai::EdgeProfile::kBalanced) profile_name = "edge-balanced";
    if (db_opts.edge_profile == pomai::EdgeProfile::kThroughput) profile_name = "edge-throughput";

    std::string json = "{";
    json += "\"profile\":\"" + std::string(profile_name) + "\",";
    json += "\"dim\":" + std::to_string(db_opts.dim) + ",";
    json += "\"shard_count\":" + std::to_string(db_opts.shard_count) + ",";
    json += "\"fsync\":\"" + std::string(db_opts.fsync == pomai::FsyncPolicy::kAlways ? "always" : "never") + "\",";
    json += "\"memtable_flush_threshold_mb\":" + std::to_string(db_opts.memtable_flush_threshold_mb) + ",";
    json += "\"max_memtable_mb\":" + std::to_string(db_opts.max_memtable_mb) + ",";
    json += "\"gateway_rate_limit_per_sec\":" + std::to_string(db_opts.gateway_rate_limit_per_sec) + ",";
    json += "\"gateway_idempotency_ttl_sec\":" + std::to_string(db_opts.gateway_idempotency_ttl_sec) + ",";
    json += "\"gateway_upstream_sync_enabled\":" + std::string(db_opts.gateway_upstream_sync_enabled ? "true" : "false") + ",";
    json += "\"gateway_upstream_sync_url\":\"" + JsonEscape(db_opts.gateway_upstream_sync_url) + "\",";
    json += "\"gateway_token_file\":\"" + JsonEscape(db_opts.gateway_token_file) + "\",";
    json += "\"gateway_require_mtls_proxy_header\":" + std::string(db_opts.gateway_require_mtls_proxy_header ? "true" : "false") + ",";
    json += "\"gateway_mtls_proxy_header\":\"" + JsonEscape(db_opts.gateway_mtls_proxy_header) + "\",";
    json += "\"index\":{";
    json += "\"type\":\"" + std::string(db_opts.index_params.type == pomai::IndexType::kHnsw ? "hnsw" : "ivf") + "\",";
    json += "\"hnsw_m\":" + std::to_string(db_opts.index_params.hnsw_m) + ",";
    json += "\"hnsw_ef_construction\":" + std::to_string(db_opts.index_params.hnsw_ef_construction) + ",";
    json += "\"hnsw_ef_search\":" + std::to_string(db_opts.index_params.hnsw_ef_search) + ",";
    json += "\"adaptive_threshold\":" + std::to_string(db_opts.index_params.adaptive_threshold);
    json += "}}";

    char* p = static_cast<char*>(palloc_malloc_aligned(json.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "allocation failed");
    std::memcpy(p, json.data(), json.size());
    p[json.size()] = '\0';
    *out_json = p;
    *out_len = json.size();
    return nullptr;
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

    pomai::DBOptions db_opts;
    db_opts.path = opts->path;
    db_opts.dim = opts->dim;
    db_opts.shard_count = opts->shards > 0 ? opts->shards : 4u;
    db_opts.fsync = (opts->fsync_policy == POMAI_FSYNC_POLICY_ALWAYS)
                       ? pomai::FsyncPolicy::kAlways
                       : pomai::FsyncPolicy::kNever;
    db_opts.metric = (opts->metric == 1) ? pomai::MetricType::kInnerProduct : pomai::MetricType::kL2;
    db_opts.index_params.adaptive_threshold = opts->adaptive_threshold;
    if (opts->index_type == 1) {
        db_opts.index_params.type = pomai::IndexType::kHnsw;
        db_opts.index_params.hnsw_m = opts->hnsw_m;
        db_opts.index_params.hnsw_ef_construction = opts->hnsw_ef_construction;
        db_opts.index_params.hnsw_ef_search = opts->hnsw_ef_search;
    } else {
        db_opts.index_params.type = pomai::IndexType::kIvfFlat;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, edge_profile) + sizeof(uint8_t))) {
        db_opts.edge_profile = static_cast<pomai::EdgeProfile>(opts->edge_profile);
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_rate_limit_per_sec) + sizeof(uint32_t)) &&
        opts->gateway_rate_limit_per_sec > 0) {
        db_opts.gateway_rate_limit_per_sec = opts->gateway_rate_limit_per_sec;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_idempotency_ttl_sec) + sizeof(uint32_t)) &&
        opts->gateway_idempotency_ttl_sec > 0) {
        db_opts.gateway_idempotency_ttl_sec = opts->gateway_idempotency_ttl_sec;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_token_file) + sizeof(const char*)) &&
        opts->gateway_token_file != nullptr) {
        db_opts.gateway_token_file = opts->gateway_token_file;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_upstream_sync_url) + sizeof(const char*)) &&
        opts->gateway_upstream_sync_url != nullptr) {
        db_opts.gateway_upstream_sync_url = opts->gateway_upstream_sync_url;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_upstream_sync_enabled) + sizeof(bool))) {
        db_opts.gateway_upstream_sync_enabled = opts->gateway_upstream_sync_enabled;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_require_mtls_proxy_header) + sizeof(bool))) {
        db_opts.gateway_require_mtls_proxy_header = opts->gateway_require_mtls_proxy_header;
    }
    if (opts->struct_size >= static_cast<uint32_t>(offsetof(pomai_options_t, gateway_mtls_proxy_header) + sizeof(const char*)) &&
        opts->gateway_mtls_proxy_header != nullptr) {
        db_opts.gateway_mtls_proxy_header = opts->gateway_mtls_proxy_header;
    }
    db_opts.ApplyEdgeProfile();

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(db_opts, &db);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    void* raw = palloc_malloc_aligned(sizeof(pomai_db_t), alignof(pomai_db_t));
    if (!raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "db handle allocation failed");
    *out_db = new (raw) pomai_db_t{std::move(db)};
    return nullptr;
}

pomai_status_t* pomai_close(pomai_db_t* db) {
    if (db == nullptr) {
        return nullptr;
    }
    auto st = db->db->Close();
    db->~pomai_db_t();
    palloc_free(db);
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
    return ToCStatus(db->db->PutVector(item->id, vec, ToMetadata(*item)));
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

    void* raw = palloc_malloc_aligned(sizeof(RecordWrapper), alignof(RecordWrapper));
    if (!raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "record allocation failed");
    auto* w = new (raw) RecordWrapper();
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
    if (record) {
        auto* w = reinterpret_cast<RecordWrapper*>(record);
        w->~RecordWrapper();
        palloc_free(w);
    }
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
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "filter_expression must use tenant/device_id/location_id=<value>");
    }
    if (query->struct_size >= static_cast<uint32_t>(sizeof(pomai_query_t))) {
        opts.as_of_ts = query->as_of_ts;
        opts.as_of_lsn = query->as_of_lsn;
        if (query->partition_device_id) opts.partition_device_id = query->partition_device_id;
        if (query->partition_location_id) opts.partition_location_id = query->partition_location_id;
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

    void* raw = palloc_malloc_aligned(sizeof(SearchResultsWrapper), alignof(SearchResultsWrapper));
    if (!raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "search results allocation failed");
    auto* w = new (raw) SearchResultsWrapper();
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
    w->pub.total_shards_count = res.total_shards_count;
    w->pub.pruned_shards_count = res.pruned_shards_count;
    w->pub.aggregate_value = 0.0;
    w->pub.aggregate_op = 0;
    w->pub.mesh_lod_level = 0;
    if (!res.aggregates.empty()) {
        w->pub.aggregate_value = res.aggregates.front().value;
        w->pub.aggregate_op = static_cast<uint32_t>(res.aggregates.front().op);
    } else if (query->struct_size >= static_cast<uint32_t>(sizeof(pomai_query_t)) && query->aggregate_op != 0) {
        w->pub.aggregate_value = ComputeAggregateValue(query->aggregate_op, res.hits);
        w->pub.aggregate_op = query->aggregate_op;
    }
    if (opts.zero_copy && !res.zero_copy_pointers.empty()) {
        size_t n = res.zero_copy_pointers.size();
        w->pub.zero_copy_pointers = static_cast<pomai_semantic_pointer_t*>(
            palloc_malloc_aligned(n * sizeof(pomai_semantic_pointer_t), alignof(pomai_semantic_pointer_t)));
        if (w->pub.zero_copy_pointers) {
            for (size_t i = 0; i < n; ++i) {
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
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "filter_expression must use tenant/device_id/location_id=<value>");
    }
    if (queries[0].struct_size >= static_cast<uint32_t>(sizeof(pomai_query_t))) {
        opts.as_of_ts = queries[0].as_of_ts;
        opts.as_of_lsn = queries[0].as_of_lsn;
        if (queries[0].partition_device_id) opts.partition_device_id = queries[0].partition_device_id;
        if (queries[0].partition_location_id) opts.partition_location_id = queries[0].partition_location_id;
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
    auto st = db->db->SearchBatch(std::span<const float>(flat_queries.data(), flat_queries.size()), static_cast<uint32_t>(num_queries), topk, opts, &batch_res);
    
    if (!st.ok() && st.code() != pomai::ErrorCode::kPartial) {
        return ToCStatus(st);
    }

    // Allocate array of results (palloc, no new)
    pomai_search_results_t* arr = static_cast<pomai_search_results_t*>(
        palloc_malloc_aligned(num_queries * sizeof(pomai_search_results_t), alignof(pomai_search_results_t)));
    if (!arr) {
        return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "batch results allocation failed");
    }
    std::memset(arr, 0, num_queries * sizeof(pomai_search_results_t));
    *out = arr;

    for (size_t q = 0; q < num_queries; ++q) {
        const auto& res = batch_res[q];
        pomai_search_results_t& pub = arr[q];

        pub.struct_size = static_cast<uint32_t>(sizeof(pomai_search_results_t));
        pub.count = res.hits.size();
        pub.total_shards_count = res.total_shards_count;
        pub.pruned_shards_count = res.pruned_shards_count;

        pub.ids = static_cast<uint64_t*>(palloc_malloc_aligned(pub.count * sizeof(uint64_t), alignof(uint64_t)));
        pub.scores = static_cast<float*>(palloc_malloc_aligned(pub.count * sizeof(float), alignof(float)));
        pub.shard_ids = static_cast<uint32_t*>(palloc_malloc_aligned(pub.count * sizeof(uint32_t), alignof(uint32_t)));
        if (!pub.ids || !pub.scores || !pub.shard_ids) {
            pomai_search_batch_free(arr, num_queries);
            return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "batch hit array allocation failed");
        }

        for (size_t i = 0; i < pub.count; ++i) {
            pub.ids[i] = res.hits[i].id;
            pub.scores[i] = res.hits[i].score;
            pub.shard_ids[i] = UINT32_MAX;
        }
        pub.aggregate_value = 0.0;
        pub.aggregate_op = 0;
        pub.mesh_lod_level = 0;
        if (!res.aggregates.empty()) {
            pub.aggregate_value = res.aggregates.front().value;
            pub.aggregate_op = static_cast<uint32_t>(res.aggregates.front().op);
        } else if (queries[q].struct_size >= static_cast<uint32_t>(sizeof(pomai_query_t)) && queries[q].aggregate_op != 0) {
            pub.aggregate_value = ComputeAggregateValue(queries[q].aggregate_op, res.hits);
            pub.aggregate_op = queries[q].aggregate_op;
        }

        if (opts.zero_copy && !res.zero_copy_pointers.empty()) {
            size_t n = res.zero_copy_pointers.size();
            pub.zero_copy_pointers = static_cast<pomai_semantic_pointer_t*>(
                palloc_malloc_aligned(n * sizeof(pomai_semantic_pointer_t), alignof(pomai_semantic_pointer_t)));
            if (pub.zero_copy_pointers) {
                for (size_t i = 0; i < n; ++i) {
                    pub.zero_copy_pointers[i].struct_size = sizeof(pomai_semantic_pointer_t);
                    pub.zero_copy_pointers[i].raw_data_ptr = res.zero_copy_pointers[i].raw_data_ptr;
                    pub.zero_copy_pointers[i].dim = res.zero_copy_pointers[i].dim;
                    pub.zero_copy_pointers[i].quant_min = res.zero_copy_pointers[i].quant_min;
                    pub.zero_copy_pointers[i].quant_inv_scale = res.zero_copy_pointers[i].quant_inv_scale;
                    pub.zero_copy_pointers[i].session_id = res.zero_copy_pointers[i].session_id;
                }
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
    if (!results) return;
    if (results->zero_copy_pointers) {
        palloc_free(results->zero_copy_pointers);
    }
    auto* w = reinterpret_cast<SearchResultsWrapper*>(results);
    w->~SearchResultsWrapper();
    palloc_free(w);
}

pomai_status_t* pomai_create_membrane_kind(pomai_db_t* db, const char* name, uint32_t dim, uint32_t shard_count, uint32_t kind) {
    if (db == nullptr || name == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/name must be non-null");
    }
    pomai::MembraneSpec spec;
    spec.name = name;
    spec.dim = dim;
    spec.shard_count = shard_count > 0 ? shard_count : 1u;
    spec.kind = static_cast<pomai::MembraneKind>(kind);
    auto st = db->db->CreateMembrane(spec);
    if (!st.ok()) return ToCStatus(st);
    return ToCStatus(db->db->OpenMembrane(name));
}

pomai_status_t* pomai_create_membrane_kind_with_retention(
    pomai_db_t* db, const char* name, uint32_t dim, uint32_t shard_count, uint32_t kind,
    uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes) {
    if (db == nullptr || name == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/name must be non-null");
    }
    pomai::MembraneSpec spec;
    spec.name = name;
    spec.dim = dim;
    spec.shard_count = shard_count > 0 ? shard_count : 1u;
    spec.kind = static_cast<pomai::MembraneKind>(kind);
    spec.ttl_sec = ttl_sec;
    spec.retention_max_count = retention_max_count;
    spec.retention_max_bytes = retention_max_bytes;
    auto st = db->db->CreateMembrane(spec);
    if (!st.ok()) return ToCStatus(st);
    return ToCStatus(db->db->OpenMembrane(name));
}

pomai_status_t* pomai_ts_put(pomai_db_t* db, const char* membrane_name, uint64_t series_id, uint64_t ts, double value) {
    if (db == nullptr || membrane_name == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/membrane_name null");
    return ToCStatus(db->db->TsPut(membrane_name, series_id, ts, value));
}

pomai_status_t* pomai_kv_put(pomai_db_t* db, const char* membrane_name, const char* key, const char* value) {
    if (db == nullptr || membrane_name == nullptr || key == nullptr || value == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->KvPut(membrane_name, key, value));
}

pomai_status_t* pomai_kv_get(pomai_db_t* db, const char* membrane_name, const char* key, char** out_value, size_t* out_len) {
    if (db == nullptr || membrane_name == nullptr || key == nullptr || out_value == nullptr || out_len == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    std::string v;
    auto st = db->db->KvGet(membrane_name, key, &v);
    if (!st.ok()) return ToCStatus(st);
    char* p = static_cast<char*>(palloc_malloc_aligned(v.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "allocation failed");
    std::memcpy(p, v.data(), v.size());
    p[v.size()] = '\0';
    *out_value = p;
    *out_len = v.size();
    return nullptr;
}

pomai_status_t* pomai_kv_delete(pomai_db_t* db, const char* membrane_name, const char* key) {
    if (db == nullptr || membrane_name == nullptr || key == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->KvDelete(membrane_name, key));
}

pomai_status_t* pomai_meta_put(pomai_db_t* db, const char* membrane_name, const char* gid, const char* value) {
    if (db == nullptr || membrane_name == nullptr || gid == nullptr || value == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->MetaPut(membrane_name, gid, value));
}

pomai_status_t* pomai_meta_get(pomai_db_t* db, const char* membrane_name, const char* gid, char** out_value, size_t* out_len) {
    if (db == nullptr || membrane_name == nullptr || gid == nullptr || out_value == nullptr || out_len == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    std::string v;
    auto st = db->db->MetaGet(membrane_name, gid, &v);
    if (!st.ok()) return ToCStatus(st);
    char* p = static_cast<char*>(palloc_malloc_aligned(v.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "allocation failed");
    std::memcpy(p, v.data(), v.size());
    p[v.size()] = '\0';
    *out_value = p;
    *out_len = v.size();
    return nullptr;
}

pomai_status_t* pomai_meta_delete(pomai_db_t* db, const char* membrane_name, const char* gid) {
    if (db == nullptr || membrane_name == nullptr || gid == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->MetaDelete(membrane_name, gid));
}

pomai_status_t* pomai_link_objects(pomai_db_t* db, const char* gid, uint64_t vector_id, uint64_t graph_vertex_id, uint64_t mesh_id) {
    if (db == nullptr || gid == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->LinkObjects(gid, vector_id, graph_vertex_id, mesh_id));
}

pomai_status_t* pomai_unlink_objects(pomai_db_t* db, const char* gid) {
    if (db == nullptr || gid == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->UnlinkObjects(gid));
}

pomai_status_t* pomai_edge_gateway_start(pomai_db_t* db, uint16_t http_port, uint16_t ingest_port) {
    if (db == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->StartEdgeGateway(http_port, ingest_port));
}

pomai_status_t* pomai_edge_gateway_start_secure(pomai_db_t* db, uint16_t http_port, uint16_t ingest_port, const char* auth_token) {
    if (db == nullptr || auth_token == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->StartEdgeGatewaySecure(http_port, ingest_port, auth_token));
}

pomai_status_t* pomai_edge_gateway_stop(pomai_db_t* db) {
    if (db == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->StopEdgeGateway());
}

pomai_status_t* pomai_list_membranes_json(pomai_db_t* db, char** out_json, size_t* out_len) {
    if (db == nullptr || out_json == nullptr || out_len == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    }
    std::vector<std::string> membranes;
    auto st = db->db->ListMembranes(&membranes);
    if (!st.ok()) return ToCStatus(st);
    std::string json = "[";
    for (size_t i = 0; i < membranes.size(); ++i) {
        if (i > 0) json += ",";
        json += "\"" + JsonEscape(membranes[i]) + "\"";
    }
    json += "]";
    char* p = static_cast<char*>(palloc_malloc_aligned(json.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "allocation failed");
    std::memcpy(p, json.data(), json.size());
    p[json.size()] = '\0';
    *out_json = p;
    *out_len = json.size();
    return nullptr;
}

pomai_status_t* pomai_compact_membrane(pomai_db_t* db, const char* membrane_name) {
    if (db == nullptr || membrane_name == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    }
    return ToCStatus(db->db->Compact(membrane_name));
}

pomai_status_t* pomai_update_membrane_retention(
    pomai_db_t* db, const char* membrane_name,
    uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes) {
    if (db == nullptr || membrane_name == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    }
    return ToCStatus(db->db->UpdateMembraneRetention(membrane_name, ttl_sec, retention_max_count, retention_max_bytes));
}

pomai_status_t* pomai_get_membrane_retention_json(
    pomai_db_t* db, const char* membrane_name, char** out_json, size_t* out_len) {
    if (db == nullptr || membrane_name == nullptr || out_json == nullptr || out_len == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    }
    uint32_t ttl = 0;
    uint32_t max_count = 0;
    uint64_t max_bytes = 0;
    auto st = db->db->GetMembraneRetention(membrane_name, &ttl, &max_count, &max_bytes);
    if (!st.ok()) return ToCStatus(st);
    std::string json = "{";
    json += "\"membrane\":\"" + JsonEscape(membrane_name) + "\",";
    json += "\"ttl_sec\":" + std::to_string(ttl) + ",";
    json += "\"retention_max_count\":" + std::to_string(max_count) + ",";
    json += "\"retention_max_bytes\":" + std::to_string(max_bytes);
    json += "}";
    char* p = static_cast<char*>(palloc_malloc_aligned(json.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "allocation failed");
    std::memcpy(p, json.data(), json.size());
    p[json.size()] = '\0';
    *out_json = p;
    *out_len = json.size();
    return nullptr;
}

pomai_status_t* pomai_sketch_add(pomai_db_t* db, const char* membrane_name, const char* key, uint64_t increment) {
    if (db == nullptr || membrane_name == nullptr || key == nullptr) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->SketchAdd(membrane_name, key, increment));
}

pomai_status_t* pomai_blob_put(pomai_db_t* db, const char* membrane_name, uint64_t blob_id, const uint8_t* data, size_t len) {
    if (db == nullptr || membrane_name == nullptr || (len > 0 && data == nullptr)) return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid args");
    return ToCStatus(db->db->BlobPut(membrane_name, blob_id, std::span<const uint8_t>(data, len)));
}

// RAG (full DB with membrane manager)
pomai_status_t* pomai_create_rag_membrane(pomai_db_t* db, const char* name, uint32_t dim, uint32_t shard_count) {
    if (db == nullptr || name == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/name must be non-null");
    }
    pomai::MembraneSpec spec;
    spec.name = name;
    spec.dim = dim;
    spec.shard_count = shard_count > 0 ? shard_count : 4u;
    spec.kind = pomai::MembraneKind::kRag;
    auto st = db->db->CreateMembrane(spec);
    if (!st.ok()) return ToCStatus(st);
    return ToCStatus(db->db->OpenMembrane(name));
}

pomai_status_t* pomai_put_chunk(pomai_db_t* db, const char* membrane_name, const pomai_rag_chunk_t* chunk) {
    if (db == nullptr || membrane_name == nullptr || chunk == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/membrane_name/chunk must be non-null");
    }
    if (chunk->token_ids == nullptr || chunk->token_count == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "chunk requires token_ids and token_count > 0");
    }
    pomai::RagChunk c;
    c.chunk_id = chunk->chunk_id;
    c.doc_id = chunk->doc_id;
    c.tokens.assign(chunk->token_ids, chunk->token_ids + chunk->token_count);
    if (chunk->vector != nullptr && chunk->dim > 0) {
        c.vec = pomai::VectorView(chunk->vector, chunk->dim);
    }
    if (chunk->chunk_text != nullptr && chunk->chunk_text_len > 0) {
        c.chunk_text.assign(chunk->chunk_text, chunk->chunk_text_len);
    }
    return ToCStatus(db->db->PutChunk(membrane_name, c));
}

pomai_status_t* pomai_search_rag(pomai_db_t* db, const char* membrane_name, const pomai_rag_query_t* query,
                                 const pomai_rag_search_options_t* opts, pomai_rag_search_result_t* out_result) {
    if (db == nullptr || membrane_name == nullptr || query == nullptr || out_result == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/membrane_name/query/out_result must be non-null");
    }
    pomai::RagQuery q;
    if (query->token_ids != nullptr && query->token_count > 0) {
        q.tokens = std::span<const pomai::TokenId>(query->token_ids, query->token_count);
    }
    if (query->vector != nullptr && query->dim > 0) {
        q.vec = pomai::VectorView(query->vector, query->dim);
    }
    q.topk = query->topk > 0 ? query->topk : 10u;
    if (q.tokens.empty() && !q.vec.has_value()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "query requires token_ids or vector");
    }
    pomai::RagSearchOptions o;
    if (opts != nullptr) {
        o.candidate_budget = opts->candidate_budget;
        o.token_budget = opts->token_budget;
        o.enable_vector_rerank = opts->enable_vector_rerank;
    }
    pomai::RagSearchResult res;
    auto st = db->db->SearchRag(membrane_name, q, o, &res);
    if (!st.ok()) return ToCStatus(st);

    out_result->hit_count = res.hits.size();
    if (res.hits.empty()) {
        out_result->hits = nullptr;
        return nullptr;
    }
    void* hits_raw = palloc_malloc_aligned(res.hits.size() * sizeof(pomai_rag_hit_t), alignof(pomai_rag_hit_t));
    if (!hits_raw) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "hits allocation failed");
    out_result->hits = static_cast<pomai_rag_hit_t*>(hits_raw);
    for (size_t i = 0; i < res.hits.size(); ++i) {
        const auto& h = res.hits[i];
        pomai_rag_hit_t* out_h = &out_result->hits[i];
        out_h->chunk_id = h.chunk_id;
        out_h->doc_id = h.doc_id;
        out_h->score = h.score;
        out_h->token_matches = h.token_matches;
        out_h->chunk_text = nullptr;
        out_h->chunk_text_len = 0;
        if (!h.chunk_text.empty()) {
            char* p = static_cast<char*>(palloc_malloc_aligned(h.chunk_text.size() + 1, alignof(char)));
            if (p) {
                std::memcpy(p, h.chunk_text.data(), h.chunk_text.size());
                p[h.chunk_text.size()] = '\0';
                out_h->chunk_text = p;
                out_h->chunk_text_len = h.chunk_text.size();
            }
        }
    }
    return nullptr;
}

void pomai_rag_search_result_free(pomai_rag_search_result_t* result) {
    if (result == nullptr) return;
    if (result->hits != nullptr) {
        for (size_t i = 0; i < result->hit_count; ++i) {
            if (result->hits[i].chunk_text != nullptr) {
                palloc_free(result->hits[i].chunk_text);
            }
        }
        palloc_free(result->hits);
    }
    result->hits = nullptr;
    result->hit_count = 0;
}

pomai_status_t* pomai_rag_pipeline_create(pomai_db_t* db, const char* membrane_name, uint32_t embedding_dim,
    const pomai_rag_chunk_options_t* chunk_options, pomai_rag_pipeline_t** out_pipeline) {
    if (db == nullptr || membrane_name == nullptr || out_pipeline == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/membrane_name/out_pipeline must be non-null");
    }
    pomai::RagPipelineOptions opts;
    if (chunk_options != nullptr) {
        opts.max_chunk_bytes = chunk_options->max_chunk_bytes > 0 ? chunk_options->max_chunk_bytes : 512u;
        opts.max_doc_bytes = chunk_options->max_doc_bytes > 0 ? chunk_options->max_doc_bytes : 4u * 1024u * 1024u;
        opts.max_chunks_per_batch = chunk_options->max_chunks_per_batch > 0 ? chunk_options->max_chunks_per_batch : 32u;
        opts.overlap_bytes = chunk_options->overlap_bytes;
    }
    auto* wrap = new pomai_rag_pipeline_t();
    wrap->mock_embed = std::make_unique<pomai::MockEmbeddingProvider>(embedding_dim);
    wrap->pipeline = std::make_unique<pomai::RagPipeline>(db->db.get(), membrane_name, embedding_dim, wrap->mock_embed.get(), opts);
    *out_pipeline = wrap;
    return nullptr;
}

pomai_status_t* pomai_rag_ingest_document(pomai_rag_pipeline_t* pipeline, uint64_t doc_id,
    const char* text_buf, size_t text_len) {
    if (pipeline == nullptr || pipeline->pipeline == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "pipeline must be non-null");
    }
    std::string_view text(text_buf ? text_buf : "", text_len);
    return ToCStatus(pipeline->pipeline->IngestDocument(doc_id, text));
}

pomai_status_t* pomai_rag_retrieve_context(pomai_rag_pipeline_t* pipeline, const char* query_buf, size_t query_len,
    uint32_t top_k, char** out_buf, size_t* out_len) {
    if (pipeline == nullptr || pipeline->pipeline == nullptr || out_buf == nullptr || out_len == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "pipeline/out_buf/out_len must be non-null");
    }
    std::string_view query(query_buf ? query_buf : "", query_len);
    std::string context;
    auto st = pipeline->pipeline->RetrieveContext(query, top_k, &context);
    if (!st.ok()) return ToCStatus(st);
    *out_len = context.size();
    if (context.empty()) {
        *out_buf = nullptr;
        return nullptr;
    }
    char* p = static_cast<char*>(palloc_malloc_aligned(context.size() + 1, alignof(char)));
    if (!p) return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "context buffer allocation failed");
    std::memcpy(p, context.data(), context.size());
    p[context.size()] = '\0';
    *out_buf = p;
    return nullptr;
}

pomai_status_t* pomai_rag_retrieve_context_buf(pomai_rag_pipeline_t* pipeline, const char* query_buf, size_t query_len,
    uint32_t top_k, char* out_buf, size_t max_len, size_t* out_len) {
    if (pipeline == nullptr || pipeline->pipeline == nullptr || out_buf == nullptr || out_len == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "pipeline/out_buf/out_len must be non-null");
    }
    std::string_view query(query_buf ? query_buf : "", query_len);
    std::string context;
    auto st = pipeline->pipeline->RetrieveContext(query, top_k, &context);
    if (!st.ok()) return ToCStatus(st);
    *out_len = (std::min)(context.size(), max_len > 0 ? max_len - 1 : 0);
    if (*out_len > 0) {
        std::memcpy(out_buf, context.data(), *out_len);
        out_buf[*out_len] = '\0';
    }
    return nullptr;
}

void pomai_rag_pipeline_free(pomai_rag_pipeline_t* pipeline) {
    if (pipeline == nullptr) return;
    delete pipeline;
}

void pomai_search_batch_free(pomai_search_results_t* results, size_t num_queries) {
    if (!results) return;
    for (size_t i = 0; i < num_queries; ++i) {
        palloc_free(results[i].ids);
        palloc_free(results[i].scores);
        palloc_free(results[i].shard_ids);
        palloc_free(results[i].zero_copy_pointers);
    }
    palloc_free(results);
}

void pomai_release_pointer(uint64_t session_id) {
    pomai::core::MemoryPinManager::Instance().Unpin(session_id);
}

void pomai_free(void* p) {
    palloc_free(p);
}

}  // extern "C"
