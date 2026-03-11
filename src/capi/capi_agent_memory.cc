#include "pomai/c_api.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "capi_utils.h"
#include "palloc_compat.h"
#include "pomai/agent_memory.h"
#include "pomai/options.h"

namespace {

struct AgentMemoryHandle {
    std::unique_ptr<pomai::AgentMemory> mem;
};

pomai::MetricType ToMetric(uint8_t metric) {
    switch (metric) {
        case 1:
            return pomai::MetricType::kInnerProduct;
        case 2:
            return pomai::MetricType::kCosine;
        default:
            return pomai::MetricType::kL2;
    }
}

}  // namespace

extern "C" {

pomai_status_t* pomai_agent_memory_open(
    const pomai_agent_memory_options_t* opts,
    pomai_agent_memory_t** out_mem) {
    if (opts == nullptr || out_mem == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts/out_mem must be non-null");
    }
    if (opts->path == nullptr || opts->path[0] == '\0') {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts.path must be non-empty");
    }
    if (opts->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "opts.dim must be > 0");
    }

    pomai::AgentMemoryOptions am_opts;
    am_opts.path = opts->path;
    am_opts.dim = opts->dim;
    am_opts.metric = ToMetric(opts->metric);
    am_opts.max_messages_per_agent = opts->max_messages_per_agent;
    am_opts.max_device_bytes = opts->max_device_bytes;

    std::unique_ptr<pomai::AgentMemory> mem;
    auto st = pomai::AgentMemory::Open(am_opts, &mem);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    void* raw = palloc_malloc_aligned(sizeof(AgentMemoryHandle), alignof(AgentMemoryHandle));
    if (!raw) {
        return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "agent_memory handle allocation failed");
    }
    auto* handle = new (raw) AgentMemoryHandle();
    handle->mem = std::move(mem);
    *out_mem = reinterpret_cast<pomai_agent_memory_t*>(handle);
    return nullptr;
}

pomai_status_t* pomai_agent_memory_close(
    pomai_agent_memory_t* mem) {
    if (mem == nullptr) {
        return nullptr;
    }
    auto* handle = reinterpret_cast<AgentMemoryHandle*>(mem);
    handle->~AgentMemoryHandle();
    palloc_free(handle);
    return nullptr;
}

pomai_status_t* pomai_agent_memory_append(
    pomai_agent_memory_t* mem,
    const pomai_agent_memory_record_t* record,
    uint64_t* out_id) {
    if (mem == nullptr || record == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "mem/record must be non-null");
    }
    if (record->agent_id == nullptr || record->embedding == nullptr || record->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "agent_id and embedding must be set");
    }

    auto* handle = reinterpret_cast<AgentMemoryHandle*>(mem);

    pomai::AgentMemoryRecord rec;
    rec.agent_id = record->agent_id;
    if (record->session_id) rec.session_id = record->session_id;
    rec.logical_ts = record->logical_ts;
    if (record->text) rec.text = record->text;
    rec.embedding.assign(record->embedding, record->embedding + record->dim);

    if (record->kind && std::strcmp(record->kind, "summary") == 0) {
        rec.kind = pomai::AgentMemoryKind::kSummary;
    } else if (record->kind && std::strcmp(record->kind, "knowledge") == 0) {
        rec.kind = pomai::AgentMemoryKind::kKnowledge;
    } else {
        rec.kind = pomai::AgentMemoryKind::kMessage;
    }

    pomai::VectorId id = 0;
    auto st = handle->mem->AppendMessage(rec, &id);
    if (!st.ok()) {
        return ToCStatus(st);
    }
    if (out_id) {
        *out_id = id;
    }
    return nullptr;
}

pomai_status_t* pomai_agent_memory_append_batch(
    pomai_agent_memory_t* mem,
    const pomai_agent_memory_record_t* records,
    size_t n,
    uint64_t* out_ids) {
    if (mem == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "mem must be non-null");
    }
    if (n == 0) {
        return nullptr;
    }
    if (records == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "records must be non-null");
    }

    auto* handle = reinterpret_cast<AgentMemoryHandle*>(mem);

    std::vector<pomai::AgentMemoryRecord> batch;
    batch.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& r = records[i];
        if (r.agent_id == nullptr || r.embedding == nullptr || r.dim == 0) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "all records require agent_id and embedding");
        }
        pomai::AgentMemoryRecord rec;
        rec.agent_id = r.agent_id;
        if (r.session_id) rec.session_id = r.session_id;
        rec.logical_ts = r.logical_ts;
        if (r.text) rec.text = r.text;
        rec.embedding.assign(r.embedding, r.embedding + r.dim);
        if (r.kind && std::strcmp(r.kind, "summary") == 0) {
            rec.kind = pomai::AgentMemoryKind::kSummary;
        } else if (r.kind && std::strcmp(r.kind, "knowledge") == 0) {
            rec.kind = pomai::AgentMemoryKind::kKnowledge;
        } else {
            rec.kind = pomai::AgentMemoryKind::kMessage;
        }
        batch.push_back(std::move(rec));
    }

    std::vector<pomai::VectorId> ids;
    auto st = handle->mem->AppendBatch(batch, &ids);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    if (out_ids != nullptr) {
        for (size_t i = 0; i < ids.size(); ++i) {
            out_ids[i] = ids[i];
        }
    }
    return nullptr;
}

pomai_status_t* pomai_agent_memory_get_recent(
    pomai_agent_memory_t* mem,
    const char* agent_id,
    const char* session_id,
    size_t limit,
    pomai_agent_memory_result_set_t** out) {
    if (mem == nullptr || agent_id == nullptr || out == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "mem/agent_id/out must be non-null");
    }
    auto* handle = reinterpret_cast<AgentMemoryHandle*>(mem);

    std::vector<pomai::AgentMemoryRecord> recs;
    auto st = handle->mem->GetRecent(agent_id,
                                  session_id ? session_id : std::string_view(),
                                  limit,
                                  &recs);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    void* raw = palloc_malloc_aligned(sizeof(pomai_agent_memory_result_set_t),
                                      alignof(pomai_agent_memory_result_set_t));
    if (!raw) {
        return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "result_set allocation failed");
    }
    auto* rs = new (raw) pomai_agent_memory_result_set_t();
    rs->struct_size = static_cast<uint32_t>(sizeof(pomai_agent_memory_result_set_t));
    rs->count = recs.size();

    if (rs->count == 0) {
        rs->records = nullptr;
        *out = rs;
        return nullptr;
    }

    rs->records = static_cast<pomai_agent_memory_record_t*>(
        palloc_malloc_aligned(rs->count * sizeof(pomai_agent_memory_record_t),
                              alignof(pomai_agent_memory_record_t)));
    if (!rs->records) {
        rs->count = 0;
        *out = rs;
        return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "records allocation failed");
    }

    for (size_t i = 0; i < rs->count; ++i) {
        const auto& src = recs[i];
        auto& dst = rs->records[i];
        dst.struct_size = static_cast<uint32_t>(sizeof(pomai_agent_memory_record_t));
        // For simplicity, store pointers into temporary std::string copies
        // owned by a side buffer; to keep ownership simple, we duplicate into
        // independent C strings.
        auto dup_cstr = [](const std::string& s) -> char* {
            char* p = static_cast<char*>(palloc_malloc_aligned(s.size() + 1, alignof(char)));
            if (!p) return nullptr;
            std::memcpy(p, s.data(), s.size());
            p[s.size()] = '\0';
            return p;
        };
        dst.agent_id = dup_cstr(src.agent_id);
        dst.session_id = dup_cstr(src.session_id);
        const char* kind_str = "message";
        if (src.kind == pomai::AgentMemoryKind::kSummary) kind_str = "summary";
        else if (src.kind == pomai::AgentMemoryKind::kKnowledge) kind_str = "knowledge";
        dst.kind = dup_cstr(std::string(kind_str));
        dst.logical_ts = src.logical_ts;
        dst.text = dup_cstr(src.text);
        dst.dim = static_cast<uint32_t>(src.embedding.size());
        if (dst.dim > 0) {
            float* emb = static_cast<float*>(
                palloc_malloc_aligned(dst.dim * sizeof(float), alignof(float)));
            if (!emb) {
                dst.embedding = nullptr;
                dst.dim = 0;
            } else {
                std::memcpy(emb, src.embedding.data(), dst.dim * sizeof(float));
                dst.embedding = emb;
            }
        } else {
            dst.embedding = nullptr;
        }
    }

    *out = rs;
    return nullptr;
}

void pomai_agent_memory_result_set_free(
    pomai_agent_memory_result_set_t* result) {
    if (!result) return;
    if (result->records) {
        for (size_t i = 0; i < result->count; ++i) {
            auto& r = result->records[i];
            if (r.agent_id) palloc_free((void*)r.agent_id);
            if (r.session_id) palloc_free((void*)r.session_id);
            if (r.kind) palloc_free((void*)r.kind);
            if (r.text) palloc_free((void*)r.text);
            if (r.embedding) palloc_free((void*)r.embedding);
        }
        palloc_free(result->records);
    }
    result->~pomai_agent_memory_result_set_t();
    palloc_free(result);
}

pomai_status_t* pomai_agent_memory_search(
    pomai_agent_memory_t* mem,
    const pomai_agent_memory_query_t* query,
    pomai_agent_memory_search_result_t** out) {
    if (mem == nullptr || query == nullptr || out == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "mem/query/out must be non-null");
    }
    if (query->agent_id == nullptr || query->embedding == nullptr || query->dim == 0) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "agent_id and embedding must be set");
    }

    pomai::AgentMemoryQuery q;
    q.agent_id = query->agent_id;
    if (query->session_id && query->session_id[0] != '\0') {
        q.session_id = query->session_id;
        q.has_session_filter = true;
    }
    if (query->kind && query->kind[0] != '\0') {
        if (std::strcmp(query->kind, "summary") == 0) {
            q.kind = pomai::AgentMemoryKind::kSummary;
        } else if (std::strcmp(query->kind, "knowledge") == 0) {
            q.kind = pomai::AgentMemoryKind::kKnowledge;
        } else {
            q.kind = pomai::AgentMemoryKind::kMessage;
        }
        q.has_kind_filter = true;
    }
    q.min_ts = query->min_ts;
    q.max_ts = query->max_ts;
    q.embedding.assign(query->embedding, query->embedding + query->dim);
    q.topk = query->topk;

    auto* handle = reinterpret_cast<AgentMemoryHandle*>(mem);

    pomai::AgentMemorySearchResult res;
    auto st = handle->mem->SemanticSearch(q, &res);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    void* raw = palloc_malloc_aligned(sizeof(pomai_agent_memory_search_result_t),
                                      alignof(pomai_agent_memory_search_result_t));
    if (!raw) {
        return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "search_result allocation failed");
    }
    auto* out_res = new (raw) pomai_agent_memory_search_result_t();
    out_res->struct_size = static_cast<uint32_t>(sizeof(pomai_agent_memory_search_result_t));
    out_res->count = res.hits.size();

    if (out_res->count == 0) {
        out_res->records = nullptr;
        out_res->scores = nullptr;
        *out = out_res;
        return nullptr;
    }

    out_res->records = static_cast<pomai_agent_memory_record_t*>(
        palloc_malloc_aligned(out_res->count * sizeof(pomai_agent_memory_record_t),
                              alignof(pomai_agent_memory_record_t)));
    out_res->scores = static_cast<float*>(
        palloc_malloc_aligned(out_res->count * sizeof(float), alignof(float)));
    if (!out_res->records || !out_res->scores) {
        if (out_res->records) palloc_free(out_res->records);
        if (out_res->scores) palloc_free(out_res->scores);
        out_res->records = nullptr;
        out_res->scores = nullptr;
        out_res->count = 0;
        *out = out_res;
        return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "records/scores allocation failed");
    }

    for (size_t i = 0; i < out_res->count; ++i) {
        const auto& src = res.hits[i].record;
        auto& dst = out_res->records[i];
        dst.struct_size = static_cast<uint32_t>(sizeof(pomai_agent_memory_record_t));
        auto dup_cstr = [](const std::string& s) -> char* {
            char* p = static_cast<char*>(palloc_malloc_aligned(s.size() + 1, alignof(char)));
            if (!p) return nullptr;
            std::memcpy(p, s.data(), s.size());
            p[s.size()] = '\0';
            return p;
        };
        dst.agent_id = dup_cstr(src.agent_id);
        dst.session_id = dup_cstr(src.session_id);
        const char* kind_str = "message";
        if (src.kind == pomai::AgentMemoryKind::kSummary) kind_str = "summary";
        else if (src.kind == pomai::AgentMemoryKind::kKnowledge) kind_str = "knowledge";
        dst.kind = dup_cstr(std::string(kind_str));
        dst.logical_ts = src.logical_ts;
        dst.text = dup_cstr(src.text);
        dst.dim = static_cast<uint32_t>(src.embedding.size());
        if (dst.dim > 0) {
            float* emb = static_cast<float*>(
                palloc_malloc_aligned(dst.dim * sizeof(float), alignof(float)));
            if (!emb) {
                dst.embedding = nullptr;
                dst.dim = 0;
            } else {
                std::memcpy(emb, src.embedding.data(), dst.dim * sizeof(float));
                dst.embedding = emb;
            }
        } else {
            dst.embedding = nullptr;
        }
        out_res->scores[i] = res.hits[i].score;
    }

    *out = out_res;
    return nullptr;
}

void pomai_agent_memory_search_result_free(
    pomai_agent_memory_search_result_t* result) {
    if (!result) return;
    if (result->records) {
        for (size_t i = 0; i < result->count; ++i) {
            auto& r = result->records[i];
            if (r.agent_id) palloc_free((void*)r.agent_id);
            if (r.session_id) palloc_free((void*)r.session_id);
            if (r.kind) palloc_free((void*)r.kind);
            if (r.text) palloc_free((void*)r.text);
            if (r.embedding) palloc_free((void*)r.embedding);
        }
        palloc_free(result->records);
    }
    if (result->scores) {
        palloc_free(result->scores);
    }
    result->~pomai_agent_memory_search_result_t();
    palloc_free(result);
}

pomai_status_t* pomai_agent_memory_prune_old(
    pomai_agent_memory_t* mem,
    const char* agent_id,
    size_t keep_last_n,
    int64_t min_ts_to_keep) {
    if (mem == nullptr || agent_id == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "mem/agent_id must be non-null");
    }
    auto* handle = reinterpret_cast<AgentMemoryHandle*>(mem);
    auto st = handle->mem->PruneOld(agent_id, keep_last_n, min_ts_to_keep);
    return ToCStatus(st);
}

pomai_status_t* pomai_agent_memory_prune_device(
    pomai_agent_memory_t* mem,
    uint64_t target_total_bytes) {
    if (mem == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "mem must be non-null");
    }
    auto* handle = reinterpret_cast<AgentMemoryHandle*>(mem);
    auto st = handle->mem->PruneDeviceWide(target_total_bytes);
    return ToCStatus(st);
}

pomai_status_t* pomai_agent_memory_freeze_if_needed(
    pomai_agent_memory_t* mem) {
    if (mem == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "mem must be non-null");
    }
    auto* handle = reinterpret_cast<AgentMemoryHandle*>(mem);
    auto st = handle->mem->FreezeIfNeeded();
    return ToCStatus(st);
}

}  // extern "C"

