#include "pomai/c_api.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "palloc_compat.h"
#include "capi_utils.h"

namespace {

constexpr uint32_t MinMembraneScanOptionsStructSize() {
    return static_cast<uint32_t>(sizeof(pomai_membrane_scan_options_t));
}

constexpr uint32_t MinMembraneRecordViewStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_membrane_record_view_t, vector_dim) + sizeof(uint32_t));
}

pomai_status_t* ToMembraneScanCStatus(const pomai::Status& st) {
    if (st.ok()) return nullptr;
    if (st.code() == pomai::ErrorCode::kAborted) {
        const char* m = st.message();
        if (m && std::strstr(m, "deadline") != nullptr) {
            return MakeStatus(POMAI_STATUS_DEADLINE_EXCEEDED, m);
        }
    }
    return ToCStatus(st);
}

} // namespace

extern "C" {

void pomai_membrane_scan_options_init(pomai_membrane_scan_options_t* opts) {
    if (opts == nullptr) return;
    opts->struct_size = static_cast<uint32_t>(sizeof(pomai_membrane_scan_options_t));
    opts->max_records = 1'000'000;
    opts->max_materialized_keys = 5'000'000;
    opts->deadline_ms = 0;
    opts->max_field_bytes = 4 * 1024 * 1024;
}

pomai_status_t* pomai_membrane_scan(pomai_db_t* db, const char* membrane_name,
                                    const pomai_membrane_scan_options_t* opts,
                                    pomai_membrane_iter_t** out_iter) {
    if (db == nullptr || membrane_name == nullptr || out_iter == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/membrane_name/out_iter must be non-null");
    }
    if (opts != nullptr && opts->struct_size < MinMembraneScanOptionsStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "membrane_scan_options.struct_size is too small");
    }

    pomai::MembraneScanOptions cpp{};
    if (opts != nullptr) {
        cpp.max_records = opts->max_records;
        cpp.max_materialized_keys = opts->max_materialized_keys;
        cpp.deadline_ms = opts->deadline_ms;
        cpp.max_field_bytes = opts->max_field_bytes;
    }

    std::unique_ptr<pomai::MembraneRecordIterator> iter;
    const auto st = db->db->NewMembraneRecordIterator(membrane_name, cpp, &iter);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    void* raw = palloc_malloc_aligned(sizeof(pomai_membrane_iter_t), alignof(pomai_membrane_iter_t));
    if (raw == nullptr) {
        return MakeStatus(POMAI_STATUS_RESOURCE_EXHAUSTED, "membrane iterator allocation failed");
    }
    *out_iter = new (raw) pomai_membrane_iter_t{std::move(iter), {}, {}, {}};
    return nullptr;
}

bool pomai_membrane_iter_valid(const pomai_membrane_iter_t* iter) {
    return iter != nullptr && iter->iter != nullptr && iter->iter->Valid();
}

void pomai_membrane_iter_next(pomai_membrane_iter_t* iter) {
    if (iter == nullptr || iter->iter == nullptr || !iter->iter->Valid()) return;
    iter->iter->Next();
}

pomai_status_t* pomai_membrane_iter_status(const pomai_membrane_iter_t* iter) {
    if (iter == nullptr || iter->iter == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "iter must be non-null");
    }
    return ToMembraneScanCStatus(iter->iter->ScanStatus());
}

bool pomai_membrane_iter_truncated(const pomai_membrane_iter_t* iter) {
    return iter != nullptr && iter->iter != nullptr && iter->iter->Truncated();
}

pomai_status_t* pomai_membrane_iter_get_record(pomai_membrane_iter_t* iter,
                                             pomai_membrane_record_view_t* out_view) {
    if (iter == nullptr || out_view == nullptr || iter->iter == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "iter/out_view must be non-null");
    }
    if (out_view->struct_size < MinMembraneRecordViewStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "membrane_record_view.struct_size is too small");
    }
    if (!iter->iter->Valid()) {
        return MakeStatus(POMAI_STATUS_NOT_FOUND, "membrane iterator not on a valid row");
    }
    const pomai::MembraneRecord& r = iter->iter->Record();
    iter->key_buf = r.key;
    iter->value_buf = r.value;
    iter->vec_buf = r.vector;
    out_view->membrane_kind = static_cast<uint8_t>(r.kind);
    out_view->id = r.id;
    out_view->key = iter->key_buf.data();
    out_view->key_len = iter->key_buf.size();
    out_view->value = iter->value_buf.data();
    out_view->value_len = iter->value_buf.size();
    out_view->vector = iter->vec_buf.empty() ? nullptr : iter->vec_buf.data();
    out_view->vector_dim = static_cast<uint32_t>(iter->vec_buf.size());
    return nullptr;
}

void pomai_membrane_iter_free(pomai_membrane_iter_t* iter) {
    if (iter != nullptr) {
        iter->~pomai_membrane_iter_t();
        palloc_free(iter);
    }
}

} // extern "C"
