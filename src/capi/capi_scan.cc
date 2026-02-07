#include "pomai/c_api.h"

#include <chrono>
#include <cstddef>

#include "capi_utils.h"

namespace {
constexpr const char* kDefaultMembrane = "__default__";

constexpr uint32_t MinScanOptionsStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_scan_options_t, has_start_id) + sizeof(bool));
}

constexpr uint32_t MinRecordViewStructSize() {
    return static_cast<uint32_t>(offsetof(pomai_record_view_t, is_deleted) + sizeof(bool));
}

bool DeadlineExceeded(uint32_t deadline_ms) {
    if (deadline_ms == 0) {
        return false;
    }
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());
    return now_ms.count() >= deadline_ms;
}
}

extern "C" {

pomai_status_t* pomai_get_snapshot(pomai_db_t* db, pomai_snapshot_t** out_snap) {
    if (db == nullptr || out_snap == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/out_snap must be non-null");
    }

    std::shared_ptr<pomai::Snapshot> snap;
    auto st = db->db->GetSnapshot(kDefaultMembrane, &snap);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    *out_snap = new pomai_snapshot_t{std::move(snap)};
    return nullptr;
}

void pomai_snapshot_free(pomai_snapshot_t* snap) {
    delete snap;
}

pomai_status_t* pomai_scan(
    pomai_db_t* db,
    const pomai_scan_options_t* opts,
    const pomai_snapshot_t* snap,
    pomai_iter_t** out_iter) {
    if (db == nullptr || snap == nullptr || out_iter == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "db/snap/out_iter must be non-null");
    }
    if (opts != nullptr) {
        if (opts->struct_size < MinScanOptionsStructSize()) {
            return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "scan_options.struct_size is too small");
        }
        if (DeadlineExceeded(opts->deadline_ms)) {
            return MakeStatus(POMAI_STATUS_DEADLINE_EXCEEDED, "deadline exceeded before scan");
        }
    }

    std::unique_ptr<pomai::SnapshotIterator> iter;
    auto st = db->db->NewIterator(kDefaultMembrane, snap->snap, &iter);
    if (!st.ok()) {
        return ToCStatus(st);
    }

    if (opts != nullptr && opts->has_start_id) {
        while (iter->Valid() && iter->id() < opts->start_id) {
            iter->Next();
        }
    }

    if (opts != nullptr && DeadlineExceeded(opts->deadline_ms)) {
        return MakeStatus(POMAI_STATUS_DEADLINE_EXCEEDED, "deadline exceeded during scan initialization");
    }

    *out_iter = new pomai_iter_t{std::move(iter)};
    return nullptr;
}

bool pomai_iter_valid(const pomai_iter_t* iter) {
    return iter != nullptr && iter->iter != nullptr && iter->iter->Valid();
}

void pomai_iter_next(pomai_iter_t* iter) {
    if (iter == nullptr || iter->iter == nullptr || !iter->iter->Valid()) {
        return;
    }
    iter->iter->Next();
}

pomai_status_t* pomai_iter_status(const pomai_iter_t* iter) {
    if (iter == nullptr || iter->iter == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "iter must be non-null");
    }
    return nullptr;
}

pomai_status_t* pomai_iter_get_record(const pomai_iter_t* iter, pomai_record_view_t* out_view) {
    if (iter == nullptr || out_view == nullptr || iter->iter == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "iter/out_view must be non-null");
    }
    if (out_view->struct_size < MinRecordViewStructSize()) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "record_view.struct_size is too small");
    }
    if (!iter->iter->Valid()) {
        return MakeStatus(POMAI_STATUS_NOT_FOUND, "iterator is not positioned on a valid row");
    }

    const auto vec = iter->iter->vector();
    out_view->id = iter->iter->id();
    out_view->dim = static_cast<uint32_t>(vec.size());
    out_view->vector = vec.data();
    out_view->metadata = nullptr;
    out_view->metadata_len = 0;
    out_view->is_deleted = false;
    return nullptr;
}

void pomai_iter_free(pomai_iter_t* iter) {
    delete iter;
}

}  // extern "C"
