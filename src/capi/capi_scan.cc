#include "pomai/c_api.h"

#include "capi_utils.h"

namespace {
constexpr const char* kDefaultMembrane = "__default__";
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
