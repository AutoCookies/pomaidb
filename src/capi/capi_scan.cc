#include "pomai/c_api.h"
#include "pomai/pomai.h"
#include "pomai/iterator.h"
#include "capi_utils.h"

struct pomai_db_t {
    std::unique_ptr<pomai::DB> db;
};

struct pomai_snapshot_t {
    std::shared_ptr<pomai::Snapshot> snap;
};

struct pomai_iter_t {
    std::unique_ptr<pomai::SnapshotIterator> iter;
};

extern "C" {

pomai_status_t* pomai_get_snapshot(pomai_db_t* db, pomai_snapshot_t** out_snap) {
    if (!db || !out_snap) return ToCStatus(pomai::Status::InvalidArgument("null args"));
    
    std::shared_ptr<pomai::Snapshot> snap;
    // Hardcoded default membrane for now
    auto st = db->db->GetSnapshot("__default__", &snap);
    if (!st.ok()) return ToCStatus(st);
    
    *out_snap = new pomai_snapshot_t{std::move(snap)};
    return nullptr;
}

void pomai_snapshot_free(pomai_snapshot_t* snap) {
    delete snap;
}

pomai_status_t* pomai_scan(pomai_db_t* db, const pomai_snapshot_t* snap, pomai_iter_t** out_iter) {
    if (!db || !snap || !out_iter) return ToCStatus(pomai::Status::InvalidArgument("null args"));
    
    std::unique_ptr<pomai::SnapshotIterator> iter;
    auto st = db->db->NewIterator("__default__", snap->snap, &iter);
    if (!st.ok()) return ToCStatus(st);
    
    *out_iter = new pomai_iter_t{std::move(iter)};
    return nullptr;
}

bool pomai_iter_valid(const pomai_iter_t* iter) {
    if (!iter || !iter->iter) return false;
    return iter->iter->Valid();
}

void pomai_iter_next(pomai_iter_t* iter) {
    if (!iter || !iter->iter) return;
    iter->iter->Next();
}

pomai_status_t* pomai_iter_status(const pomai_iter_t* iter) {
    if (!iter) return ToCStatus(pomai::Status::InvalidArgument("null iter"));
    return nullptr; // No status in SnapshotIterator yet
}

pomai_status_t* pomai_iter_get_record(const pomai_iter_t* iter, pomai_record_t* out_view) {
    if (!iter || !out_view) return ToCStatus(pomai::Status::InvalidArgument("null args"));
    if (!iter->iter->Valid()) return ToCStatus(pomai::Status::InvalidArgument("invalid iterator"));
    
    out_view->id = iter->iter->id();
    auto vec = iter->iter->vector();
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

} // extern C
