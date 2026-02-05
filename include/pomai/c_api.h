#ifndef POMAI_C_API_H
#define POMAI_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "pomai/c_status.h"
#include "pomai/c_types.h"
#include "pomai/c_version.h"

void pomai_options_init(pomai_options_t* opts);
void pomai_scan_options_init(pomai_scan_options_t* opts);

pomai_status_t* pomai_open(const pomai_options_t* opts, pomai_db_t** out_db);
pomai_status_t* pomai_close(pomai_db_t* db);

pomai_status_t* pomai_put(pomai_db_t* db, const pomai_upsert_t* item);
pomai_status_t* pomai_put_batch(pomai_db_t* db, const pomai_upsert_t* items, size_t n);
pomai_status_t* pomai_delete(pomai_db_t* db, uint64_t id);

pomai_status_t* pomai_get(pomai_db_t* db, uint64_t id, pomai_record_t** out_record);
void pomai_record_free(pomai_record_t* record);
pomai_status_t* pomai_exists(pomai_db_t* db, uint64_t id, bool* out_exists);

pomai_status_t* pomai_search(pomai_db_t* db, const pomai_query_t* query, pomai_search_results_t** out);
void pomai_search_results_free(pomai_search_results_t* results);

pomai_status_t* pomai_get_snapshot(pomai_db_t* db, pomai_snapshot_t** out_snap);
void pomai_snapshot_free(pomai_snapshot_t* snap);

pomai_status_t* pomai_scan(
    pomai_db_t* db,
    const pomai_scan_options_t* opts,
    const pomai_snapshot_t* snap,
    pomai_iter_t** out_iter);
bool pomai_iter_valid(const pomai_iter_t* iter);
void pomai_iter_next(pomai_iter_t* iter);
pomai_status_t* pomai_iter_status(const pomai_iter_t* iter);
pomai_status_t* pomai_iter_get_record(const pomai_iter_t* iter, pomai_record_view_t* out_view);
void pomai_iter_free(pomai_iter_t* iter);

void pomai_free(void* p);

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_API_H
