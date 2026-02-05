#ifndef POMAI_C_API_H
#define POMAI_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "pomai/c_version.h"
#include "pomai/c_status.h"
#include "pomai/c_types.h"

// =========================================================
// Initialization
// =========================================================

// Initialize options with default values
void pomai_options_init(pomai_options_t* opts);

// =========================================================
// DB Management
// =========================================================

// Open a database.
// On success: returns NULL, sets *out_db.
// On error: returns status, *out_db is undefined.
pomai_status_t* pomai_open(const pomai_options_t* opts, pomai_db_t** out_db);

// Close and free the database handle.
// Always returns NULL (success) or a warning status.
pomai_status_t* pomai_close(pomai_db_t* db);

// =========================================================
// Data Operations
// =========================================================

// Insert or update a single vector.
pomai_status_t* pomai_put(pomai_db_t* db, const pomai_upsert_t* item);

// Insert or update a batch of vectors.
// 'items' is an array of 'n' pomai_upsert_t structs.
pomai_status_t* pomai_put_batch(pomai_db_t* db, const pomai_upsert_t* items, size_t n);

// Get a record by ID.
// On success: returns NULL. If found, *out_record is populated.
// If NOT found, returns NULL (OK legacy) or NOT_FOUND status? 
// Convention: returns OK, checks out_record->vector != NULL or specific flag?
// Let's stick to explicitly returning NOT_FOUND status if it doesn't exist, OR 
// return OK and set *out_record to NULL if pointer based?
// The C++ API returns OK even if not found? No, C++ `Get` usually returns Status::NotFound.
// So here we return Status::NotFound if not found.
// The caller must free the record with pomai_record_free.
pomai_status_t* pomai_get(pomai_db_t* db, uint64_t id, pomai_record_t** out_record);

// Free a record obtained from pomai_get.
void pomai_record_free(pomai_record_t* record);

// Check if a vector exists.
pomai_status_t* pomai_exists(pomai_db_t* db, uint64_t id, bool* out_exists);

// Delete a vector by ID.
pomai_status_t* pomai_delete(pomai_db_t* db, uint64_t id);

// =========================================================
// Search
// =========================================================

// Search for nearest neighbors.
// On success: returns NULL, sets *out_results.
// Caller must free results with pomai_search_results_free.
pomai_status_t* pomai_search(pomai_db_t* db, const pomai_query_t* query, pomai_search_results_t** out_results);

// Free search results.
void pomai_search_results_free(pomai_search_results_t* results);

// =========================================================
// Snapshot & Scanning
// =========================================================

// Get a consistent snapshot of the database.
// Caller must free with pomai_snapshot_free.
pomai_status_t* pomai_get_snapshot(pomai_db_t* db, pomai_snapshot_t** out_snap);

// Free a snapshot.
void pomai_snapshot_free(pomai_snapshot_t* snap);

// Create an iterator from a snapshot.
// Caller must free with pomai_iter_free.
pomai_status_t* pomai_scan(pomai_db_t* db, const pomai_snapshot_t* snap, pomai_iter_t** out_iter);

// Iterator operations
bool pomai_iter_valid(const pomai_iter_t* iter);
void pomai_iter_next(pomai_iter_t* iter);
pomai_status_t* pomai_iter_status(const pomai_iter_t* iter);

// Get current record view from iterator.
// The view is valid until Next() or Free().
pomai_status_t* pomai_iter_get_record(const pomai_iter_t* iter, pomai_record_t* out_view);

void pomai_iter_free(pomai_iter_t* iter);

// =========================================================
// Utilities
// =========================================================

// Free generic memory allocated by PomaiDB (if returned as void* or similar).
void pomai_free(void* p);

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_API_H
