#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pomai/c_api.h"

static void check_status(const char* step, pomai_status_t* st) {
    if (st == NULL) {
        return;
    }
    fprintf(stderr, "%s failed [%d]: %s\n", step, pomai_status_code(st), pomai_status_message(st));
    pomai_status_free(st);
    exit(1);
}

int main(void) {
    pomai_options_t opts;
    pomai_options_init(&opts);
    opts.path = "example_db_c_basic";
    opts.dim = 4;

    pomai_db_t* db = NULL;
    check_status("open", pomai_open(&opts, &db));

    float vectors[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0.9f, 0.1f, 0, 0}};
    pomai_upsert_t batch[3];
    memset(batch, 0, sizeof(batch));
    for (size_t i = 0; i < 3; ++i) {
        batch[i].id = (uint64_t)(i + 1);
        batch[i].vector = vectors[i];
        batch[i].dim = 4;
    }
    check_status("put_batch", pomai_put_batch(db, batch, 3));

    float query_vec[4] = {1, 0, 0, 0};
    pomai_query_t query;
    memset(&query, 0, sizeof(query));
    query.vector = query_vec;
    query.dim = 4;
    query.topk = 2;

    pomai_search_results_t* res = NULL;
    check_status("search", pomai_search(db, &query, &res));
    for (size_t i = 0; i < res->count; ++i) {
        printf("hit[%zu] id=%llu score=%f\n", i, (unsigned long long)res->ids[i], res->scores[i]);
    }
    pomai_search_results_free(res);

    pomai_snapshot_t* snap = NULL;
    check_status("get_snapshot", pomai_get_snapshot(db, &snap));

    pomai_scan_options_t scan_opts;
    pomai_scan_options_init(&scan_opts);

    pomai_iter_t* iter = NULL;
    check_status("scan", pomai_scan(db, &scan_opts, snap, &iter));

    while (pomai_iter_valid(iter)) {
        pomai_record_view_t view;
        check_status("iter_get_record", pomai_iter_get_record(iter, &view));
        printf("scan id=%llu dim=%u v0=%f\n", (unsigned long long)view.id, view.dim, view.vector[0]);
        pomai_iter_next(iter);
    }
    check_status("iter_status", pomai_iter_status(iter));

    pomai_iter_free(iter);
    pomai_snapshot_free(snap);
    check_status("close", pomai_close(db));
    return 0;
}
