#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    opts.path = "example_db_c_export";
    opts.dim = 4;

    pomai_db_t* db = NULL;
    check_status("open", pomai_open(&opts, &db));

    pomai_snapshot_t* snap = NULL;
    check_status("get_snapshot", pomai_get_snapshot(db, &snap));

    pomai_scan_options_t scan_opts;
    pomai_scan_options_init(&scan_opts);

    pomai_iter_t* it = NULL;
    check_status("scan", pomai_scan(db, &scan_opts, snap, &it));

    puts("[");
    int first = 1;
    while (pomai_iter_valid(it)) {
        pomai_record_view_t row;
        check_status("iter_get_record", pomai_iter_get_record(it, &row));

        if (!first) {
            puts(",");
        }
        first = 0;
        printf("  {\"id\": %llu, \"dim\": %u}", (unsigned long long)row.id, row.dim);

        pomai_iter_next(it);
    }
    puts("\n]");

    pomai_iter_free(it);
    pomai_snapshot_free(snap);
    check_status("close", pomai_close(db));
    return 0;
}
