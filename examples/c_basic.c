#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pomai/c_api.h"
#include "pomai/c_status.h"
#include "pomai/c_types.h"

void check_status(const char* msg, pomai_status_t* st) {
    if (st) {
        fprintf(stderr, "%s failed: %s\n", msg, pomai_status_message(st));
        pomai_status_free(st);
        exit(1);
    }
    printf("%s: OK\n", msg);
}

int main() {
    printf("PomaiDB C API Example\n");
    
    pomai_options_t opts;
    pomai_options_init(&opts);
    opts.path = "example_db_c";
    
    // Clean up previous run
    system("rm -rf example_db_c");
    
    pomai_db_t* db = NULL;
    check_status("Open DB", pomai_open(&opts, &db));
    
    // --- Put ---
    float vec[512];
    for(int i=0; i<512; ++i) vec[i] = (float)i / 512.0f;
    
    pomai_upsert_t item;
    item.id = 42;
    item.dim = 512;
    item.vector = vec;
    item.metadata = NULL;
    item.metadata_len = 0;
    
    check_status("Put", pomai_put(db, &item));
    
    // --- Get ---
    pomai_record_t* rec = NULL;
    check_status("Get", pomai_get(db, 42, &rec));
    
    if (rec->id == 42) {
        printf("Retrieved vector with ID %lu\n", rec->id);
    } else {
        fprintf(stderr, "ID mismatch!\n");
    }
    pomai_record_free(rec);
    
    // --- Close ---
    check_status("Close DB", pomai_close(db));
    
    printf("Example completed successfully.\n");
    return 0;
}
