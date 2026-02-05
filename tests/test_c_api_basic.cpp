#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <vector>
#include "pomai/c_api.h"
#include "pomai/c_status.h"
#include "pomai/c_types.h"

#define ASSERT_POMAI_OK(expr) \
    do { \
        pomai_status_t* _st = (expr); \
        if (_st) { \
            std::string _msg = pomai_status_message(_st); \
            pomai_status_free(_st); \
            FAIL() << "Status not OK: " << _msg; \
        } \
    } while (0)

class CApiTest : public ::testing::Test {
protected:
    void SetUp() override {
        pomai_options_init(&opts);
        test_dir = "test_db_capi_" + std::to_string(rand());
        opts.path = test_dir.c_str();
        opts.shard_count = 2; // Test multi-shard logic implicitly
        
        std::string cmd = "rm -rf " + test_dir;
        system(cmd.c_str());
        
        ASSERT_POMAI_OK(pomai_open(&opts, &db));
    }

    void TearDown() override {
        if (db) {
            pomai_status_t* st = pomai_close(db);
            if (st) pomai_status_free(st);
            db = nullptr;
        }
        std::string cmd = "rm -rf " + test_dir;
        system(cmd.c_str());
    }

    std::string test_dir;
    pomai_options_t opts;
    pomai_db_t* db = nullptr;
};

TEST_F(CApiTest, BasicCRUD) {
    float vec1[512] = {0};
    vec1[0] = 1.0f;
    
    pomai_upsert_t up;
    up.id = 100;
    up.dim = 512;
    up.vector = vec1;
    up.metadata = nullptr;
    up.metadata_len = 0;
    
    ASSERT_POMAI_OK(pomai_put(db, &up));
    
    // Start asynchronous indexing if necessary, or flush/freeze?
    // C API doesn't expose Flush yet explicitly? Ah, pomai_put is durable enough?
    // Wait, DB Put is in-memory. Get checks memtable.
    
    bool exists = false;
    ASSERT_POMAI_OK(pomai_exists(db, 100, &exists));
    EXPECT_TRUE(exists);
    
    pomai_record_t* rec = nullptr;
    ASSERT_POMAI_OK(pomai_get(db, 100, &rec));
    ASSERT_NE(rec, nullptr);
    EXPECT_EQ(rec->id, 100);
    EXPECT_EQ(rec->dim, 512);
    EXPECT_FLOAT_EQ(rec->vector[0], 1.0f);
    EXPECT_FALSE(rec->is_deleted);
    pomai_record_free(rec);
    
    ASSERT_POMAI_OK(pomai_delete(db, 100));
    
    ASSERT_POMAI_OK(pomai_exists(db, 100, &exists));
    EXPECT_FALSE(exists); // Should be false or returns true but is_deleted?
    // DB::Exists returns true if found (even if deleted? No, Exists usually checks live).
    // Let's verify DB behavior.
}

TEST_F(CApiTest, Search) {
    // Insert a few items
    int n = 10;
    std::vector<float> data(n * 512, 0.0f);
    std::vector<pomai_upsert_t> items(n);
    
    for(int i=0; i<n; ++i) {
        data[i*512 + 0] = static_cast<float>(i); // Distinguish them
        items[i].id = i;
        items[i].dim = 512;
        items[i].vector = &data[i*512];
        items[i].metadata = nullptr;
        items[i].metadata_len = 0;
    }
    
    ASSERT_POMAI_OK(pomai_put_batch(db, items.data(), n));
    
    // Search for ID 5
    float query_vec[512] = {0};
    query_vec[0] = 5.0f;
    
    pomai_query_t q;
    q.vector = query_vec;
    q.dim = 512;
    q.topk = 5;
    q.filter_expr = nullptr;
    
    pomai_search_results_t* res = nullptr;
    ASSERT_POMAI_OK(pomai_search(db, &q, &res));
    ASSERT_NE(res, nullptr);
    
    EXPECT_GE(res->count, 1);
    EXPECT_EQ(res->hits[0].id, 5); // Should be exact match
    
    pomai_search_results_free(res);
}

TEST_F(CApiTest, SnapshotScan) {
    float vec[512] = {0};
    pomai_upsert_t up = {1, 512, vec, nullptr, 0};
    ASSERT_POMAI_OK(pomai_put(db, &up));
    
    // Take Snapshot
    pomai_snapshot_t* snap = nullptr;
    ASSERT_POMAI_OK(pomai_get_snapshot(db, &snap));
    ASSERT_NE(snap, nullptr);
    
    // Create Iterator from snapshot
    pomai_iter_t* iter = nullptr;
    ASSERT_POMAI_OK(pomai_scan(db, snap, &iter));
    ASSERT_NE(iter, nullptr);
    
    // Add new item (should not be in snapshot)
    up.id = 2;
    ASSERT_POMAI_OK(pomai_put(db, &up));
    
    // Verify Iterator sees ID 1 only
    // Iterator starts invalid? Or valid if items exist?
    // Depends on implementation.
    // Usually iterator is positioned at start or first element.
    // Pomai::SnapshotIterator usually needs Seek or starts at beginning?
    // It has `Valid()`.
    
    // If it's `ShardIterator`, it iterates everything.
    // Wait, ShardIterator logic: constructor initializes it. `Valid()` false initially? 
    // Usually iterators start valid if not empty.
    
    // Let's check:
    bool found1 = false;
    bool found2 = false;
    
    // Step 22: class SnapshotIterator { virtual bool Next() ... }
    // It doesn't have `SeekToFirst()`. Does it start ready?
    // Or do I need to call `Next()` first?
    // Usually `Next` advances.
    
    // Re-check `ShardIterator` logic (Step 22/105).
    // It was not fully implemented in snippet.
    // Assuming standard iterator: Starts valid at first element?
    // Or starts "before first"?
    
    // If `Valid()` is true initially:
    while(pomai_iter_valid(iter)) {
        pomai_record_t rec;
        ASSERT_POMAI_OK(pomai_iter_get_record(iter, &rec));
        if (rec.id == 1) found1 = true;
        if (rec.id == 2) found2 = true;
        pomai_iter_next(iter);
    }
    
    EXPECT_TRUE(found1);
    EXPECT_FALSE(found2); // Snapshot isolation
    
    pomai_iter_free(iter);
    pomai_snapshot_free(snap);
}
