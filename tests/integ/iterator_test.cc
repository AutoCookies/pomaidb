#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include "pomai/iterator.h"
#include <memory>
#include <vector>
#include <unordered_set>
#include <algorithm>

namespace {

using namespace pomai;

static std::vector<float> MakeVec(std::uint32_t dim, float val) {
    return std::vector<float>(dim, val);
}

// Test basic iteration over all inserted vectors
POMAI_TEST(Iterator_AllInsertedIDs) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("iterator_all_ids");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));
    
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 1;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Insert 100 vectors
    std::unordered_set<VectorId> inserted_ids;
    for (std::uint32_t i = 0; i < 100; ++i) {
        VectorId id = static_cast<VectorId>(i);
        auto v = MakeVec(4, static_cast<float>(i));
        POMAI_EXPECT_OK(db->Put("default", id, v));
        inserted_ids.insert(id);
    }
    
    // Freeze to make data visible to iterator
    db->Freeze("default");

    // Iterate and collect IDs
    std::unique_ptr<SnapshotIterator> it;
    POMAI_EXPECT_OK(db->NewIterator("default", &it));
    
    std::unordered_set<VectorId> iterated_ids;
    while (it->Valid()) {
        iterated_ids.insert(it->id());
        it->Next();
    }
    
    // Verify all inserted IDs were iterated
    POMAI_EXPECT_EQ(iterated_ids.size(), inserted_ids.size());
    for (VectorId id : inserted_ids) {
        POMAI_EXPECT_TRUE(iterated_ids.count(id) > 0);
    }
    
    db->Close();
}

// Test that tombstones (deleted vectors) are not returned
POMAI_TEST(Iterator_TombstoneFiltering) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("iterator_tombstones");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 1;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Insert vectors
    for (std::uint32_t i = 0; i < 50; ++i) {
        auto v = MakeVec(4, static_cast<float>(i));
        db->Put("default", static_cast<VectorId>(i), v);
    }
    
    // Delete every other vector
    std::unordered_set<VectorId> deleted_ids;
    for (std::uint32_t i = 0; i < 50; i += 2) {
        db->Delete("default", static_cast<VectorId>(i));
        deleted_ids.insert(static_cast<VectorId>(i));
    }
    
    db->Freeze("default");

    // Iterate
    std::unique_ptr<SnapshotIterator> it;
    db->NewIterator("default", &it);
    
    std::unordered_set<VectorId> iterated_ids;
    while (it->Valid()) {
        VectorId id = it->id();
        iterated_ids.insert(id);
        
        // Verify this ID was not deleted
        POMAI_EXPECT_TRUE(deleted_ids.count(id) == 0);
        
        it->Next();
    }
    
    // Should have 25 live vectors (half were deleted)
    POMAI_EXPECT_EQ(iterated_ids.size(), 25u);
    
    db->Close();
}

// Test that iterator returns newest version for duplicate IDs
POMAI_TEST(Iterator_NewestWinsDeduplication) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("iterator_dedup");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 1;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Insert vector with value 1.0
    VectorId id = 42;
    auto v1 = MakeVec(4, 1.0f);
    db->Put("default", id, v1);
    db->Freeze("default");
    
    // Update same ID with value 2.0
    auto v2 = MakeVec(4, 2.0f);
    db->Put("default", id, v2);
    db->Freeze("default");

    // Iterator should return only newest version (2.0)
    std::unique_ptr<SnapshotIterator> it;
    db->NewIterator("default", &it);
    
    int count = 0;
    while (it->Valid()) {
        if (it->id() == id) {
            auto vec = it->vector();
            POMAI_EXPECT_EQ(vec.size(), 4u);
            // Should be newest version (2.0)
            POMAI_EXPECT_EQ(vec[0], 2.0f);
            count++;
        }
        it->Next();
    }
    
    // Should see ID exactly once
    POMAI_EXPECT_EQ(count, 1);
    
    db->Close();
}

// Test snapshot isolation: iterator sees state at snapshot time
POMAI_TEST(Iterator_SnapshotIsolation) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("iterator_snapshot_isolation");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 1;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Insert initial data
    for (std::uint32_t i = 0; i < 50; ++i) {
        auto v = MakeVec(4, 1.0f);
        db->Put("default", static_cast<VectorId>(i), v);
    }
    db->Freeze("default");

    // Create iterator (captures snapshot)
    std::unique_ptr<SnapshotIterator> it;
    db->NewIterator("default", &it);
    
    // Insert more data after iterator creation
    for (std::uint32_t i = 100; i < 150; ++i) {
        auto v = MakeVec(4, 2.0f);
        db->Put("default", static_cast<VectorId>(i), v);
    }
    db->Freeze("default");

    // Iterator should NOT see data inserted after snapshot
    std::unordered_set<VectorId> iterated_ids;
    while (it->Valid()) {
        VectorId id = it->id();
        iterated_ids.insert(id);
        
        // Should be from original range [0, 50)
        POMAI_EXPECT_TRUE(id < static_cast<VectorId>(50));
        
        it->Next();
    }
    
    // Should have exactly 50 vectors (not 100)
    POMAI_EXPECT_EQ(iterated_ids.size(), 50u);
    
    db->Close();
}

// Test deterministic ordering: iterator should return IDs in consistent order
POMAI_TEST(Iterator_DeterministicOrdering) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("iterator_deterministic");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    DB::Open(opt, &db);
    MembraneSpec spec;
    spec.name = "default";
    spec.dim = 4;
    spec.shard_count = 1;
    db->CreateMembrane(spec);
    db->OpenMembrane("default");

    // Insert data
    for (std::uint32_t i = 0; i < 30; ++i) {
        auto v = MakeVec(4, static_cast<float>(i));
        db->Put("default", static_cast<VectorId>(i), v);
    }
    db->Freeze("default");

    // Iterate twice and collect order
    std::vector<VectorId> order1, order2;
    
    {
        std::unique_ptr<SnapshotIterator> it;
        db->NewIterator("default", &it);
        while (it->Valid()) {
            order1.push_back(it->id());
            it->Next();
        }
    }
    
    {
        std::unique_ptr<SnapshotIterator> it;
        db->NewIterator("default", &it);
        while (it->Valid()) {
            order2.push_back(it->id());
            it->Next();
        }
    }
    
    // Orders should be identical
    POMAI_EXPECT_EQ(order1.size(), order2.size());
    for (size_t i = 0; i < order1.size(); ++i) {
        POMAI_EXPECT_EQ(order1[i], order2[i]);
    }
    
    db->Close();
}

} // namespace
