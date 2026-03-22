#include "tests/common/test_main.h"
#include "pomai/database.h"
#include "pomai/options.h"
#include "pomai/metadata.h"
#include <vector>
#include <memory>
#include <filesystem>

namespace pomai {

POMAI_TEST(MultiModal_AutoEdgeHookCreatesSemanticLink) {
    std::string test_path = "/tmp/pomaidb_multi_modal_1";
    std::filesystem::remove_all(test_path);
    
    Database db;
    EmbeddedOptions opts;
    opts.path = test_path;
    opts.dim = 4;
    opts.enable_auto_edge = true;
    POMAI_EXPECT_OK(db.Open(opts));

    // 1. Add a "Topic" vertex (Vid: 100)
    POMAI_EXPECT_OK(db.AddVertex(100, 1 /* Tag */, Metadata{"tenant_1"}));

    // 2. Ingest a vector that "belongs" to Vid 100 via metadata
    std::vector<float> vec = {1.0f, 0.0f, 0.0f, 0.0f};
    Metadata meta{"tenant_1"};
    meta.src_vid = 100; // Trigger linkage
    
    VectorId vid = 500;
    POMAI_EXPECT_OK(db.AddVector(vid, vec, meta));

    // 3. Verify that an edge (500 -> 100) was created automatically
    std::vector<Neighbor> nb;
    POMAI_EXPECT_OK(db.GetNeighbors(500, &nb));
    
    bool found = false;
    for (const auto& n : nb) {
        if (n.id == 100) {
            found = true;
            break;
        }
    }
    POMAI_EXPECT_TRUE(found);

    (void)db.Close();
    std::filesystem::remove_all(test_path);
}

POMAI_TEST(MultiModal_AtomicSyncConsistencyCheck) {
    std::string test_path = "/tmp/pomaidb_multi_modal_2";
    std::filesystem::remove_all(test_path);
    
    Database db;
    EmbeddedOptions opts;
    opts.path = test_path;
    opts.dim = 4;
    opts.enable_auto_edge = true;
    POMAI_EXPECT_OK(db.Open(opts));

    // Ingest with hook
    POMAI_EXPECT_OK(db.AddVertex(200, 1, Metadata{"t1"}));
    std::vector<float> vec = {0.0f, 1.0f, 0.0f, 0.0f};
    Metadata meta{"t1"};
    meta.src_vid = 200;
    POMAI_EXPECT_OK(db.AddVector(600, vec, meta));

    // Force flush to ensure WAL is persisted
    POMAI_EXPECT_OK(db.Flush());
    
    std::vector<Neighbor> nb;
    POMAI_EXPECT_OK(db.GetNeighbors(600, &nb));
    POMAI_EXPECT_EQ(nb.size(), 1u);

    (void)db.Close();
    std::filesystem::remove_all(test_path);
}

POMAI_TEST(MultiModal_UnifiedSearchOrchestration) {
    std::string test_path = "/tmp/pomaidb_multi_modal_3";
    std::filesystem::remove_all(test_path);
    
    Database db;
    EmbeddedOptions opts;
    opts.path = test_path;
    opts.dim = 4;
    opts.enable_auto_edge = true;
    POMAI_EXPECT_OK(db.Open(opts));

    // Ingest data
    POMAI_EXPECT_OK(db.AddVertex(300, 1, Metadata{"t1"}));
    std::vector<float> vec = {1.0f, 1.0f, 0.0f, 0.0f};
    Metadata meta{"t1"};
    meta.src_vid = 300;
    POMAI_EXPECT_OK(db.AddVector(700, vec, meta));
    
    // 2-hop neighbor of 700
    POMAI_EXPECT_OK(db.AddVertex(800, 1, Metadata{"t1"}));
    POMAI_EXPECT_OK(db.AddEdge(700, 800, 1, 1, Metadata{"t1"}));

    POMAI_EXPECT_OK(db.Flush());
    POMAI_EXPECT_OK(db.Freeze()); // Build index

    // Unified Query
    MultiModalQuery mmq;
    mmq.vector = {1.0f, 1.0f, 0.05f, 0.0f};
    mmq.top_k = 1;
    mmq.graph_hops = 2;
    
    SearchResult res;
    POMAI_EXPECT_OK(db.SearchMultiModal(mmq, &res));
    
    POMAI_EXPECT_EQ(res.hits.size(), 1u);
    POMAI_EXPECT_EQ(res.hits[0].id, 700u);
    
    // Should have 300 (src_vid) and 800 (manual edge) in related_ids
    bool found_300 = false;
    bool found_800 = false;
    for (auto rid : res.hits[0].related_ids) {
        if (rid == 300) found_300 = true;
        if (rid == 800) found_800 = true;
    }
    POMAI_EXPECT_TRUE(found_300);
    POMAI_EXPECT_TRUE(found_800);

    (void)db.Close();
    std::filesystem::remove_all(test_path);
}

} // namespace pomai
