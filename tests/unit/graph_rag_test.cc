#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/database.h"
#include "pomai/options.h"
#include "pomai/status.h"
#include <vector>
#include <string>

namespace pomai {

POMAI_TEST(VertexEdgeStorage) {
    std::string test_dir = test::TempDir("vertex_edge_storage");
    Database db;
    EmbeddedOptions opts;
    opts.path = test_dir;
    opts.dim = 4;
    
    POMAI_EXPECT_OK(db.Open(opts));

    // 1. Add Vertices
    POMAI_EXPECT_OK(db.AddVertex(1, 101, Metadata("tenant_a")));
    POMAI_EXPECT_OK(db.AddVertex(2, 101, Metadata("tenant_a")));
    POMAI_EXPECT_OK(db.AddVertex(3, 101, Metadata("tenant_a")));

    // 2. Add Edges (1 -> 2, 1 -> 3)
    POMAI_EXPECT_OK(db.AddEdge(1, 2, 1, 0, Metadata("link")));
    POMAI_EXPECT_OK(db.AddEdge(1, 3, 1, 0, Metadata("link")));

    // 3. Verify Neighbors
    std::vector<Neighbor> neighbors;
    POMAI_EXPECT_OK(db.GetNeighbors(1, &neighbors));
    POMAI_EXPECT_EQ(neighbors.size(), 2);
    
    bool found_2 = false, found_3 = false;
    for (const auto& n : neighbors) {
        if (n.id == 2) found_2 = true;
        if (n.id == 3) found_3 = true;
    }
    POMAI_EXPECT_TRUE(found_2);
    POMAI_EXPECT_TRUE(found_3);

    POMAI_EXPECT_OK(db.Close());
}

POMAI_TEST(SearchGraphRAGExpansion) {
    std::string test_dir = test::TempDir("search_graph_expansion");
    Database db;
    EmbeddedOptions opts;
    opts.path = test_dir;
    opts.dim = 4;
    POMAI_EXPECT_OK(db.Open(opts));

    // 1. Populate Vector Data
    std::vector<float> v1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v2 = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> v3 = {0.0f, 0.0f, 1.0f, 0.0f};
    
    POMAI_EXPECT_OK(db.AddVector(1, v1));
    POMAI_EXPECT_OK(db.AddVector(2, v2));
    POMAI_EXPECT_OK(db.AddVector(3, v3));

    // 2. Build Graph (1 -> 2 -> 3)
    POMAI_EXPECT_OK(db.AddEdge(1, 2, 1, 0, Metadata()));
    POMAI_EXPECT_OK(db.AddEdge(2, 3, 1, 0, Metadata()));

    // 3. Search GraphRAG (topk=1, k_hops=2)
    // Query near v1
    std::vector<float> query = {0.9f, 0.1f, 0.0f, 0.0f};
    std::vector<SearchResult> results;
    SearchOptions sopt;
    
    POMAI_EXPECT_OK(db.SearchGraphRAG(query, 1, sopt, 2, &results));
    POMAI_EXPECT_EQ(results.size(), 1);
    POMAI_EXPECT_EQ(results[0].hits.size(), 1);
    
    auto& hit = results[0].hits[0];
    POMAI_EXPECT_EQ(hit.id, 1);
    
    // Expect neighbors 2 and 3 in related_ids
    POMAI_EXPECT_EQ(hit.related_ids.size(), 2);
    bool found_2 = false, found_3 = false;
    for (auto rid : hit.related_ids) {
        if (rid == 2) found_2 = true;
        if (rid == 3) found_3 = true;
    }
    POMAI_EXPECT_TRUE(found_2);
    POMAI_EXPECT_TRUE(found_3);

    POMAI_EXPECT_OK(db.Close());
}

} // namespace pomai
