#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "core/membrane/manager.h"
#include "pomai/options.h"
#include "pomai/status.h"
#include "pomai/types.h"

#include <algorithm>
#include <string>
#include <vector>

namespace pomai {
namespace {

DBOptions MakeOpts(const std::string& path) {
    DBOptions opt;
    opt.path = path;
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kAlways;
    return opt;
}

void CreateGraphMembrane(core::MembraneManager& mgr, const std::string& name) {
    MembraneSpec spec;
    spec.name = name;
    spec.kind = MembraneKind::kGraph;
    spec.dim = 4;
    spec.shard_count = 1;
    POMAI_EXPECT_OK(mgr.CreateMembrane(spec));
    POMAI_EXPECT_OK(mgr.OpenMembrane(name));
}

bool HasNeighbor(const std::vector<Neighbor>& neighbors, VertexId id) {
    return std::any_of(neighbors.begin(), neighbors.end(),
                       [id](const Neighbor& n) { return n.id == id; });
}

// After close+reopen, GetNeighbors must return what was written before close.
POMAI_TEST(GraphPersistence_ReplayAfterRestart) {
    std::string dir = test::TempDir("graph_persist_replay");

    // Write phase
    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());
        CreateGraphMembrane(mgr, "g");

        POMAI_EXPECT_OK(mgr.AddVertex("g", 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddVertex("g", 2, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddVertex("g", 3, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddEdge("g", 1, 2, 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddEdge("g", 1, 3, 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.FlushAll());
        POMAI_EXPECT_OK(mgr.Close());
    }

    // Read phase — reopen and verify adjacency was restored
    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());

        std::vector<Neighbor> neighbors;
        POMAI_EXPECT_OK(mgr.GetNeighbors("g", 1, &neighbors));
        POMAI_EXPECT_EQ(neighbors.size(), 2u);
        POMAI_EXPECT_TRUE(HasNeighbor(neighbors, 2));
        POMAI_EXPECT_TRUE(HasNeighbor(neighbors, 3));

        // Vertices with no outgoing edges should still exist (empty neighbor list)
        std::vector<Neighbor> n2;
        POMAI_EXPECT_OK(mgr.GetNeighbors("g", 2, &n2));
        POMAI_EXPECT_EQ(n2.size(), 0u);

        POMAI_EXPECT_OK(mgr.Close());
    }
}

// A→B→C chain must survive restart end-to-end.
POMAI_TEST(GraphPersistence_MultiHopChainSurvivesRestart) {
    std::string dir = test::TempDir("graph_persist_multihop");

    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());
        CreateGraphMembrane(mgr, "g");

        POMAI_EXPECT_OK(mgr.AddVertex("g", 10, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddVertex("g", 20, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddVertex("g", 30, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddVertex("g", 40, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddEdge("g", 10, 20, 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddEdge("g", 20, 30, 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddEdge("g", 30, 40, 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.FlushAll());
        POMAI_EXPECT_OK(mgr.Close());
    }

    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());

        std::vector<Neighbor> n;
        POMAI_EXPECT_OK(mgr.GetNeighbors("g", 10, &n));
        POMAI_EXPECT_EQ(n.size(), 1u);
        POMAI_EXPECT_TRUE(HasNeighbor(n, 20));

        n.clear();
        POMAI_EXPECT_OK(mgr.GetNeighbors("g", 20, &n));
        POMAI_EXPECT_EQ(n.size(), 1u);
        POMAI_EXPECT_TRUE(HasNeighbor(n, 30));

        n.clear();
        POMAI_EXPECT_OK(mgr.GetNeighbors("g", 30, &n));
        POMAI_EXPECT_EQ(n.size(), 1u);
        POMAI_EXPECT_TRUE(HasNeighbor(n, 40));

        POMAI_EXPECT_OK(mgr.Close());
    }
}

// A deleted edge must remain deleted after restart.
POMAI_TEST(GraphPersistence_DeleteEdgeIsReplayed) {
    std::string dir = test::TempDir("graph_persist_delete_edge");

    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());
        CreateGraphMembrane(mgr, "g");

        POMAI_EXPECT_OK(mgr.AddVertex("g", 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddVertex("g", 2, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddVertex("g", 3, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddEdge("g", 1, 2, 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddEdge("g", 1, 3, 1, 0, Metadata{}));
        // Delete the edge 1→2
        POMAI_EXPECT_OK(mgr.DeleteEdge("g", 1, 2, 1));
        POMAI_EXPECT_OK(mgr.FlushAll());
        POMAI_EXPECT_OK(mgr.Close());
    }

    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());

        std::vector<Neighbor> neighbors;
        POMAI_EXPECT_OK(mgr.GetNeighbors("g", 1, &neighbors));
        // Only edge 1→3 should survive
        POMAI_EXPECT_EQ(neighbors.size(), 1u);
        POMAI_EXPECT_TRUE(HasNeighbor(neighbors, 3));
        POMAI_EXPECT_TRUE(!HasNeighbor(neighbors, 2));

        POMAI_EXPECT_OK(mgr.Close());
    }
}

// A deleted vertex must remain deleted after restart.
POMAI_TEST(GraphPersistence_DeleteVertexIsReplayed) {
    std::string dir = test::TempDir("graph_persist_delete_vertex");

    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());
        CreateGraphMembrane(mgr, "g");

        POMAI_EXPECT_OK(mgr.AddVertex("g", 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddVertex("g", 2, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.AddEdge("g", 1, 2, 1, 0, Metadata{}));
        POMAI_EXPECT_OK(mgr.DeleteVertex("g", 1));
        POMAI_EXPECT_OK(mgr.FlushAll());
        POMAI_EXPECT_OK(mgr.Close());
    }

    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());

        // Vertex 1 was deleted — GetNeighbors should return empty (not error)
        std::vector<Neighbor> neighbors;
        POMAI_EXPECT_OK(mgr.GetNeighbors("g", 1, &neighbors));
        POMAI_EXPECT_EQ(neighbors.size(), 0u);

        POMAI_EXPECT_OK(mgr.Close());
    }
}

// WarmUp on an empty (freshly created) graph must be a no-op.
POMAI_TEST(GraphPersistence_EmptyWarmUpIsNoop) {
    std::string dir = test::TempDir("graph_persist_empty");

    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());
        CreateGraphMembrane(mgr, "g");
        POMAI_EXPECT_OK(mgr.Close());
    }

    {
        auto opt = MakeOpts(dir);
        core::MembraneManager mgr(opt);
        POMAI_EXPECT_OK(mgr.Open());

        std::vector<Neighbor> neighbors;
        POMAI_EXPECT_OK(mgr.GetNeighbors("g", 999, &neighbors));
        POMAI_EXPECT_EQ(neighbors.size(), 0u);

        POMAI_EXPECT_OK(mgr.Close());
    }
}

} // namespace
} // namespace pomai
