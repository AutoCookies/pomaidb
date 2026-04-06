#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include "core/graph/graph_membrane_impl.h"
#include "core/graph/graph_key.h"
#include "pomai/env.h"
#include "pomai/options.h"
#include "storage/wal/wal.h"

#include <cstddef>
#include <memory>
#include <string>

namespace pomai {
namespace {

std::unique_ptr<core::GraphMembraneImpl> MakeGraph(const std::string& dir) {
    auto* env = Env::Default();
    auto wal = std::make_unique<storage::Wal>(
        env, dir, /*shard_id=*/0, /*segment_bytes=*/1 << 20,
        FsyncPolicy::kNever);
    POMAI_EXPECT_OK(wal->Open());
    return std::make_unique<core::GraphMembraneImpl>(std::move(wal));
}

POMAI_TEST(GraphMemory_ZeroOnInit) {
    std::string dir = test::TempDir("graph_mem_zero");
    auto g = MakeGraph(dir);
    POMAI_EXPECT_EQ(g->MemoryBytesUsed(), 0u);
}

POMAI_TEST(GraphMemory_IncreasesOnAddVertex) {
    std::string dir = test::TempDir("graph_mem_vertex");
    auto g = MakeGraph(dir);

    std::size_t before = g->MemoryBytesUsed();
    POMAI_EXPECT_OK(g->AddVertex(1, 0, Metadata{}));
    std::size_t after = g->MemoryBytesUsed();
    POMAI_EXPECT_TRUE(after > before);

    // Adding the same vertex again should not double-count
    POMAI_EXPECT_OK(g->AddVertex(1, 0, Metadata{}));
    POMAI_EXPECT_EQ(g->MemoryBytesUsed(), after);

    // Adding a second distinct vertex increases further
    POMAI_EXPECT_OK(g->AddVertex(2, 0, Metadata{}));
    POMAI_EXPECT_TRUE(g->MemoryBytesUsed() > after);
}

POMAI_TEST(GraphMemory_IncreasesOnAddEdge) {
    std::string dir = test::TempDir("graph_mem_edge");
    auto g = MakeGraph(dir);

    POMAI_EXPECT_OK(g->AddVertex(1, 0, Metadata{}));
    POMAI_EXPECT_OK(g->AddVertex(2, 0, Metadata{}));
    std::size_t before = g->MemoryBytesUsed();

    POMAI_EXPECT_OK(g->AddEdge(1, 2, 1, 0, Metadata{}));
    POMAI_EXPECT_TRUE(g->MemoryBytesUsed() > before);
}

POMAI_TEST(GraphMemory_DecreasesOnDeleteVertex) {
    std::string dir = test::TempDir("graph_mem_del_vertex");
    auto g = MakeGraph(dir);

    POMAI_EXPECT_OK(g->AddVertex(1, 0, Metadata{}));
    POMAI_EXPECT_OK(g->AddEdge(1, 2, 1, 0, Metadata{}));
    POMAI_EXPECT_OK(g->AddEdge(1, 3, 1, 0, Metadata{}));
    std::size_t before = g->MemoryBytesUsed();

    POMAI_EXPECT_OK(g->DeleteVertex(1));
    POMAI_EXPECT_TRUE(g->MemoryBytesUsed() < before);
}

POMAI_TEST(GraphMemory_DecreasesOnDeleteEdge) {
    std::string dir = test::TempDir("graph_mem_del_edge");
    auto g = MakeGraph(dir);

    POMAI_EXPECT_OK(g->AddVertex(1, 0, Metadata{}));
    POMAI_EXPECT_OK(g->AddEdge(1, 2, 1, 0, Metadata{}));
    POMAI_EXPECT_OK(g->AddEdge(1, 3, 1, 0, Metadata{}));
    std::size_t before = g->MemoryBytesUsed();

    POMAI_EXPECT_OK(g->DeleteEdge(1, 2, 1));
    POMAI_EXPECT_TRUE(g->MemoryBytesUsed() < before);
}

POMAI_TEST(GraphMemory_DeleteNonexistentIsIdempotent) {
    std::string dir = test::TempDir("graph_mem_del_noop");
    auto g = MakeGraph(dir);

    // Deleting a vertex that was never added should not change accounting or error
    std::size_t before = g->MemoryBytesUsed();
    POMAI_EXPECT_OK(g->DeleteVertex(9999));
    POMAI_EXPECT_EQ(g->MemoryBytesUsed(), before);
}

} // namespace
} // namespace pomai
