#include "tests/common/test_main.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "pomai/agent_memory.h"
#include "tests/common/test_tmpdir.h"

namespace
{

POMAI_TEST(AgentMemory_BasicAppendAndRecent)
{
    pomai::AgentMemoryOptions opts;
    opts.path = pomai::test::TempDir("agent_mem_basic");
    opts.dim = 4;

    std::unique_ptr<pomai::AgentMemory> mem;
    POMAI_EXPECT_OK(pomai::AgentMemory::Open(opts, &mem));

    pomai::AgentMemoryRecord r1;
    r1.agent_id = "agentA";
    r1.session_id = "sess1";
    r1.kind = pomai::AgentMemoryKind::kMessage;
    r1.logical_ts = 1;
    r1.text = "hello";
    r1.embedding = {0.1f, 0.2f, 0.3f, 0.4f};

    pomai::AgentMemoryRecord r2 = r1;
    r2.logical_ts = 2;
    r2.text = "world";

    pomai::VectorId id1 = 0, id2 = 0;
    POMAI_EXPECT_OK(mem->AppendMessage(r1, &id1));
    POMAI_EXPECT_OK(mem->AppendMessage(r2, &id2));

    std::vector<pomai::AgentMemoryRecord> recent;
    POMAI_EXPECT_OK(mem->GetRecent("agentA", "sess1", 10, &recent));
    POMAI_EXPECT_EQ(recent.size(), 2u);
    POMAI_EXPECT_EQ(recent[0].text, "hello");
    POMAI_EXPECT_EQ(recent[1].text, "world");
}

POMAI_TEST(AgentMemory_SemanticSearchBasic)
{
    pomai::AgentMemoryOptions opts;
    opts.path = pomai::test::TempDir("agent_mem_search");
    opts.dim = 2;

    std::unique_ptr<pomai::AgentMemory> mem;
    POMAI_EXPECT_OK(pomai::AgentMemory::Open(opts, &mem));

    // Two distinct directions in 2D
    pomai::AgentMemoryRecord r1;
    r1.agent_id = "agentA";
    r1.session_id = "sess1";
    r1.kind = pomai::AgentMemoryKind::kMessage;
    r1.logical_ts = 1;
    r1.text = "east";
    r1.embedding = {1.0f, 0.0f};

    pomai::AgentMemoryRecord r2 = r1;
    r2.logical_ts = 2;
    r2.text = "north";
    r2.embedding = {0.0f, 1.0f};

    POMAI_EXPECT_OK(mem->AppendMessage(r1, nullptr));
    POMAI_EXPECT_OK(mem->AppendMessage(r2, nullptr));

    pomai::AgentMemoryQuery q;
    q.agent_id = "agentA";
    q.embedding = {1.0f, 0.0f};
    q.topk = 1;

    pomai::AgentMemorySearchResult res;
    POMAI_EXPECT_OK(mem->SemanticSearch(q, &res));
    POMAI_EXPECT_EQ(res.hits.size(), 1u);
    POMAI_EXPECT_EQ(res.hits[0].record.text, "east");
}

POMAI_TEST(AgentMemory_PruneOldRespectsKeepLast)
{
    pomai::AgentMemoryOptions opts;
    opts.path = pomai::test::TempDir("agent_mem_prune");
    opts.dim = 2;

    std::unique_ptr<pomai::AgentMemory> mem;
    POMAI_EXPECT_OK(pomai::AgentMemory::Open(opts, &mem));

    for (int i = 0; i < 5; ++i)
    {
        pomai::AgentMemoryRecord r;
        r.agent_id = "agentA";
        r.session_id = "sess1";
        r.kind = pomai::AgentMemoryKind::kMessage;
        r.logical_ts = i;
        r.text = "m" + std::to_string(i);
        r.embedding = {1.0f, 0.0f};
        POMAI_EXPECT_OK(mem->AppendMessage(r, nullptr));
    }

    POMAI_EXPECT_OK(mem->PruneOld("agentA", /*keep_last_n=*/2, /*min_ts_to_keep=*/0));

    std::vector<pomai::AgentMemoryRecord> recent;
    POMAI_EXPECT_OK(mem->GetRecent("agentA", "sess1", 10, &recent));
    POMAI_EXPECT_EQ(recent.size(), 2u);
    POMAI_EXPECT_EQ(recent[0].logical_ts, 3);
    POMAI_EXPECT_EQ(recent[1].logical_ts, 4);
}

} // namespace

