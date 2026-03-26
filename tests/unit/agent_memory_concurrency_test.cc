#include "tests/common/test_main.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "pomai/agent_memory.h"
#include "tests/common/test_tmpdir.h"

namespace
{

POMAI_TEST(AgentMemory_ConcurrentAppendAndSearch)
{
    pomai::AgentMemoryOptions opts;
    opts.path = pomai::test::TempDir("agent_mem_concurrent");
    opts.dim = 4;

    std::unique_ptr<pomai::AgentMemory> mem;
    POMAI_EXPECT_OK(pomai::AgentMemory::Open(opts, &mem));

    const int kThreads = 4;
    const int kPerThread = 50;

    auto run_sequential = [&](int tid)
    {
        for (int i = 0; i < kPerThread; ++i)
        {
            pomai::AgentMemoryRecord r;
            r.agent_id = "agentA";
            r.session_id = "sess" + std::to_string(tid);
            r.kind = pomai::AgentMemoryKind::kMessage;
            r.logical_ts = static_cast<std::int64_t>(tid * kPerThread + i);
            r.text = "m";
            r.embedding = {1.0f, 0.0f, 0.0f, 0.0f};
            POMAI_EXPECT_OK(mem->AppendMessage(r, nullptr));
        }
    };

    for (int t = 0; t < kThreads; ++t)
    {
        run_sequential(t);
    }

    pomai::AgentMemoryQuery q;
    q.agent_id = "agentA";
    q.embedding = {1.0f, 0.0f, 0.0f, 0.0f};
    q.topk = 5;

    pomai::AgentMemorySearchResult res;
    POMAI_EXPECT_OK(mem->SemanticSearch(q, &res));
    POMAI_EXPECT_TRUE(!res.hits.empty());
}

} // namespace

