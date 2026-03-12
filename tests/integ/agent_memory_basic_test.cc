#include "tests/common/test_main.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "pomai/agent_memory.h"
#include "tests/common/test_tmpdir.h"

namespace
{

POMAI_TEST(AgentMemory_PersistenceAcrossRestart)
{
    const std::string path = pomai::test::TempDir("agent_mem_persist");

    {
        pomai::AgentMemoryOptions opts;
        opts.path = path;
        opts.dim = 3;

        std::unique_ptr<pomai::AgentMemory> mem;
        POMAI_EXPECT_OK(pomai::AgentMemory::Open(opts, &mem));

        pomai::AgentMemoryRecord r;
        r.agent_id = "agentA";
        r.session_id = "sess1";
        r.kind = pomai::AgentMemoryKind::kMessage;
        r.logical_ts = 42;
        r.text = "persisted";
        r.embedding = {0.5f, 0.1f, 0.2f};
        POMAI_EXPECT_OK(mem->AppendMessage(r, nullptr));
    }

    {
        pomai::AgentMemoryOptions opts;
        opts.path = path;
        opts.dim = 3;

        std::unique_ptr<pomai::AgentMemory> mem;
        POMAI_EXPECT_OK(pomai::AgentMemory::Open(opts, &mem));

        std::vector<pomai::AgentMemoryRecord> recent;
        POMAI_EXPECT_OK(mem->GetRecent("agentA", "sess1", 10, &recent));
        POMAI_EXPECT_EQ(recent.size(), 1u);
        POMAI_EXPECT_EQ(recent[0].text, "persisted");
        POMAI_EXPECT_EQ(recent[0].logical_ts, 42);
    }
}

} // namespace

