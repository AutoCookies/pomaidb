#include "tests/common/test_main.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "pomai/c_api.h"
#include "tests/common/test_tmpdir.h"

namespace
{

POMAI_TEST(AgentMemory_CApi_BasicFlow)
{
    std::string path = pomai::test::TempDir("agent_mem_capi");

    pomai_agent_memory_options_t opts;
    opts.struct_size = static_cast<uint32_t>(sizeof(pomai_agent_memory_options_t));
    opts.path = path.c_str();
    opts.dim = 4;
    opts.metric = 0;
    opts.max_messages_per_agent = 0;
    opts.max_device_bytes = 0;

    pomai_agent_memory_t* mem = nullptr;
    pomai_status_t* st = pomai_agent_memory_open(&opts, &mem);
    POMAI_EXPECT_TRUE(st == nullptr);

    pomai_agent_memory_record_t rec{};
    rec.struct_size = static_cast<uint32_t>(sizeof(pomai_agent_memory_record_t));
    rec.agent_id = "agentA";
    rec.session_id = "sess1";
    rec.kind = "message";
    rec.logical_ts = 1;
    rec.text = "hello";
    float vec[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    rec.embedding = vec;
    rec.dim = 4;

    uint64_t id = 0;
    st = pomai_agent_memory_append(mem, &rec, &id);
    POMAI_EXPECT_TRUE(st == nullptr);

    pomai_agent_memory_result_set_t* recent = nullptr;
    st = pomai_agent_memory_get_recent(mem, "agentA", "sess1", 10, &recent);
    POMAI_EXPECT_TRUE(st == nullptr);
    POMAI_EXPECT_TRUE(recent != nullptr);
    POMAI_EXPECT_EQ(recent->count, 1u);
    if (recent->count == 1u)
    {
        POMAI_EXPECT_EQ(std::string(recent->records[0].text), "hello");
    }
    pomai_agent_memory_result_set_free(recent);

    pomai_agent_memory_query_t q{};
    q.struct_size = static_cast<uint32_t>(sizeof(pomai_agent_memory_query_t));
    q.agent_id = "agentA";
    q.session_id = "sess1";
    q.kind = "message";
    q.min_ts = 0;
    q.max_ts = 10;
    q.embedding = vec;
    q.dim = 4;
    q.topk = 1;

    pomai_agent_memory_search_result_t* search_res = nullptr;
    st = pomai_agent_memory_search(mem, &q, &search_res);
    POMAI_EXPECT_TRUE(st == nullptr);
    POMAI_EXPECT_TRUE(search_res != nullptr);
    POMAI_EXPECT_EQ(search_res->count, 1u);
    if (search_res->count == 1u)
    {
        POMAI_EXPECT_EQ(std::string(search_res->records[0].text), "hello");
    }
    pomai_agent_memory_search_result_free(search_res);

    st = pomai_agent_memory_close(mem);
    POMAI_EXPECT_TRUE(st == nullptr);
}

} // namespace

