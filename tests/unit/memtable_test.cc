#include "tests/common/test_main.h"
#include <cstdint>
#include <vector>

#include "pomai/status.h"
#include "pomai/types.h"
#include "table/memtable.h"

POMAI_TEST(MemTable_PutDeleteForEach)
{
    constexpr std::uint32_t kDim = 4;
    pomai::table::MemTable mem(kDim, /*arena_block_bytes*/ 1u << 20);

    std::vector<float> v1 = {1, 2, 3, 4};
    std::vector<float> v2 = {4, 3, 2, 1};

    POMAI_EXPECT_OK(mem.Put(10, v1));
    POMAI_EXPECT_OK(mem.Put(20, v2));
    POMAI_EXPECT_OK(mem.Delete(10));

    std::size_t seen = 0;
    pomai::VectorId only_id = 0;

    mem.ForEach([&](pomai::VectorId id, std::span<const float> vec)
                {
    ++seen;
    only_id = id;
    POMAI_EXPECT_EQ(vec.size(), kDim); });

    POMAI_EXPECT_EQ(seen, 1u);
    POMAI_EXPECT_EQ(only_id, 20u);
}
