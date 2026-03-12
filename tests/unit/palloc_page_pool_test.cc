#include "tests/common/test_main.h"

#include "palloc_page_pool.h"

#include <cstdint>
#include <cstring>

POMAI_TEST(PallocPagePool_CreateAndStats)
{
    palloc_page_pool_options opts{};
    opts.page_size = 4096;
    opts.capacity_bytes = 4096 * 4;  // 4 pages
    opts.swap_file_path = nullptr;   // swap disabled for this basic test
    opts.device_profile = PALLOC_DEVICE_PROFILE_SMALL;

    palloc_page_pool* pool = palloc_page_pool_create(&opts);
    POMAI_EXPECT_TRUE(pool != nullptr);

    palloc_page_pool_stats stats{};
    palloc_page_pool_get_stats(pool, &stats);
    POMAI_EXPECT_EQ(stats.page_size, opts.page_size);
    POMAI_EXPECT_EQ(stats.capacity_bytes, opts.capacity_bytes);

    palloc_page_pool_destroy(pool);
}

