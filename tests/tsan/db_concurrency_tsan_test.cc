#include "tests/common/test_main.h"
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "pomai/options.h"
#include "pomai/pomai.h"
#include "pomai/search.h"
#include "tests/common/db_test_util.h"
#include "tests/common/test_random.h"
#include "tests/common/test_tmpdir.h"

POMAI_TEST(DB_ConcurrentPutDeleteSearch_TSAN)
{
  pomai::DBOptions opt;
  opt.path = pomai::test::TempDir("pomai-tsan-db");
  opt.dim = 16;
  opt.shard_count = 4;

  auto db = pomai::test::OpenDB(opt);
  POMAI_EXPECT_TRUE(db != nullptr);

  std::atomic<bool> stop{false};

  std::jthread writer([&]
                      {
    pomai::test::Rng rng(123);
    for (std::uint64_t i = 1; i <= 20000; ++i) {
      auto v = rng.Vec(opt.dim);
      (void)db->Put(i, v);
      if ((i % 7) == 0) (void)db->Delete(i - 3);
    }
    stop.store(true, std::memory_order_release); });

  std::jthread reader([&]
                      {
    pomai::test::Rng rng(999);
    while (!stop.load(std::memory_order_acquire)) {
      auto q = rng.Vec(opt.dim);
      pomai::SearchResult out;
      (void)db->Search(q, 10, &out);
    } });

  writer.join();
  reader.join();

  POMAI_EXPECT_OK(db->Flush());
  POMAI_EXPECT_OK(db->Close());
}
