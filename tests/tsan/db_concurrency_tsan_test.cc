#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "pomai/options.h"
#include "pomai/pomai.h"
#include "pomai/search.h"

namespace
{

  std::vector<float> MakeVec(std::uint32_t dim, float seed)
  {
    std::vector<float> v(dim);
    for (std::uint32_t i = 0; i < dim; ++i)
      v[i] = seed + static_cast<float>(i) * 0.0001f;
    return v;
  }

  POMAI_TEST(DB_TSAN_ConcurrentPutDeleteSearch)
  {
    pomai::DBOptions opt;
    opt.path = pomai::test::TempDir("pomai-db_concurrency_tsan_test");
    opt.dim = 32;
    opt.shard_count = 4;
    opt.fsync = pomai::FsyncPolicy::kNever; // tsan: tránh fsync nhiễu

    std::unique_ptr<pomai::DB> db;
    POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

    constexpr int kThreads = 6;
    constexpr int kOpsPerThread = 2000;

    std::atomic<bool> start{false};
    std::vector<std::jthread> th;
    th.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t)
    {
      th.emplace_back([&, t]
                      {
                while (!start.load(std::memory_order_acquire)) {}

                for (int i = 0; i < kOpsPerThread; ++i)
                {
                    const auto id = static_cast<pomai::VectorId>(t * 1'000'000 + i);
                    auto v = MakeVec(opt.dim, static_cast<float>(id % 1000) * 0.01f);

                    if ((i % 7) == 0)
                        (void)db->Delete(id);
                    else
                        (void)db->Put(id, v);

                    if ((i % 11) == 0)
                    {
                        pomai::SearchResult r;
                        (void)db->Search(v, /*topk*/ 5, &r);
                        POMAI_EXPECT_TRUE(r.hits.size() <= 5);
                    }
                } });
    }

    start.store(true, std::memory_order_release);
    for (auto &t : th)
      t.join();

    POMAI_EXPECT_OK(db->Flush());
    POMAI_EXPECT_OK(db->Close());
  }

} // namespace
