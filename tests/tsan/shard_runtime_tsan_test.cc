#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "core/shard/runtime.h"
#include "pomai/options.h"
#include "pomai/search.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"

namespace
{

  std::vector<float> MakeVec(std::uint32_t dim, float base)
  {
    std::vector<float> v(dim);
    for (std::uint32_t i = 0; i < dim; ++i)
      v[i] = base + static_cast<float>(i) * 0.001f;
    return v;
  }

  POMAI_TEST(ShardRuntime_TSAN_ActorSerializesCommands)
  {
    const std::uint32_t dim = 32;
    const std::uint32_t shard_id = 0;

    const std::string path = pomai::test::TempDir("pomai-shard_runtime_tsan_test");

    auto wal = std::make_unique<pomai::storage::Wal>(
        path, shard_id,
        /*wal_segment_bytes*/ (1u << 20),
        /*fsync*/ pomai::FsyncPolicy::kNever);

    POMAI_EXPECT_OK(wal->Open());

    auto mem = std::make_unique<pomai::table::MemTable>(dim, /*arena_block_bytes*/ (1u << 20));

    // replay nếu có dữ liệu cũ (test idempotent)
    POMAI_EXPECT_OK(wal->ReplayInto(*mem));

    pomai::core::ShardRuntime rt(shard_id, path, dim, std::move(wal), std::move(mem), /*mailbox_cap*/ (1u << 14), pomai::IndexParams{});
    POMAI_EXPECT_OK(rt.Start());

    constexpr int kThreads = 4;
    constexpr int kOps = 2000;

    std::vector<std::jthread> th;
    th.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t)
    {
      th.emplace_back([&, t]
                      {
                for (int i = 0; i < kOps; ++i)
                {
                    const pomai::VectorId id = static_cast<pomai::VectorId>(t * 1'000'000 + i);
                    auto v = MakeVec(dim, static_cast<float>(id % 1000) * 0.01f);

                    pomai::core::PutCmd c;
                    c.id = id;
                    c.vec = pomai::VectorView(v);

                    auto fut = c.done.get_future();
                    POMAI_EXPECT_OK(rt.Enqueue(pomai::core::Command{std::move(c)}));
                    POMAI_EXPECT_OK(fut.get());
                } });
    }

    for (auto &t : th)
      t.join();

    // Search sanity
    auto q = MakeVec(dim, 0.0f);
    std::vector<pomai::SearchHit> out;
    POMAI_EXPECT_OK(rt.Search(q, /*topk*/ 10, &out));
    POMAI_EXPECT_TRUE(out.size() <= 10);
  }

} // namespace
