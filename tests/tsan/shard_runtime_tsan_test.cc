#include "tests/common/test_main.h"
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "core/shard/runtime.h"
#include "pomai/options.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"
#include "tests/common/test_tmpdir.h"

POMAI_TEST(ShardRuntime_ActorSerializesCommands_TSAN)
{
    constexpr std::uint32_t kDim = 8;
    auto dir = pomai::test::TempDir("pomai-tsan-shardrt");

    auto wal = std::make_unique<pomai::storage::Wal>(
        dir, /*shard_id*/ 0, /*wal_segment_bytes*/ (1u << 20),
        /*fsync*/ pomai::FsyncPolicy::kNever);

    POMAI_EXPECT_OK(wal->Open());

    auto mem = std::make_unique<pomai::table::MemTable>(kDim, (1u << 20));
    POMAI_EXPECT_OK(wal->ReplayInto(*mem));

    pomai::core::ShardRuntime rt(0, kDim, std::move(wal), std::move(mem), /*mailbox*/ 1u << 16);
    POMAI_EXPECT_OK(rt.Start());

    constexpr int kThreads = 6;
    constexpr int kOpsPer = 3000;

    std::vector<std::jthread> th;
    th.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t)
    {
        th.emplace_back([&, t]
                        {
      for (int i = 0; i < kOpsPer; ++i) {
        std::vector<float> v(kDim, 0.0f);
        v[static_cast<std::size_t>((t + i) % kDim)] = 1.0f;

        pomai::core::PutCmd c;
        c.id = static_cast<pomai::VectorId>(1ull + static_cast<std::uint64_t>(t) * kOpsPer + i);
        c.vec = v.data();
        c.dim = kDim;

        auto fut = c.done.get_future();
        POMAI_EXPECT_OK(rt.Enqueue(pomai::core::Command{std::move(c)}));
        POMAI_EXPECT_OK(fut.get());
      } });
    }

    th.clear();

    // If TSAN is happy here, it means: mailbox + actor thread + WAL+memtable interactions are race-clean.
    // Destructor of rt will Stop() safely.
}
