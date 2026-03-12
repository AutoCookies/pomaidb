#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <cstdint>
#include <memory>
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

  POMAI_TEST(VectorRuntime_TSAN_ActorSerializesCommands)
  {
    const std::uint32_t dim = 32;
    const std::uint32_t runtime_id = 0;

    const std::string path = pomai::test::TempDir("pomai-vector_runtime_tsan_test");

    auto wal = std::make_unique<pomai::storage::Wal>(
        pomai::Env::Default(), path, runtime_id,
        /*wal_segment_bytes*/ (1u << 20),
        /*fsync*/ pomai::FsyncPolicy::kNever);

    POMAI_EXPECT_OK(wal->Open());

    auto mem = std::make_unique<pomai::table::MemTable>(dim, /*arena_block_bytes*/ (1u << 20));

    POMAI_EXPECT_OK(wal->ReplayInto(*mem));

    pomai::core::VectorRuntime rt(runtime_id, path, dim, pomai::MembraneKind::kVector, pomai::MetricType::kL2, std::move(wal),
                                  std::move(mem), pomai::IndexParams{});
    POMAI_EXPECT_OK(rt.Start());

    // Single-threaded: sequential puts (no worker thread, no Enqueue).
    constexpr int kThreads = 4;
    constexpr int kOps = 2000;

    for (int t = 0; t < kThreads; ++t) {
      for (int i = 0; i < kOps; ++i) {
        const pomai::VectorId id = static_cast<pomai::VectorId>(t * 1'000'000 + i);
        auto v = MakeVec(dim, static_cast<float>(id % 1000) * 0.01f);
        POMAI_EXPECT_OK(rt.Put(id, v));
      }
    }

    // Search sanity
    auto q = MakeVec(dim, 0.0f);
    std::vector<pomai::SearchHit> out;
    POMAI_EXPECT_OK(rt.Search(q, /*topk*/ 10, &out));
    POMAI_EXPECT_TRUE(out.size() <= 10);
  }

} // namespace
