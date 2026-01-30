#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "common/test_utils.h"

#include <thread>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Concurrent ingest", "[concurrency][ingest]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 8, 2);
    PomaiDB db(opts);
    db.Start();

    constexpr std::size_t per_thread = 20;
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < 4; ++t)
    {
        threads.emplace_back([&, t]()
                             {
                                 std::vector<UpsertRequest> batch;
                                 batch.reserve(per_thread);
                                 for (std::size_t i = 0; i < per_thread; ++i)
                                 {
                                     Id id = static_cast<Id>(t * per_thread + i + 1);
                                     batch.push_back(MakeUpsert(id, 8, 0.1f + static_cast<float>(id), 1));
                                 }
                                 db.UpsertBatch(batch, true).get();
                             });
    }

    for (auto &th : threads)
        th.join();

    ScanRequest req;
    req.batch_size = 256;
    auto ids = ScanAll(db, req);

    db.Stop();

    REQUIRE(ids.size() == per_thread * 4);
}
