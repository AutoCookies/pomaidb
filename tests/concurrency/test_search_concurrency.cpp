#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "common/test_utils.h"

#include <atomic>
#include <thread>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("Concurrent search during ingest", "[concurrency][search]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 8, 2);
    PomaiDB db(opts);
    REQUIRE(db.Start().ok());

    auto initial = MakeBatch(50, 8, 0.1f, 1);
    auto upsert_res = db.UpsertBatch(initial, true).get();
    REQUIRE(upsert_res.ok());

    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};

    std::thread ingest([&]()
                       {
                           for (std::size_t i = 0; i < 25; ++i)
                           {
                               auto req = MakeUpsert(1000 + i, 8, 2.0f + static_cast<float>(i), 1);
                               auto res = db.UpsertBatch({req}, false).get();
                               if (!res.ok())
                                   failures.fetch_add(1, std::memory_order_relaxed);
                           }
                           stop.store(true, std::memory_order_release);
                       });

    std::thread searcher([&]()
                         {
                             while (!stop.load(std::memory_order_acquire))
                             {
                                 SearchRequest req = MakeSearchRequest(initial[0].vec, 5);
                                 auto resp_res = db.Search(req);
                                 if (!resp_res.ok() || resp_res.value().items.empty())
                                     failures.fetch_add(1, std::memory_order_relaxed);
                             }
                         });

    ingest.join();
    searcher.join();

    REQUIRE(db.Stop().ok());

    REQUIRE(failures.load() == 0);
}
