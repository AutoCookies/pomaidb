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

TEST_CASE("Shutdown races do not crash", "[concurrency][shutdown]")
{
    TempDir dir;
    auto opts = DefaultDbOptions(dir.str(), 8, 2);
    PomaiDB db(opts);
    REQUIRE(db.Start().ok());

    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};

    std::thread ingester([&]()
                         {
                             for (std::size_t i = 0; i < 30; ++i)
                             {
                                 if (stop.load(std::memory_order_acquire))
                                     break;
                                 auto req = MakeUpsert(500 + i, 8, 1.1f + static_cast<float>(i), 1);
                                 auto res = db.UpsertBatch({req}, true).get();
                                 if (!res.ok())
                                     failures.fetch_add(1, std::memory_order_relaxed);
                             }
                         });

    std::thread searcher([&]()
                         {
                             for (std::size_t i = 0; i < 30; ++i)
                             {
                                 if (stop.load(std::memory_order_acquire))
                                     break;
                                 SearchRequest req = MakeSearchRequest(MakeVector(8, 0.1f), 5);
                                 auto resp_res = db.Search(req);
                                 if (!resp_res.ok() || resp_res.value().items.empty())
                                     failures.fetch_add(1, std::memory_order_relaxed);
                             }
                         });

    REQUIRE(db.Stop().ok());
    stop.store(true, std::memory_order_release);

    ingester.join();
    searcher.join();

    REQUIRE(failures.load() == 0);
}
