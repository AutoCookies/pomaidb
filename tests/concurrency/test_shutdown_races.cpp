#include <catch2/catch.hpp>

#include <pomai/api/pomai_db.h>

#include "tests/common/test_utils.h"

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
    db.Start();

    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};

    std::thread ingester([&]()
                         {
                             for (std::size_t i = 0; i < 30; ++i)
                             {
                                 if (stop.load(std::memory_order_acquire))
                                     break;
                                 auto req = MakeUpsert(500 + i, 8, 1.1f + static_cast<float>(i), 1);
                                 try
                                 {
                                     db.UpsertBatch({req}, true).get();
                                 }
                                 catch (...)
                                 {
                                     failures.fetch_add(1, std::memory_order_relaxed);
                                 }
                             }
                         });

    std::thread searcher([&]()
                         {
                             for (std::size_t i = 0; i < 30; ++i)
                             {
                                 if (stop.load(std::memory_order_acquire))
                                     break;
                                 SearchRequest req = MakeSearchRequest(MakeVector(8, 0.1f), 5);
                                 auto resp = db.Search(req);
                                 if (resp.items.empty())
                                     failures.fetch_add(1, std::memory_order_relaxed);
                             }
                         });

    db.Stop();
    stop.store(true, std::memory_order_release);

    ingester.join();
    searcher.join();

    REQUIRE(failures.load() == 0);
}
