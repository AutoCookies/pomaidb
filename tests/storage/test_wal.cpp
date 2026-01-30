#include <catch2/catch.hpp>

#include <pomai/storage/wal.h>

#include "tests/common/test_utils.h"

#include <algorithm>
#include <thread>

namespace
{
    using namespace pomai;
    using namespace pomai::test;
}

TEST_CASE("WAL durable LSN monotonic", "[storage][wal]")
{
    TempDir dir;
    Wal wal("shard-0", dir.str(), 4);
    wal.Start();

    auto batch1 = MakeBatch(3, 4, 0.1f);
    Lsn lsn1 = wal.AppendUpserts(batch1, true);
    wal.WaitDurable(lsn1);
    auto durable1 = wal.DurableLsn();

    auto batch2 = MakeBatch(2, 4, 1.0f);
    Lsn lsn2 = wal.AppendUpserts(batch2, true);
    wal.WaitDurable(lsn2);
    auto durable2 = wal.DurableLsn();

    wal.Stop();

    REQUIRE(lsn2 > lsn1);
    REQUIRE(durable2 >= durable1);
    REQUIRE(durable2 >= lsn2);
}

TEST_CASE("WAL handles concurrent appends", "[storage][wal]")
{
    TempDir dir;
    Wal wal("shard-0", dir.str(), 4);
    wal.Start();

    std::vector<std::thread> threads;
    std::vector<Lsn> lsns(4, 0);
    for (std::size_t t = 0; t < lsns.size(); ++t)
    {
        threads.emplace_back([&, t]()
                             {
                                 auto batch = MakeBatch(5, 4, 0.5f + static_cast<float>(t));
                                 lsns[t] = wal.AppendUpserts(batch, true);
                             });
    }
    for (auto &th : threads)
        th.join();

    for (auto lsn : lsns)
    {
        REQUIRE(lsn > 0);
        wal.WaitDurable(lsn);
    }

    wal.Stop();
    REQUIRE(wal.DurableLsn() >= *std::max_element(lsns.begin(), lsns.end()));
}
