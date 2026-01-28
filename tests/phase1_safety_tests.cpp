#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "orbit_index.h"
#include "seed.h"
#include "shard.h"
#include "wal.h"

using namespace pomai;
namespace fs = std::filesystem;

struct ShardTestAccessor
{
    static void AddSegment(Shard &shard, const Seed::Snapshot &snap)
    {
        std::lock_guard<std::mutex> lk(shard.state_mu_);
        shard.segments_.push_back(IndexedSegment{snap, nullptr});
    }

    static void AttachIndex(Shard &shard,
                            std::size_t segment_pos,
                            Seed::Snapshot snap,
                            std::shared_ptr<pomai::core::OrbitIndex> idx)
    {
        shard.AttachIndex(segment_pos, std::move(snap), std::move(idx));
    }

    static bool SegmentSnapshotReleased(const Shard &shard, std::size_t segment_pos)
    {
        std::lock_guard<std::mutex> lk(shard.state_mu_);
        if (segment_pos >= shard.segments_.size())
            return false;
        return shard.segments_[segment_pos].snap == nullptr;
    }
};

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_phase1.XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char *res = mkdtemp(buf.data());
    if (!res)
        throw std::runtime_error("mkdtemp failed");
    return std::string(res);
}

static void RemoveDir(const std::string &d)
{
    std::error_code ec;
    fs::remove_all(d, ec);
}

int main()
{
    std::cout << "Phase1 safety tests starting...\n";
    int failures = 0;

    // Test 1: WAL rejects oversized batch payloads.
    try
    {
        std::string dir = MakeTempDir();
        Wal w("shard-0", dir, 1);
        w.Start();

        std::vector<UpsertRequest> batch;
        batch.reserve(MAX_BATCH_ROWS + 1);
        for (std::size_t i = 0; i < MAX_BATCH_ROWS + 1; ++i)
        {
            UpsertRequest r;
            r.id = static_cast<Id>(i + 1);
            r.vec.data = {1.0f};
            batch.push_back(std::move(r));
        }

        bool threw = false;
        try
        {
            w.AppendUpserts(batch, false);
        }
        catch (const std::runtime_error &e)
        {
            threw = true;
            const std::size_t per_entry_bytes = sizeof(uint64_t) + sizeof(float);
            const std::size_t expected_size =
                sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t) +
                (MAX_BATCH_ROWS + 1) * per_entry_bytes;
            const std::string expected =
                "WAL append rejected: payload too large (" + std::to_string(expected_size) +
                " bytes); split batch or reduce vector dimensions.";
            if (expected != e.what())
            {
                std::cerr << "Test1 FAILED: unexpected message: " << e.what() << "\n";
                ++failures;
            }
        }
        if (!threw)
        {
            std::cerr << "Test1 FAILED: expected WAL rejection\n";
            ++failures;
        }

        w.Stop();
        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test1 exception: " << e.what() << "\n";
        ++failures;
    }

    // Test 2: Snapshot is released after index attach.
    try
    {
        std::string dir = MakeTempDir();
        Shard shard("shard-0", 2, 8, dir);

        Seed seed(2);
        std::vector<UpsertRequest> batch;
        for (int i = 0; i < 3; ++i)
        {
            UpsertRequest r;
            r.id = static_cast<Id>(i + 1);
            r.vec.data = {static_cast<float>(i), static_cast<float>(i + 1)};
            batch.push_back(std::move(r));
        }
        seed.ApplyUpserts(batch);
        auto snap = seed.MakeSnapshot();

        ShardTestAccessor::AddSegment(shard, snap);

        auto idx = std::make_shared<pomai::core::OrbitIndex>(2);
        idx->Build(snap->data, snap->ids);

        ShardTestAccessor::AttachIndex(shard, 0, snap, idx);

        if (!ShardTestAccessor::SegmentSnapshotReleased(shard, 0))
        {
            std::cerr << "Test2 FAILED: snapshot not released after AttachIndex\n";
            ++failures;
        }

        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test2 exception: " << e.what() << "\n";
        ++failures;
    }

    if (failures == 0)
    {
        std::cout << "All Phase1 safety tests PASS\n";
        return 0;
    }

    std::cerr << failures << " Phase1 safety tests FAILED\n";
    return 1;
}
