#include <pomai/core/shard.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;

static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_compact.XXXXXX";
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
    int failures = 0;
    std::string dir = MakeTempDir();
    try
    {
        pomai::CompactionConfig cfg;
        cfg.compaction_trigger_threshold = 2;
        cfg.max_concurrent_compactions = 1;
        pomai::Shard shard("shard-0", 2, 16, dir, cfg);
        shard.Start();

        for (int i = 0; i < 4; ++i)
        {
            pomai::UpsertRequest req;
            req.id = static_cast<pomai::Id>(100 + i);
            req.vec.data = {static_cast<float>(i), static_cast<float>(i + 1)};
            std::vector<pomai::UpsertRequest> batch;
            batch.push_back(std::move(req));
            shard.EnqueueUpserts(batch, true).get();
            shard.RequestEmergencyFreeze();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto state = shard.SnapshotState();
        if (!state)
        {
            std::cerr << "Missing shard state\n";
            ++failures;
        }
        else if (state->segments.size() > 3)
        {
            std::cerr << "Compaction did not reduce segments\n";
            ++failures;
        }

        std::unordered_set<pomai::Id> ids;
        if (state)
        {
            for (const auto &seg : state->segments)
            {
                if (!seg.snap)
                    continue;
                for (auto id : seg.snap->ids)
                    ids.insert(id);
            }
            if (state->live_snap)
            {
                for (auto id : state->live_snap->ids)
                    ids.insert(id);
            }
        }
        if (ids.size() != 4)
        {
            std::cerr << "Expected 4 ids after compaction, got " << ids.size() << "\n";
            ++failures;
        }

        shard.Stop();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Compaction test exception: " << e.what() << "\n";
        ++failures;
    }

    RemoveDir(dir);
    return failures == 0 ? 0 : 1;
}
