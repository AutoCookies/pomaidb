#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/database.h"
#include "pomai/graph.h"
#include "pomai/hooks.h"
#include "pomai/options.h"
#include "pomai/pomai.h"
#include "core/membrane/manager.h"
#include "core/storage/sync_provider.h"
#include "core/concurrency/scheduler.h"
#include <filesystem>
#include <vector>
#include <thread>
#include <chrono>

namespace pomai {

class MockHook : public PostPutHook {
public:
    void OnPostPut(VectorId id, std::span<const float> vec, const Metadata& /*meta*/) override {
        last_id = id;
        last_vec.assign(vec.begin(), vec.end());
        call_count++;
    }

    VectorId last_id = 0;
    std::vector<float> last_vec;
    int call_count = 0;
};

class CounterTask : public core::DatabaseTask {
public:
    Status Run() override {
        count++;
        return Status::Ok();
    }
    std::string Name() const override { return "Counter"; }
    int count = 0;
};

POMAI_TEST(Edge_TaskScheduler) {
    core::TaskScheduler scheduler;
    auto task = std::make_unique<CounterTask>();
    auto* task_ptr = task.get();
    
    scheduler.RegisterPeriodic(std::move(task), std::chrono::milliseconds(10));
    
    // Ensure now > next_run for the first Poll()
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    
    // Poll immediately - should run because next_run = now
    scheduler.Poll();
    POMAI_EXPECT_EQ(task_ptr->count, 1);
    
    // Wait LESS than interval and poll - should NOT run
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    scheduler.Poll();
    POMAI_EXPECT_EQ(task_ptr->count, 1);
    
    // Wait total > interval and poll - should run
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    scheduler.Poll();
    POMAI_EXPECT_EQ(task_ptr->count, 2);
}

POMAI_TEST(Edge_PostPutHook) {
    std::string dir = pomai::test::TempDir("edge_hooks");
    EmbeddedOptions opt;
    opt.path = dir;
    opt.dim = 2;
    
    Database db;
    POMAI_EXPECT_OK(db.Open(opt));
    
    auto hook = std::make_shared<MockHook>();
    db.AddPostPutHook(hook);
    
    std::vector<float> v1 = {1.0f, 2.0f};
    POMAI_EXPECT_OK(db.AddVector(10, v1));
    
    POMAI_EXPECT_EQ(hook->call_count, 1);
    POMAI_EXPECT_EQ(hook->last_id, 10);
    POMAI_EXPECT_EQ(hook->last_vec[0], 1.0f);
    POMAI_EXPECT_EQ(hook->last_vec[1], 2.0f);
    
    (void)db.Close();
    std::filesystem::remove_all(dir);
}

POMAI_TEST(Edge_PushSync) {
    std::string dir = pomai::test::TempDir("edge_sync");
    EmbeddedOptions opt;
    opt.path = dir;
    opt.dim = 2;
    opt.fsync = FsyncPolicy::kAlways;
    
    Database db;
    POMAI_EXPECT_OK(db.Open(opt));
    
    auto receiver = std::make_shared<core::MockSyncReceiver>();
    db.RegisterSyncReceiver(receiver);
    
    // Wait a tiny bit to ensure now > next_run for the scheduler
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Add some data
    std::vector<float> v1 = {1.1f, 2.2f};
    POMAI_EXPECT_OK(db.AddVector(1, v1));
    std::vector<float> v2 = {3.3f, 4.4f};
    POMAI_EXPECT_OK(db.AddVector(2, v2));
    
    // Trigger sync manually via internal impl or let it run periodically.
    // Since periodic is 10s, we'll force it for test speed.
    // Forcing a "Poll" is hard via public API, so we just call a write that polls.
    POMAI_EXPECT_OK(db.AddVector(3, v1));
    
    // Wait a bit if it was async, but here it's sync in Poll().
    // MockSyncReceiver should have seen entries.
    POMAI_EXPECT_OK(db.Flush());
    if (receiver->last_received_lsn == 0) {
        std::cerr << "Sync failed: last_received_lsn is still 0\n";
    }
    POMAI_EXPECT_TRUE(receiver->last_received_lsn > 0);
    
    (void)db.Close();
    std::filesystem::remove_all(dir);
}

POMAI_TEST(Edge_GatewaySyncReplay) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("edge_sync_replay");
    opt.dim = 4;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;
    core::MembraneManager mgr(opt);
    POMAI_EXPECT_OK(mgr.Open());

    MembraneSpec gspec;
    gspec.name = "gx";
    gspec.kind = MembraneKind::kGraph;
    gspec.dim = 4;
    gspec.shard_count = 1;
    POMAI_EXPECT_OK(mgr.CreateMembrane(gspec));
    POMAI_EXPECT_OK(mgr.OpenMembrane("gx"));
    POMAI_EXPECT_OK(mgr.ReplayGatewaySyncEvent(1, "graph_vertex_put", "gx", 42, 0, 7, 0, "", ""));
    POMAI_EXPECT_OK(mgr.ReplayGatewaySyncEvent(2, "graph_vertex_put", "gx", 43, 0, 7, 0, "", ""));
    POMAI_EXPECT_OK(mgr.ReplayGatewaySyncEvent(3, "graph_edge_put", "gx", 42, 43, 0, 0, "", ""));
    std::vector<Neighbor> nbr;
    POMAI_EXPECT_OK(mgr.GetNeighbors("gx", 42, &nbr));
    POMAI_EXPECT_TRUE(!nbr.empty());

    MembraneSpec tspec;
    tspec.name = "tsm";
    tspec.kind = MembraneKind::kTimeSeries;
    tspec.dim = 4;
    tspec.shard_count = 1;
    POMAI_EXPECT_OK(mgr.CreateMembrane(tspec));
    POMAI_EXPECT_OK(mgr.OpenMembrane("tsm"));
    const uint64_t ts_ms = 1700000000000ULL;
    POMAI_EXPECT_OK(mgr.ReplayGatewaySyncEvent(4, "timeseries_put", "tsm", 99, ts_ms, 0, 0, "", "12.5"));
    std::vector<TimeSeriesPoint> pts;
    POMAI_EXPECT_OK(mgr.TsRange("tsm", 99, 0, 2000000000000ULL, &pts));
    POMAI_EXPECT_TRUE(!pts.empty());
    POMAI_EXPECT_EQ(pts[0].value, 12.5);

    std::filesystem::remove_all(opt.path);
}

} // namespace pomai
