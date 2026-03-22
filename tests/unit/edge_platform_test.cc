#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/database.h"
#include "pomai/hooks.h"
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

} // namespace pomai
