#include "tests/common/test_main.h"

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "core/kernel/micro_kernel.h"
#include "core/concurrency/scheduler.h"

namespace pomai {
namespace {

class NoopPod final : public core::Pod {
public:
    explicit NoopPod(core::PodId id) : id_(id) {}
    void Handle(core::Message&& msg) override {
        if (msg.status_ptr) {
            *msg.status_ptr = Status::Ok();
        }
    }
    core::PodId Id() const override { return id_; }
    std::string Name() const override { return "Noop"; }
    core::MemoryQuota GetQuota() const override { return {}; }
private:
    core::PodId id_;
};

class NamedCounterTask final : public core::DatabaseTask {
public:
    explicit NamedCounterTask(std::string n) : name_(std::move(n)) {}
    Status Run() override { ++count; return Status::Ok(); }
    std::string Name() const override { return name_; }
    int count = 0;
private:
    std::string name_;
};

POMAI_TEST(Kernel_UnknownTargetPropagatesError) {
    core::MicroKernel k;
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kFlush);
    msg.status_ptr = &st;
    k.Enqueue(std::move(msg));
    k.ProcessAll();
    POMAI_EXPECT_EQ(st.code(), ErrorCode::kNotFound);
}

POMAI_TEST(Kernel_ProcessBudgetBoundsWork) {
    core::MicroKernel k;
    POMAI_EXPECT_OK(k.RegisterPod(std::make_unique<NoopPod>(core::PodId::kIndex)));
    for (int i = 0; i < 16; ++i) {
        Status st = Status::Ok();
        core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kFlush);
        msg.status_ptr = &st;
        k.Enqueue(std::move(msg));
    }
    const uint32_t processed = k.ProcessBudget(3, 5);
    POMAI_EXPECT_EQ(processed, 3u);
}

POMAI_TEST(Kernel_SchedulerBudgetDeterministicOrder) {
    core::TaskScheduler scheduler;
    auto ztask = std::make_unique<NamedCounterTask>("zeta");
    auto atask = std::make_unique<NamedCounterTask>("alpha");
    auto* z_ptr = ztask.get();
    auto* a_ptr = atask.get();
    scheduler.RegisterPeriodic(std::move(ztask), std::chrono::milliseconds(1));
    scheduler.RegisterPeriodic(std::move(atask), std::chrono::milliseconds(1));
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    scheduler.PollBudget(1, 10, true);
    POMAI_EXPECT_EQ(a_ptr->count, 1);
    POMAI_EXPECT_EQ(z_ptr->count, 0);
}

} // namespace
} // namespace pomai
