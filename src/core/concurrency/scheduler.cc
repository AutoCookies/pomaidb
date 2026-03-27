#include "core/concurrency/scheduler.h"
#include <algorithm>

namespace pomai::core {

TaskScheduler::TaskScheduler() = default;
TaskScheduler::~TaskScheduler() = default;

void TaskScheduler::RegisterPeriodic(std::unique_ptr<DatabaseTask> task, std::chrono::milliseconds interval) {
    ScheduledTask st;
    st.task = std::move(task);
    st.interval = interval;
    st.next_run = std::chrono::steady_clock::now();
    tasks_.push_back(std::move(st));
}

void TaskScheduler::Poll() {
    auto now = std::chrono::steady_clock::now();
    for (auto& st : tasks_) {
        if (now >= st.next_run) {
            (void)st.task->Run();
            st.next_run = now + st.interval;
        }
    }
}

void TaskScheduler::PollBudget(uint32_t max_ops, uint32_t max_ms, bool deterministic) {
    if (tasks_.empty()) return;
    if (max_ops == 0) max_ops = 1;
    if (max_ms == 0) max_ms = 1;
    auto now = std::chrono::steady_clock::now();
    const auto deadline = now + std::chrono::milliseconds(max_ms);
    uint32_t ops = 0;
    while (ops < max_ops && std::chrono::steady_clock::now() < deadline) {
        size_t idx = tasks_.size();
        if (!deterministic) {
            for (size_t i = 0; i < tasks_.size(); ++i) {
                if (now >= tasks_[i].next_run) {
                    idx = i;
                    break;
                }
            }
        } else {
            for (size_t i = 0; i < tasks_.size(); ++i) {
                if (now < tasks_[i].next_run) continue;
                if (idx == tasks_.size() || tasks_[i].task->Name() < tasks_[idx].task->Name()) {
                    idx = i;
                }
            }
        }
        if (idx == tasks_.size()) break;
        if (ops >= max_ops) break;
        auto& st = tasks_[idx];
        (void)st.task->Run();
        st.next_run = now + st.interval;
        ++ops;
    }
}

} // namespace pomai::core
