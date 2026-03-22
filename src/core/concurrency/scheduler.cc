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

} // namespace pomai::core
