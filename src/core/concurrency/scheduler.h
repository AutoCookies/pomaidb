#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

/**
 * @brief Represents a task to be executed by the scheduler.
 */
class DatabaseTask {
public:
    virtual ~DatabaseTask() = default;
    
    /**
     * @brief Executes the task. Should be non-blocking.
     * @return Status of the operation.
     */
    virtual Status Run() = 0;
    
    virtual std::string Name() const = 0;
};

/**
 * @brief Simple task scheduler for periodic and one-off database maintenance.
 * This runs within the single-threaded event loop.
 */
class TaskScheduler {
public:
    TaskScheduler();
    ~TaskScheduler();

    /**
     * @brief Registers a periodic task.
     * @param task The task to run.
     * @param interval Time between runs.
     */
    void RegisterPeriodic(std::unique_ptr<DatabaseTask> task, std::chrono::milliseconds interval);

    /**
     * @brief Polls the scheduler to see if any tasks are due.
     * Should be called frequently from the main loop.
     */
    void Poll();

private:
    struct ScheduledTask {
        std::unique_ptr<DatabaseTask> task;
        std::chrono::milliseconds interval;
        std::chrono::steady_clock::time_point next_run;

        bool operator>(const ScheduledTask& other) const {
            return next_run > other.next_run;
        }
    };

    // Use a priority queue to manage tasks by their next run time.
    // However, since we want to iterate and handle periodic tasks easily,
    // a simple vector might be enough for a small number of maintenance tasks.
    // Let's use a vector for simplicity as the number of tasks is very low (Sync, Freeze, Compact).
    std::vector<ScheduledTask> tasks_;
};

} // namespace pomai::core
