// executor.h — Shard-local task executor with "Gentle Yielding".
// Inspired by ScyllaDB's stall-free patterns.
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <functional>
#include <vector>
#include "core/concurrency/mpsc_queue.h"

namespace pomai::core::concurrency {

/**
 * Task: An intrusive task wrapper that can be queued without allocation.
 * Uses a virtual Invoke() for C++20 compatibility instead of C++23 move_only_function.
 */
struct Task : public MPSCQueueEntry {
    virtual ~Task() = default;
    virtual void Invoke() = 0;
};

/**
 * LambdaTask: A concrete Task that wraps a lambda.
 */
template <typename F>
struct LambdaTask : public Task {
    F func;
    explicit LambdaTask(F&& f) : func(std::forward<F>(f)) {}
    void Invoke() override { func(); }
};

/**
 * Executor: Manages the execution of tasks on a specific shard.
 */
class Executor {
public:
    Executor() = default;

    /**
     * Submit: Add a task to this executor's queue.
     */
    void Submit(Task* task) {
        queue_.Push(task);
    }

    /**
     * Poll: Execute pending tasks.
     * Implements "Gentle Execution": only runs up to 'max_tasks' to prevent
     * starving other background operations or causing latency spikes.
     */
    size_t Poll(size_t max_tasks = 64) {
        size_t executed = 0;
        while (executed < max_tasks) {
            Task* t = queue_.Pop();
            if (!t) break;
            
            t->Invoke();
            delete t; 
            executed++;
        }
        return executed;
    }

    bool HasPending() const {
        // Technically Popping is needed to strictly know if empty in MPSCIntrusiveQueue
        // but for this distal engine, we use it in a polling loop.
        return true; 
    }

private:
    MPSCIntrusiveQueue<Task> queue_;
};

} // namespace pomai::core::concurrency
