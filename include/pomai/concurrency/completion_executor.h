#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include <pomai/util/logger.h>

namespace pomai
{

    class CompletionExecutor
    {
    public:
        explicit CompletionExecutor(std::size_t max_queue = 1024, Logger *logger = nullptr)
            : max_queue_(max_queue == 0 ? 1 : max_queue),
              logger_(logger) {}

        ~CompletionExecutor() { Stop(); }

        CompletionExecutor(const CompletionExecutor &) = delete;
        CompletionExecutor &operator=(const CompletionExecutor &) = delete;

        void Start()
        {
            bool expected = false;
            if (!running_.compare_exchange_strong(expected, true))
                return;

            {
                std::lock_guard<std::mutex> lk(mu_);
                stop_requested_ = false;
                // Do NOT clear tasks_.
            }

            worker_ = std::thread([this]()
                                  { WorkerLoop(); });

            // Optional: wake in case worker starts and tasks arrive concurrently.
            cv_.notify_all();
        }

        void Stop()
        {
            // Make Stop idempotent and ALWAYS set stop_requested_.
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stop_requested_)
                    return;
                stop_requested_ = true;

                // Policy: cancel pending tasks to guarantee fast shutdown.
                while (!tasks_.empty())
                    tasks_.pop();
            }

            cv_.notify_all();

            // Only join if we actually started the worker thread.
            if (running_.exchange(false) && worker_.joinable())
                worker_.join();
        }

        bool Enqueue(std::function<void()> fn, std::chrono::steady_clock::duration delay = {})
        {
            if (!fn)
                return false;

            Task task;
            task.fn = std::move(fn);
            task.run_at = std::chrono::steady_clock::now() + delay;

            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stop_requested_)
                    return false;
                if (tasks_.size() >= max_queue_)
                    return false;
                tasks_.push(std::move(task));
            }

            // Notify only matters if the worker is running; safe to notify anyway.
            cv_.notify_one();
            return true;
        }

        std::size_t Pending() const
        {
            std::lock_guard<std::mutex> lk(mu_);
            return tasks_.size();
        }

    private:
        struct Task
        {
            std::function<void()> fn;
            std::chrono::steady_clock::time_point run_at;
        };

        struct TaskCmp
        {
            bool operator()(const Task &a, const Task &b) const
            {
                // priority_queue puts "largest" on top; we want earliest run_at
                return a.run_at > b.run_at;
            }
        };

        void WorkerLoop()
        {
            std::unique_lock<std::mutex> lk(mu_);

            for (;;)
            {
                // Wait until there is work, or stop is requested.
                cv_.wait(lk, [this]()
                         { return stop_requested_ || !tasks_.empty(); });

                if (stop_requested_)
                    return;

                // There is at least one task.
                while (!tasks_.empty())
                {
                    if (stop_requested_)
                        return;

                    const auto now = std::chrono::steady_clock::now();
                    const auto next_time = tasks_.top().run_at;

                    // If the next task is scheduled in the future, wait until that time
                    // OR until the queue changes (new earlier task) OR stop is requested.
                    if (next_time > now)
                    {
                        cv_.wait_until(lk, next_time, [this, next_time]()
                                       {
                                           return stop_requested_ ||
                                                  tasks_.empty() ||
                                                  tasks_.top().run_at < next_time; // earlier task inserted
                                       });

                        // Loop back to re-evaluate stop/queue/time
                        continue;
                    }

                    // Task is ready to run now.
                    Task task = tasks_.top();
                    tasks_.pop();

                    lk.unlock();
                    try
                    {
                        task.fn();
                    }
                    catch (...)
                    {
                        if (logger_)
                            logger_->Error("completion.task", "Completion task threw an exception");
                    }
                    lk.lock();
                }
            }
        }

        std::size_t max_queue_{1024};
        std::atomic<bool> running_{false};

        // Guarded by mu_
        bool stop_requested_{false};

        std::thread worker_;
        mutable std::mutex mu_;
        std::condition_variable cv_;
        std::priority_queue<Task, std::vector<Task>, TaskCmp> tasks_;
        Logger *logger_{nullptr};
    };

} // namespace pomai
