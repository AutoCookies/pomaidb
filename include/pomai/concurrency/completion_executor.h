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
            worker_ = std::thread([this]()
                                  { WorkerLoop(); });
        }

        void Stop()
        {
            bool expected = true;
            if (!running_.compare_exchange_strong(expected, false))
                return;
            {
                std::lock_guard<std::mutex> lk(mu_);
                stop_requested_ = true;
            }
            cv_.notify_all();
            if (worker_.joinable())
                worker_.join();
        }

        bool Enqueue(std::function<void()> fn, std::chrono::steady_clock::duration delay = {})
        {
            if (!fn)
                return false;
            if (!running_.load(std::memory_order_acquire))
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
            cv_.notify_all();
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
                return a.run_at > b.run_at;
            }
        };

        void WorkerLoop()
        {
            std::unique_lock<std::mutex> lk(mu_);
            while (true)
            {
                if (stop_requested_ && tasks_.empty())
                    return;
                if (tasks_.empty())
                {
                    cv_.wait(lk, [this]()
                             { return stop_requested_ || !tasks_.empty(); });
                    continue;
                }
                auto next_time = tasks_.top().run_at;
                if (cv_.wait_until(lk, next_time, [this, next_time]()
                                   { return stop_requested_ || tasks_.empty() || tasks_.top().run_at != next_time; }))
                {
                    continue;
                }
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

        std::size_t max_queue_{1024};
        std::atomic<bool> running_{false};
        bool stop_requested_{false};
        std::thread worker_;
        mutable std::mutex mu_;
        std::condition_variable cv_;
        std::priority_queue<Task, std::vector<Task>, TaskCmp> tasks_;
        Logger *logger_{nullptr};
    };

} // namespace pomai
