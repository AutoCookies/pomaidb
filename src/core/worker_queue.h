#pragma once
#include <deque>
#include <mutex>
#include <condition_variable>
#include <optional>
#include "src/core/worker.h"

namespace pomai::core
{

class WorkerQueue
{
public:
    WorkerQueue() = default;
    WorkerQueue(const WorkerQueue&) = delete;
    WorkerQueue& operator=(const WorkerQueue&) = delete;

    void push(Work &&w)
    {
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.emplace_back(std::move(w));
        }
        cv_.notify_one();
    }

    // Blocking pop: waits until an element is available
    Work pop_wait()
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return !q_.empty(); });
        Work w = std::move(q_.front());
        q_.pop_front();
        return w;
    }

    // Try pop non-blocking
    std::optional<Work> try_pop()
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return std::nullopt;
        Work w = std::move(q_.front());
        q_.pop_front();
        return w;
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Work> q_;
};

} // namespace pomai::core