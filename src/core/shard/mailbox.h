#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

namespace pomai::core
{

    template <class T>
    class BoundedMpscQueue
    {
    public:
        explicit BoundedMpscQueue(std::size_t capacity) : capacity_(capacity) {}

        bool TryPush(T &&v)
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (closed_ || q_.size() >= capacity_)
                return false;
            q_.emplace_back(std::move(v));
            cv_.notify_one();
            return true;
        }

        bool PushBlocking(T &&v)
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_space_.wait(lk, [&]
                           { return closed_ || q_.size() < capacity_; });
            if (closed_)
                return false;
            q_.emplace_back(std::move(v));
            cv_.notify_one();
            return true;
        }

        std::optional<T> PopBlocking()
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&]
                     { return closed_ || !q_.empty(); });
            if (q_.empty())
                return std::nullopt;
            T v = std::move(q_.front());
            q_.pop_front();
            cv_space_.notify_one();
            return v;
        }

        void Close()
        {
            std::lock_guard<std::mutex> lk(mu_);
            closed_ = true;
            cv_.notify_all();
            cv_space_.notify_all();
        }

    private:
        std::mutex mu_;
        std::condition_variable cv_;
        std::condition_variable cv_space_;
        std::deque<T> q_;
        std::size_t capacity_ = 0;
        bool closed_ = false;
    };

    // Backward-compatible name used across the codebase.
    template <class T>
    using Mailbox = BoundedMpscQueue<T>;

} // namespace pomai::core
