#pragma once
// A small bounded thread-pool suitable for request-scoped parallel fanout.
// - Submit() accepts a no-arg callable (or lambda) and returns a std::future for its result.
// - The pool has a fixed number of worker threads and a task queue.
// - Safe to use from multiple threads.
//
// Usage:
//   SearchThreadPool pool(std::min<size_t>(std::thread::hardware_concurrency(), 8), 1024);
//   auto fut = pool.Submit([&]{ return DoWork(); });
//   auto result = fut.get();
//
// Notes:
// - This implementation is intentionally small and dependency-free (header-only).
// - It throws std::runtime_error if Submit() is called after Stop() / destruction.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <stdexcept>

namespace pomai
{

    class SearchThreadPool
    {
    public:
        explicit SearchThreadPool(std::size_t workers = std::thread::hardware_concurrency(),
                                  std::size_t max_queue_size = 1024)
            : stop_(false),
              max_queue_size_(max_queue_size)
        {
            if (workers == 0)
                workers = 1;
            if (max_queue_size_ == 0)
                max_queue_size_ = 1;
            workers_ = workers;
            threads_.reserve(workers_);
            for (std::size_t i = 0; i < workers_; ++i)
            {
                threads_.emplace_back([this]
                                      { this->WorkerLoop(); });
            }
        }

        ~SearchThreadPool()
        {
            Stop();
        }

        // Non-copyable
        SearchThreadPool(const SearchThreadPool &) = delete;
        SearchThreadPool &operator=(const SearchThreadPool &) = delete;

        // Submit a no-arg callable. Returns a future for the call result.
        template <typename F>
        auto Submit(F &&f)
            -> std::future<typename std::invoke_result_t<F>>
        {
            using R = typename std::invoke_result_t<F>;

            auto task = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
            std::future<R> fut = task->get_future();

            const auto queued_at = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stop_.load(std::memory_order_acquire))
                    throw std::runtime_error("SearchThreadPool: submit on stopped pool");
                if (tasks_.size() >= max_queue_size_)
                    throw std::runtime_error("SearchThreadPool: queue capacity reached");
                tasks_.push_back(QueuedTask{[task]()
                                            { (*task)(); },
                                            queued_at});
            }
            cv_.notify_one();
            return fut;
        }

        // Stop the pool and join all worker threads. Safe to call multiple times.
        void Stop()
        {
            bool expected = false;
            if (!stop_.compare_exchange_strong(expected, true))
                return;

            {
                std::lock_guard<std::mutex> lk(mu_);
                // clear queued tasks (optional). We keep tasks to let them finish normally.
            }
            cv_.notify_all();
            for (auto &t : threads_)
            {
                if (t.joinable())
                    t.join();
            }
            threads_.clear();
        }

        std::size_t WorkerCount() const noexcept { return workers_; }
        double QueueWaitEmaMs() const noexcept { return queue_wait_ema_ms_.load(std::memory_order_relaxed); }

    private:
        struct QueuedTask
        {
            std::function<void()> fn;
            std::chrono::steady_clock::time_point queued_at;
        };

        void WorkerLoop()
        {
            while (true)
            {
                QueuedTask job;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [this]
                             { return stop_.load(std::memory_order_acquire) || !tasks_.empty(); });
                    if (stop_.load(std::memory_order_acquire) && tasks_.empty())
                        return;
                    job = std::move(tasks_.front());
                    tasks_.pop_front();
                }
                const auto start = std::chrono::steady_clock::now();
                const std::chrono::duration<double, std::milli> wait_ms = start - job.queued_at;
                UpdateQueueWaitEma(wait_ms.count());
                try
                {
                    job.fn();
                }
                catch (...)
                {
                    // Swallow exceptions from tasks; individual task futures capture exceptions.
                }
            }
        }

        void UpdateQueueWaitEma(double wait_ms)
        {
            constexpr double kAlpha = 0.1;
            double old = queue_wait_ema_ms_.load(std::memory_order_relaxed);
            if (old <= 0.0)
            {
                queue_wait_ema_ms_.store(wait_ms, std::memory_order_relaxed);
                return;
            }
            double next = kAlpha * wait_ms + (1.0 - kAlpha) * old;
            queue_wait_ema_ms_.store(next, std::memory_order_relaxed);
        }

        std::size_t workers_{0};
        std::vector<std::thread> threads_;
        std::deque<QueuedTask> tasks_;
        mutable std::mutex mu_;
        std::condition_variable cv_;
        std::atomic<bool> stop_{false};
        std::atomic<double> queue_wait_ema_ms_{0.0};
        std::size_t max_queue_size_{1024};
    };

} // namespace pomai
