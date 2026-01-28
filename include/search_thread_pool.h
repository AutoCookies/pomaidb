#pragma once
// A small bounded thread-pool suitable for request-scoped parallel fanout.
// - Submit() accepts a no-arg callable (or lambda) and returns a std::future for its result.
// - The pool has a fixed number of worker threads and a task queue.
// - Safe to use from multiple threads.
//
// Usage:
//   SearchThreadPool pool(std::min<size_t>(std::thread::hardware_concurrency(), 8));
//   auto fut = pool.Submit([&]{ return DoWork(); });
//   auto result = fut.get();
//
// Notes:
// - This implementation is intentionally small and dependency-free (header-only).
// - It throws std::runtime_error if Submit() is called after Stop() / destruction.
#include <atomic>
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
        explicit SearchThreadPool(std::size_t workers = std::thread::hardware_concurrency())
            : stop_(false)
        {
            if (workers == 0)
                workers = 1;
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

            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stop_.load(std::memory_order_acquire))
                    throw std::runtime_error("SearchThreadPool: submit on stopped pool");
                tasks_.emplace_back([task]()
                                    { (*task)(); });
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

    private:
        void WorkerLoop()
        {
            while (true)
            {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [this]
                             { return stop_.load(std::memory_order_acquire) || !tasks_.empty(); });
                    if (stop_.load(std::memory_order_acquire) && tasks_.empty())
                        return;
                    job = std::move(tasks_.front());
                    tasks_.pop_front();
                }
                try
                {
                    job();
                }
                catch (...)
                {
                    // Swallow exceptions from tasks; individual task futures capture exceptions.
                }
            }
        }

        std::size_t workers_{0};
        std::vector<std::thread> threads_;
        std::deque<std::function<void()>> tasks_;
        mutable std::mutex mu_;
        std::condition_variable cv_;
        std::atomic<bool> stop_{false};
    };

} // namespace pomai