#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>

namespace pomai::util
{
    class ThreadPool
    {
    public:
        explicit ThreadPool(size_t threads)
        {
            for (size_t i = 0; i < threads; ++i)
            {
                workers_.emplace_back([this]
                                      {
                    for(;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex_);
                            condition_.wait(lock, [this]{ return stop_ || !tasks_.empty(); });
                            if(stop_ && tasks_.empty()) return;
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                        task();
                    } });
            }
        }

        ~ThreadPool()
        {
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                stop_ = true;
            }
            condition_.notify_all();
            // jthread joins automatically
        }

        template <class F, class... Args>
        auto Enqueue(F &&f, Args &&...args)
            -> std::future<typename std::invoke_result<F, Args...>::type>
        {
            using return_type = typename std::invoke_result<F, Args...>::type;

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));

            std::future<return_type> res = task->get_future();
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (stop_)
                    throw std::runtime_error("enqueue on stopped ThreadPool");

                tasks_.emplace([task]()
                               { (*task)(); });
            }
            condition_.notify_one();
            return res;
        }

    private:
        std::vector<std::jthread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable condition_;
        bool stop_ = false;
    };
} // namespace pomai::util
