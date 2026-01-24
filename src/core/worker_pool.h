#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <future>
#include <optional>
#include <cstdint>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <functional>
#include "src/facade/sql_executor.h"
#include "src/core/pomai_db.h"
#include "src/ai/whispergrain.h"

namespace pomai::server
{

    struct WorkerTaskResult
    {
        std::string resp;
        ClientState state;
    };

    class WorkerPool
    {
    public:
        using Handler = std::function<WorkerTaskResult(const std::string &, ClientState &&)>;

        WorkerPool(size_t n, Handler h)
            : handler_(std::move(h)), running_(true), rr_(0)
        {
            if (n == 0)
                n = 1;
            queues_.resize(n);
            mtxs_.resize(n);
            cvs_.resize(n);
            threads_.reserve(n);
            for (size_t i = 0; i < n; ++i)
            {
                threads_.emplace_back([this, i]()
                                      { this->worker_loop(i); });
            }
        }

        ~WorkerPool()
        {
            running_.store(false, std::memory_order_release);
            for (auto &cv : cvs_)
                cv.notify_all();
            for (auto &t : threads_)
                if (t.joinable())
                    t.join();
        }

        std::future<WorkerTaskResult> submit(std::string cmd, ClientState state)
        {
            size_t idx = rr_.fetch_add(1, std::memory_order_relaxed) % queues_.size();
            std::promise<WorkerTaskResult> p;
            auto fut = p.get_future();
            {
                std::lock_guard<std::mutex> lk(mtxs_[idx]);
                queues_[idx].emplace_back(std::move(cmd), std::move(state), std::move(p));
            }
            cvs_[idx].notify_one();
            return fut;
        }

    private:
        struct Item
        {
            std::string cmd;
            ClientState state;
            std::promise<WorkerTaskResult> prom;
            Item(std::string c, ClientState s, std::promise<WorkerTaskResult> p)
                : cmd(std::move(c)), state(std::move(s)), prom(std::move(p)) {}
        };

        void worker_loop(size_t idx)
        {
            while (running_.load(std::memory_order_acquire))
            {
                Item it("", ClientState{}, std::promise<WorkerTaskResult>());
                bool have = false;
                {
                    std::unique_lock<std::mutex> lk(mtxs_[idx]);
                    cvs_[idx].wait_for(lk, std::chrono::milliseconds(100), [this, idx]()
                                       { return !queues_[idx].empty() || !running_.load(std::memory_order_acquire); });
                    if (!queues_[idx].empty())
                    {
                        it = std::move(queues_[idx].front());
                        queues_[idx].pop_front();
                        have = true;
                    }
                }
                if (!have)
                    continue;
                try
                {
                    WorkerTaskResult r = handler_(it.cmd, std::move(it.state));
                    it.prom.set_value(std::move(r));
                }
                catch (...)
                {
                    try
                    {
                        it.prom.set_exception(std::current_exception());
                    }
                    catch (...)
                    {
                    }
                }
            }
            // drain remaining
            for (size_t i = 0; i < queues_.size(); ++i)
            {
                std::deque<Item> local;
                {
                    std::lock_guard<std::mutex> lk(mtxs_[i]);
                    local.swap(queues_[i]);
                }
                for (auto &it : local)
                {
                    try
                    {
                        WorkerTaskResult r = handler_(it.cmd, std::move(it.state));
                        it.prom.set_value(std::move(r));
                    }
                    catch (...)
                    {
                        try
                        {
                            it.prom.set_exception(std::current_exception());
                        }
                        catch (...)
                        {
                        }
                    }
                }
            }
        }

        Handler handler_;
        std::atomic<bool> running_;
        std::vector<std::thread> threads_;
        std::vector<std::deque<Item>> queues_;
        std::vector<std::mutex> mtxs_;
        std::vector<std::condition_variable> cvs_;
        std::atomic<size_t> rr_;
    };

} // namespace pomai::server