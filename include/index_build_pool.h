#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "seed.h"
#include "orbit_index.h"
#include "memory_manager.h"

namespace pomai
{

    // Global background pool to build OrbitIndex asynchronously.
    // No dependency on Shard (uses callback to attach result).
    class IndexBuildPool
    {
    public:
        using AttachFn = std::function<void(std::size_t segment_pos,
                                            Seed::Snapshot snap,
                                            std::shared_ptr<pomai::core::OrbitIndex> idx)>;

        struct Job
        {
            std::size_t segment_pos{0};
            Seed::Snapshot snap;

            std::size_t M{48};
            std::size_t ef_construction{200};

            AttachFn attach; // called after build finishes
        };

        explicit IndexBuildPool(std::size_t workers)
            : worker_count_(workers ? workers : 1) {}

        ~IndexBuildPool() { Stop(); }

        IndexBuildPool(const IndexBuildPool &) = delete;
        IndexBuildPool &operator=(const IndexBuildPool &) = delete;

        void Start()
        {
            bool expected = false;
            if (!running_.compare_exchange_strong(expected, true))
                return;

            threads_.reserve(worker_count_);
            for (std::size_t i = 0; i < worker_count_; ++i)
            {
                threads_.emplace_back([this]
                                      { WorkerLoop(); });
            }
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

            for (auto &t : threads_)
            {
                if (t.joinable())
                    t.join();
            }
            threads_.clear();
        }

        bool Enqueue(Job job)
        {
            if (!running_.load(std::memory_order_acquire))
                return false;
            if (!job.snap)
                return false;
            if (!job.attach)
                return false;

            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stop_requested_)
                    return false;
                q_.push_back(std::move(job));
            }
            cv_.notify_one();
            return true;
        }

        std::size_t Pending() const
        {
            std::lock_guard<std::mutex> lk(mu_);
            return q_.size();
        }

    private:
        void WorkerLoop()
        {
            while (true)
            {
                Job job;

                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [&]
                             {
                                 if (stop_requested_)
                                     return true;
                                 if (q_.empty())
                                     return false;
                                 if (MemoryManager::Instance().AtOrAboveSoftWatermark() &&
                                     active_builds_.load(std::memory_order_acquire) >= 1)
                                     return false;
                                 return true;
                             });
                    if (stop_requested_ && q_.empty())
                        return;

                    job = std::move(q_.front());
                    q_.pop_front();
                    active_builds_.fetch_add(1, std::memory_order_release);
                }

                // Build index outside lock
                auto idx = std::make_shared<pomai::core::OrbitIndex>(
                    job.snap->dim, job.M, job.ef_construction);
                std::vector<float> moved_data;
                std::vector<Id> moved_ids;
                if (Seed::TryDetachSnapshot(job.snap, moved_data, moved_ids))
                {
                    idx->BuildFromMove(std::move(moved_data), std::move(moved_ids));
                }
                else
                {
                    idx->Build(job.snap->data, job.snap->ids);
                }

                // Attach callback
                job.attach(job.segment_pos, job.snap, std::move(idx));

                active_builds_.fetch_sub(1, std::memory_order_release);
                cv_.notify_all();
            }
        }

    private:
        std::size_t worker_count_{1};

        mutable std::mutex mu_;
        std::condition_variable cv_;
        std::deque<Job> q_;
        std::vector<std::thread> threads_;

        std::atomic<bool> running_{false};
        bool stop_requested_{false};
        std::atomic<std::size_t> active_builds_{0};
    };

} // namespace pomai
