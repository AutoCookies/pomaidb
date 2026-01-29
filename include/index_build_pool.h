#pragma once

#include <atomic>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
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
        static constexpr std::size_t kQueueCapacity = 1024;

        using AttachFn = std::function<void(std::size_t segment_pos,
                                            Seed::Snapshot snap,
                                            std::shared_ptr<pomai::core::OrbitIndex> idx)>;

        struct Job
        {
            std::function<void()> task;
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
            if (!job.task && !job.snap)
                return false;
            if (!job.task && !job.attach)
                return false;

            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stop_requested_)
                    return false;
                if (q_.Size() >= kQueueCapacity)
                    return false;
                if (!q_.Push(std::move(job)))
                    return false;
            }
            cv_.notify_one();
            return true;
        }

        bool EnqueueTask(std::function<void()> task)
        {
            Job job;
            job.task = std::move(task);
            return Enqueue(std::move(job));
        }

        std::size_t Pending() const
        {
            std::lock_guard<std::mutex> lk(mu_);
            return q_.Size();
        }

        std::size_t ActiveBuilds() const
        {
            return active_builds_.load(std::memory_order_acquire);
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
                                 if (q_.Empty())
                                     return false;
                                 std::size_t limit = worker_count_;
                                 if (MemoryManager::Instance().AtOrAboveSoftWatermark())
                                     limit = 1;
                                 if (active_builds_.load(std::memory_order_acquire) >= limit)
                                     return false;
                                 return true;
                             });
                    if (stop_requested_ && q_.Empty())
                        return;

                    q_.Pop(job);
                    active_builds_.fetch_add(1, std::memory_order_release);
                }

                if (job.task)
                {
                    job.task();
                }
                else
                {
                    auto idx = std::make_shared<pomai::core::OrbitIndex>(
                        job.snap->dim, job.M, job.ef_construction);
                    auto &mm = MemoryManager::Instance();
                    const std::size_t total = mm.TotalUsage();
                    const std::size_t hard = mm.HardWatermarkBytes();
                    const std::size_t budget = (hard > total) ? (hard - total) : 0;
                    std::vector<float> data = Seed::DequantizeSnapshotBounded(job.snap, budget);
                    if (!data.empty())
                    {
                        std::vector<Id> ids = job.snap->ids;
                        idx->BuildFromMove(std::move(data), std::move(ids));
                        job.attach(job.segment_pos, job.snap, std::move(idx));
                    }
                }

                active_builds_.fetch_sub(1, std::memory_order_release);
                cv_.notify_all();
            }
        }

    private:
        std::size_t worker_count_{1};

        template <typename T, std::size_t Capacity>
        class FixedQueue
        {
        public:
            bool Empty() const noexcept { return size_ == 0; }
            std::size_t Size() const noexcept { return size_; }

            bool Push(T value)
            {
                if (size_ >= Capacity)
                    return false;
                buffer_[tail_] = std::move(value);
                tail_ = (tail_ + 1) % Capacity;
                ++size_;
                return true;
            }

            bool Pop(T &out)
            {
                if (size_ == 0)
                    return false;
                out = std::move(buffer_[head_]);
                head_ = (head_ + 1) % Capacity;
                --size_;
                return true;
            }

        private:
            std::array<T, Capacity> buffer_{};
            std::size_t head_{0};
            std::size_t tail_{0};
            std::size_t size_{0};
        };

        mutable std::mutex mu_;
        std::condition_variable cv_;
        FixedQueue<Job, kQueueCapacity> q_;
        std::vector<std::thread> threads_;

        std::atomic<bool> running_{false};
        bool stop_requested_{false};
        std::atomic<std::size_t> active_builds_{0};
    };

} // namespace pomai
