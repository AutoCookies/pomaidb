#pragma once

#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstring>
#include <memory>
#include <iostream>
#include <cassert>
#include "src/core/config.h"

namespace pomai::core
{
    struct IngestBatch
    {
        alignas(64) std::atomic<uint32_t> cursor{0};
        std::vector<uint64_t> labels;
        std::vector<float> data_block;
        size_t capacity;
        size_t dim;

        IngestBatch(size_t cap, size_t d) : capacity(cap), dim(d)
        {
            labels.resize(cap);
            data_block.resize(cap * d);
            cursor.store(0, std::memory_order_relaxed);
        }

        void reset()
        {
            cursor.store(0, std::memory_order_relaxed);
        }

        bool is_full() const
        {
            return cursor.load(std::memory_order_relaxed) >= capacity;
        }
    };

    class Ingestor
    {
    public:
        Ingestor(const pomai::config::PomaiConfig &config, size_t dim)
            : cfg_(config.ingestor), dim_(dim), running_(true)
        {
            active_batch_ = std::make_unique<IngestBatch>(cfg_.batch_size, dim_);
            worker_thread_ = std::thread(&Ingestor::worker_loop, this);
        }

        ~Ingestor()
        {
            {
                std::lock_guard<std::mutex> lk(mu_);
                running_ = false;
                cv_.notify_all();
            }
            if (worker_thread_.joinable())
                worker_thread_.join();
        }

        bool submit(uint64_t label, const float *vec)
        {
            IngestBatch *batch = active_batch_.get();
            uint32_t idx = batch->cursor.fetch_add(1, std::memory_order_acq_rel);

            if (idx < batch->capacity)
            {
                batch->labels[idx] = label;
                size_t offset = static_cast<size_t>(idx) * dim_;
                std::memcpy(batch->data_block.data() + offset, vec, dim_ * sizeof(float));

                if (idx == batch->capacity - 1)
                {
                    rotate_batch();
                }
                return true;
            }
            else
            {
                rotate_batch();
                return submit_retry(label, vec);
            }
        }

        bool submit(uint64_t label, const std::vector<float> &vec)
        {
            if (vec.size() != dim_)
                return false;
            return submit(label, vec.data());
        }

    private:
        pomai::config::IngestorConfig cfg_;
        size_t dim_;
        std::atomic<bool> running_;
        std::unique_ptr<IngestBatch> active_batch_;
        std::queue<std::unique_ptr<IngestBatch>> full_queue_;
        std::queue<std::unique_ptr<IngestBatch>> free_queue_;
        std::mutex mu_;
        std::condition_variable cv_;
        std::thread worker_thread_;

        bool submit_retry(uint64_t label, const float *vec)
        {
            IngestBatch *batch = active_batch_.get();
            uint32_t idx = batch->cursor.fetch_add(1, std::memory_order_acq_rel);
            if (idx < batch->capacity)
            {
                batch->labels[idx] = label;
                size_t offset = static_cast<size_t>(idx) * dim_;
                std::memcpy(batch->data_block.data() + offset, vec, dim_ * sizeof(float));
                return true;
            }
            return false;
        }

        void rotate_batch()
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (active_batch_->is_full())
            {
                full_queue_.push(std::move(active_batch_));
                if (!free_queue_.empty())
                {
                    active_batch_ = std::move(free_queue_.front());
                    free_queue_.pop();
                    active_batch_->reset();
                }
                else
                {
                    active_batch_ = std::make_unique<IngestBatch>(cfg_.batch_size, dim_);
                }
                cv_.notify_one();
            }
        }

        void worker_loop()
        {
            while (true)
            {
                std::unique_ptr<IngestBatch> batch;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [this]
                             { return !full_queue_.empty() || !running_; });
                    if (!running_ && full_queue_.empty())
                        break;
                    if (full_queue_.empty())
                        continue;
                    batch = std::move(full_queue_.front());
                    full_queue_.pop();
                }
                process_batch(batch.get());
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    if (free_queue_.size() < cfg_.max_free_batches)
                    {
                        free_queue_.push(std::move(batch));
                    }
                }
            }
        }

        void process_batch(IngestBatch *batch)
        {
        }
    };
}