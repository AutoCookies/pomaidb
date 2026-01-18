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
#include <type_traits>

#include "src/core/config.h"
#include "src/core/types.h" // [REQUIRED] For DataType enum & dtype_size

namespace pomai::core
{
    // IngestBatch: Holds raw bytes to support any data type (float, int8, fp16, etc.)
    struct IngestBatch
    {
        alignas(64) std::atomic<uint32_t> cursor{0};
        std::vector<uint64_t> labels;
        std::vector<uint8_t> data_block; // [CHANGED] Raw bytes storage
        size_t capacity;
        size_t dim;
        size_t element_size;   // Size of one scalar element (e.g., 4 for float, 1 for int8)
        size_t vector_bytes;   // Size of one full vector in bytes

        IngestBatch(size_t cap, size_t d, pomai::core::DataType dtype) 
            : capacity(cap), dim(d)
        {
            element_size = pomai::core::dtype_size(dtype);
            vector_bytes = d * element_size;

            labels.resize(cap);
            // Allocate exactly enough bytes for the batch based on the data type
            data_block.resize(cap * vector_bytes); 
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
        // NOTE: keep constructor backward-compatible. If no global data_type is configured,
        // fall back to FLOAT32 for ingestion.
        Ingestor(const pomai::config::PomaiConfig &config, size_t dim)
            : cfg_(config.ingestor), dim_(dim), running_(true)
        {
            // Try to pick a global data type, but config.storage currently does not define one.
            // Use FLOAT32 by default. If you want another default, add a string field to config.
            dtype_ = pomai::core::DataType::FLOAT32;

            active_batch_ = std::make_unique<IngestBatch>(cfg_.batch_size, dim_, dtype_);
            worker_thread_ = std::thread(&Ingestor::worker_loop, this);
            
            std::clog << "[Ingestor] Initialized. Dim=" << dim_ 
                      << ", Type=" << pomai::core::dtype_name(dtype_) 
                      << ", BatchSize=" << cfg_.batch_size << "\n";
        }

        // Optional constructor to explicitly set DataType
        Ingestor(const pomai::config::PomaiConfig &config, size_t dim, pomai::core::DataType dtype)
            : cfg_(config.ingestor), dim_(dim), dtype_(dtype), running_(true)
        {
            active_batch_ = std::make_unique<IngestBatch>(cfg_.batch_size, dim_, dtype_);
            worker_thread_ = std::thread(&Ingestor::worker_loop, this);
            std::clog << "[Ingestor] Initialized (explicit). Dim=" << dim_ 
                      << ", Type=" << pomai::core::dtype_name(dtype_) 
                      << ", BatchSize=" << cfg_.batch_size << "\n";
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

        // Generic submit for any pointer type (float*, int8_t*, double*, etc.)
        // Returns false if sizeof(T) doesn't match the configured system data type.
        template <typename T>
        bool submit(uint64_t label, const T *vec)
        {
            // STRICT SAFETY CHECK: Ensure input type size matches storage configuration.
            if (sizeof(T) != pomai::core::dtype_size(dtype_))
            {
                return false; 
            }

            IngestBatch *batch = active_batch_.get();
            // Optimistic reservation
            uint32_t idx = batch->cursor.fetch_add(1, std::memory_order_acq_rel);

            if (idx < batch->capacity)
            {
                write_to_batch(batch, idx, label, vec);
                
                // If we filled the last slot, trigger rotation
                if (idx == batch->capacity - 1)
                {
                    rotate_batch();
                }
                return true;
            }
            else
            {
                // Batch full, force rotate and retry
                rotate_batch();
                return submit_retry(label, vec);
            }
        }

        // Overload for std::vector
        template <typename T>
        bool submit(uint64_t label, const std::vector<T> &vec)
        {
            if (vec.size() != dim_)
                return false;
            return submit(label, vec.data());
        }

    private:
        pomai::config::IngestorConfig cfg_;
        size_t dim_;
        pomai::core::DataType dtype_; // Stored system type
        
        std::atomic<bool> running_;
        std::unique_ptr<IngestBatch> active_batch_;
        std::queue<std::unique_ptr<IngestBatch>> full_queue_;
        std::queue<std::unique_ptr<IngestBatch>> free_queue_;
        std::mutex mu_;
        std::condition_variable cv_;
        std::thread worker_thread_;

        // Helper to write data into the batch at a specific index
        template <typename T>
        inline void write_to_batch(IngestBatch *batch, uint32_t idx, uint64_t label, const T *vec)
        {
            batch->labels[idx] = label;
            
            // Calculate byte offset
            size_t offset = static_cast<size_t>(idx) * batch->vector_bytes;
            
            // Memcpy raw bytes. Safe because we checked sizeof(T) vs element_size in public submit.
            std::memcpy(batch->data_block.data() + offset, vec, batch->vector_bytes);
        }

        template <typename T>
        bool submit_retry(uint64_t label, const T *vec)
        {
            IngestBatch *batch = active_batch_.get();
            uint32_t idx = batch->cursor.fetch_add(1, std::memory_order_acq_rel);
            if (idx < batch->capacity)
            {
                write_to_batch(batch, idx, label, vec);
                return true;
            }
            // Should rarely happen unless ingestion is massively outpacing workers
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
                    // Create new batch with same dimensions and type
                    active_batch_ = std::make_unique<IngestBatch>(cfg_.batch_size, dim_, dtype_);
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

        // This function will need to hand off data to HotTier/WAL.
        // HotTier should now accept raw bytes or handle casting if needed.
        void process_batch(IngestBatch *batch)
        {
            // TODO: Connect this to GlobalOrchestrator or HotTier.
            // Ensure downstream accepts: (const uint8_t* data, size_t count, DataType type)
        }
    };
}