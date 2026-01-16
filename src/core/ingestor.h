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

namespace pomai::core
{

    // Cấu hình cứng cho Performance
    static constexpr size_t BATCH_SIZE = 4096; // Lớn để giảm tần suất lock
    static constexpr size_t VECTOR_DIM = 512;  // Kích thước vector cố định (hoặc truyền vào ctor)

    // -----------------------------------------------------------
    // IngestBatch: Container chứa dữ liệu (được tái sử dụng)
    // -----------------------------------------------------------
    struct IngestBatch
    {
        // Cache-line aligned cursor để tránh False Sharing
        alignas(64) std::atomic<uint32_t> cursor{0};
        
        // Dữ liệu nhãn
        std::vector<uint64_t> labels;
        
        // Dữ liệu vector (Flat layout: [v1_1...v1_n, v2_1...v2_n])
        // Tốt hơn mảng struct vì dễ prefetch và memcpy hàng loạt
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

    // -----------------------------------------------------------
    // Ingestor: Actor quản lý luồng ghi
    // -----------------------------------------------------------
    class Ingestor
    {
    public:
        Ingestor(size_t dim = VECTOR_DIM) : dim_(dim), running_(true)
        {
            // Pre-allocate active batch
            active_batch_ = std::make_unique<IngestBatch>(BATCH_SIZE, dim_);
            
            // Start background worker
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

        // HOT PATH: Wait-free (99% cases), Thread-safe
        // Trả về false nếu quá tải (hiếm khi xảy ra nếu worker nhanh)
        bool submit(uint64_t label, const float *vec)
        {
            // 1. Atomic Reserve: Lấy index hiện tại và tăng lên
            // Đây là thao tác không khóa (lock-free), cực nhanh.
            IngestBatch* batch = active_batch_.get();
            uint32_t idx = batch->cursor.fetch_add(1, std::memory_order_acq_rel);

            // 2. Check Capacity (Safe Check)
            if (idx < batch->capacity)
            {
                // HAPPY PATH: Chúng ta sở hữu slot [idx]. Ghi dữ liệu vào.
                // Không sợ race condition vì idx là duy nhất cho thread này.
                batch->labels[idx] = label;
                
                // Tính offset trong mảng flat
                size_t offset = static_cast<size_t>(idx) * dim_;
                std::memcpy(batch->data_block.data() + offset, vec, dim_ * sizeof(float));
                
                // Nếu đây là slot cuối cùng, báo hiệu worker đổi batch
                if (idx == batch->capacity - 1)
                {
                    rotate_batch();
                }
                return true;
            }
            else
            {
                // FULL PATH: Batch đã đầy. 
                // Các thread đến chậm (idx >= capacity) phải chờ hoặc giúp rotate.
                // Để đơn giản và an toàn, ta gọi rotate (có lock) và thử lại.
                rotate_batch();
                
                // Thử lại đệ quy (recursive retry) hoặc trả về false để caller retry
                // Ở đây ta retry 1 lần trên batch mới
                return submit_retry(label, vec);
            }
        }

        // Helper cho std::vector wrapper
        bool submit(uint64_t label, const std::vector<float> &vec)
        {
            if (vec.size() != dim_) return false;
            return submit(label, vec.data());
        }

    private:
        size_t dim_;
        std::atomic<bool> running_;
        
        // Active Batch: Không cần atomic pointer vì ta bảo vệ việc ROTATE bằng mutex.
        // Việc WRITE vào active batch là lock-free.
        std::unique_ptr<IngestBatch> active_batch_;

        // Queue: Chuyển dữ liệu cho Worker
        std::queue<std::unique_ptr<IngestBatch>> full_queue_;
        
        // Queue: Tái sử dụng bộ nhớ (Object Pool)
        std::queue<std::unique_ptr<IngestBatch>> free_queue_;

        std::mutex mu_;
        std::condition_variable cv_;
        std::thread worker_thread_;

        // Retry logic khi vừa rotate xong
        bool submit_retry(uint64_t label, const float* vec)
        {
            IngestBatch* batch = active_batch_.get();
            uint32_t idx = batch->cursor.fetch_add(1, std::memory_order_acq_rel);
            if (idx < batch->capacity)
            {
                batch->labels[idx] = label;
                size_t offset = static_cast<size_t>(idx) * dim_;
                std::memcpy(batch->data_block.data() + offset, vec, dim_ * sizeof(float));
                return true;
            }
            return false; // Quá tải nặng, worker không kịp xử lý
        }

        // Logic đổi batch (Cold path - có lock nhưng hiếm)
        void rotate_batch()
        {
            std::lock_guard<std::mutex> lk(mu_);
            
            // Double-check: Có thể thread khác đã rotate rồi
            if (active_batch_->is_full())
            {
                // 1. Đẩy batch đầy vào hàng đợi xử lý
                full_queue_.push(std::move(active_batch_));
                
                // 2. Lấy batch rỗng từ hồ chứa (recycle) hoặc tạo mới
                if (!free_queue_.empty())
                {
                    active_batch_ = std::move(free_queue_.front());
                    free_queue_.pop();
                    active_batch_->reset(); // Reset cursor về 0
                }
                else
                {
                    // Nếu không có sẵn, tạo mới (chỉ xảy ra lúc đầu hoặc load tăng đột biến)
                    active_batch_ = std::make_unique<IngestBatch>(BATCH_SIZE, dim_);
                }

                // 3. Đánh thức worker
                cv_.notify_one();
            }
        }

        // BACKGROUND WORKER
        void worker_loop()
        {
            while (true)
            {
                std::unique_ptr<IngestBatch> batch;

                // 1. Wait for data
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    cv_.wait(lk, [this] { return !full_queue_.empty() || !running_; });

                    if (!running_ && full_queue_.empty())
                        break;

                    if (full_queue_.empty()) continue;

                    batch = std::move(full_queue_.front());
                    full_queue_.pop();
                }

                // 2. Process Batch (Quantize -> Route -> Insert)
                // Đây là chỗ thực hiện các tác vụ nặng mà không block luồng submit
                process_batch(batch.get());

                // 3. Recycle Batch (Trả vỏ chai về nơi sản xuất)
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    // Giới hạn pool size để không ăn hết RAM nếu queue quá dài
                    if (free_queue_.size() < 100) 
                    {
                        free_queue_.push(std::move(batch));
                    }
                    // Nếu pool đầy, batch sẽ tự hủy (destructor) tại đây, giải phóng RAM
                }
            }
        }

        void process_batch(IngestBatch* batch)
        {
            // [PLACEHOLDER] Logic xử lý thật của bạn ở đây
            // 1. Quantize: encode_batch(batch->data_block)
            // 2. Route: sort/group by centroid
            // 3. Insert: shards_[id]->bulk_insert(...)
            
            // Ví dụ log:
            // std::cout << "[Ingestor] Processed batch of " << batch->capacity << " vectors.\n";
        }
    };

} // namespace pomai::core