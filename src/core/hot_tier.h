#pragma once

#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <algorithm>
#include <queue>
#include <cstdint>
#include <chrono>

#include "src/core/config.h"
#include "src/core/types.h"
#include "src/core/cpu_kernels.h"

namespace pomai::core
{

    struct HotBatch
    {
        std::vector<uint64_t> labels;
        std::vector<uint8_t> data;
        size_t dim = 0;
        uint32_t element_size = 0;
        DataType data_type = DataType::FLOAT32;

        bool empty() const { return labels.empty(); }
        size_t count() const { return labels.size(); }
    };

    class HotTier
    {
    public:
        enum class InsertResult : uint8_t
        {
            OK = 0,
            FULL = 1,
            INVALID = 2,
            TIMEOUT = 3
        };

        HotTier(size_t dim, const pomai::config::HotTierConfig &cfg, DataType dt = DataType::FLOAT32)
            : dim_(dim),
              data_type_(dt),
              capacity_limit_(std::max<size_t>(1, cfg.initial_capacity)),
              element_size_(static_cast<uint32_t>(dtype_size(dt))),
              stride_(dim_ * element_size_),
              head_(0),
              tail_(0),
              block_when_full_(true),
              push_timeout_ms_(0)
        {
            labels_.resize(capacity_limit_);
            data_.resize(capacity_limit_ * stride_);
        }

        uint32_t element_size() const noexcept { return element_size_; }
        DataType data_type() const noexcept { return data_type_; }
        std::string data_type_string() const { return dtype_name(data_type_); }
        size_t dim() const noexcept { return dim_; }
        size_t capacity_hint() const noexcept { return capacity_limit_; }

        void set_block_when_full(bool v) noexcept { block_when_full_ = v; }
        void set_push_timeout_ms(uint32_t ms) noexcept { push_timeout_ms_ = ms; }

        InsertResult push(uint64_t label, const float *vec)
        {
            if (!vec)
                return InsertResult::INVALID;

            std::unique_lock<std::mutex> lock(prod_mutex_);

            auto is_full = [this]() noexcept
            {
                uint32_t next = (head_ + 1) % static_cast<uint32_t>(capacity_limit_);
                return next == tail_;
            };

            if (!block_when_full_)
            {
                if (is_full())
                {
                    dropped_count_.fetch_add(1, std::memory_order_relaxed);
                    return InsertResult::FULL;
                }
            }
            else
            {
                if (push_timeout_ms_ == 0)
                {
                    while (is_full())
                        not_full_cv_.wait(lock);
                }
                else
                {
                    auto timeout = std::chrono::milliseconds(push_timeout_ms_);
                    if (!not_full_cv_.wait_for(lock, timeout, [&]
                                               { return !is_full(); }))
                        return InsertResult::TIMEOUT;
                }
            }

            uint32_t idx = head_;
            head_ = (head_ + 1) % static_cast<uint32_t>(capacity_limit_);

            // write data under producer lock to avoid consumer seeing a partially-written slot
            labels_[idx] = label;
            uint8_t *dst = data_.data() + (static_cast<size_t>(idx) * stride_);

            switch (data_type_)
            {
            case DataType::FLOAT32:
                std::memcpy(dst, vec, dim_ * sizeof(float));
                break;
            case DataType::FLOAT16:
            {
                uint16_t *hptr = reinterpret_cast<uint16_t *>(dst);
                for (size_t i = 0; i < dim_; ++i)
                    hptr[i] = fp32_to_fp16(vec[i]);
                break;
            }
            case DataType::INT8:
            {
                int8_t *iptr = reinterpret_cast<int8_t *>(dst);
                for (size_t i = 0; i < dim_; ++i)
                    iptr[i] = static_cast<int8_t>(std::clamp(vec[i], -128.0f, 127.0f));
                break;
            }
            case DataType::INT32:
            {
                int32_t *iptr = reinterpret_cast<int32_t *>(dst);
                for (size_t i = 0; i < dim_; ++i)
                    iptr[i] = static_cast<int32_t>(vec[i]);
                break;
            }
            case DataType::FLOAT64:
            {
                double *dptr = reinterpret_cast<double *>(dst);
                for (size_t i = 0; i < dim_; ++i)
                    dptr[i] = static_cast<double>(vec[i]);
                break;
            }
            default:
                std::memcpy(dst, vec, std::min(stride_, dim_ * sizeof(float)));
                break;
            }

            lock.unlock();
            not_empty_cv_.notify_one();
            return InsertResult::OK;
        }

        HotBatch swap_and_flush()
        {
            HotBatch batch;
            batch.dim = dim_;
            batch.element_size = element_size_;
            batch.data_type = data_type_;

            std::unique_lock<std::mutex> lock(prod_mutex_);
            if (head_ == tail_)
                return batch;

            uint32_t start = tail_;
            uint32_t end = head_;
            size_t count = 0;
            if (end >= start)
                count = static_cast<size_t>(end - start);
            else
                count = static_cast<size_t>(capacity_limit_ - start + end);

            if (count == 0)
                return batch;

            batch.labels.resize(count);
            batch.data.resize(count * stride_);

            if (end > start)
            {
                std::memcpy(batch.labels.data(), labels_.data() + start, count * sizeof(uint64_t));
                std::memcpy(batch.data.data(), data_.data() + static_cast<size_t>(start) * stride_, count * stride_);
            }
            else
            {
                uint32_t first = static_cast<uint32_t>(capacity_limit_ - start);
                std::memcpy(batch.labels.data(), labels_.data() + start, first * sizeof(uint64_t));
                std::memcpy(batch.data.data(), data_.data() + static_cast<size_t>(start) * stride_, first * stride_);

                uint32_t second = end;
                std::memcpy(batch.labels.data() + first, labels_.data(), second * sizeof(uint64_t));
                std::memcpy(batch.data.data() + static_cast<size_t>(first) * stride_, data_.data(), static_cast<size_t>(second) * stride_);
            }

            tail_ = end;
            lock.unlock();
            not_full_cv_.notify_all();
            return batch;
        }

        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k) const
        {
            if (!query || k == 0)
                return {};
            std::lock_guard<std::mutex> lock(prod_mutex_);
            uint32_t start = tail_;
            uint32_t end = head_;
            size_t count = 0;
            if (end >= start)
                count = static_cast<size_t>(end - start);
            else
                count = static_cast<size_t>(capacity_limit_ - start + end);

            if (count == 0)
                return {};

            using ScorePair = std::pair<float, uint64_t>;
            std::priority_queue<ScorePair> pq;
            auto dist_fn = get_pomai_l2sq_kernel();
            std::vector<float> scratch(dim_);

            for (size_t i = 0; i < count; ++i)
            {
                uint32_t idx = static_cast<uint32_t>((start + i) % capacity_limit_);
                const uint8_t *slot = data_.data() + (static_cast<size_t>(idx) * stride_);
                float dist;
                if (data_type_ == DataType::FLOAT32)
                {
                    dist = dist_fn(query, reinterpret_cast<const float *>(slot), dim_);
                }
                else
                {
                    decode_slot_to_float(slot, scratch.data());
                    dist = dist_fn(query, scratch.data(), dim_);
                }
                if (pq.size() < k)
                    pq.push({dist, labels_[idx]});
                else if (dist < pq.top().first)
                {
                    pq.pop();
                    pq.push({dist, labels_[idx]});
                }
            }

            std::vector<std::pair<uint64_t, float>> res;
            while (!pq.empty())
            {
                res.push_back({pq.top().second, pq.top().first});
                pq.pop();
            }
            std::reverse(res.begin(), res.end());
            return res;
        }

        size_t size() const
        {
            uint32_t h = head_;
            uint32_t t = tail_;
            if (h >= t)
                return static_cast<size_t>(h - t);
            return static_cast<size_t>(capacity_limit_ - t + h);
        }

        static uint64_t dropped_count() { return dropped_count_.load(std::memory_order_relaxed); }

    private:
        void decode_slot_to_float(const uint8_t *slot, float *out) const
        {
            switch (data_type_)
            {
            case DataType::FLOAT64:
            {
                const double *d = reinterpret_cast<const double *>(slot);
                for (size_t i = 0; i < dim_; ++i)
                    out[i] = static_cast<float>(d[i]);
                break;
            }
            case DataType::INT32:
            {
                const int32_t *p = reinterpret_cast<const int32_t *>(slot);
                for (size_t i = 0; i < dim_; ++i)
                    out[i] = static_cast<float>(p[i]);
                break;
            }
            case DataType::INT8:
            {
                const int8_t *p = reinterpret_cast<const int8_t *>(slot);
                for (size_t i = 0; i < dim_; ++i)
                    out[i] = static_cast<float>(p[i]);
                break;
            }
            case DataType::FLOAT16:
            {
                const uint16_t *p = reinterpret_cast<const uint16_t *>(slot);
                for (size_t i = 0; i < dim_; ++i)
                    out[i] = fp16_to_fp32(p[i]);
                break;
            }
            default:
                break;
            }
        }

        const size_t dim_;
        const DataType data_type_;
        const size_t capacity_limit_;
        uint32_t element_size_;
        size_t stride_;

        alignas(64) uint32_t head_;
        alignas(64) uint32_t tail_;

        alignas(64) static inline std::atomic<uint64_t> dropped_count_{0};

        mutable std::mutex prod_mutex_;
        std::condition_variable not_full_cv_;
        std::condition_variable not_empty_cv_;

        bool block_when_full_;
        uint32_t push_timeout_ms_;

        std::vector<uint64_t> labels_;
        std::vector<uint8_t> data_;
    };
}