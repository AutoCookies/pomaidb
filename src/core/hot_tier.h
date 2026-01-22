#pragma once

#include <vector>
#include <atomic>
#include <mutex>
#include <cstring>
#include <algorithm>
#include <queue>
#include <cstdint>

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
        HotTier(size_t dim, const pomai::config::HotTierConfig &cfg, DataType dt = DataType::FLOAT32)
            : dim_(dim),
              data_type_(dt),
              capacity_limit_(cfg.initial_capacity),
              cursor_(0)
        {

            element_size_ = static_cast<uint32_t>(dtype_size(dt));
            stride_ = dim_ * element_size_;

            labels_.resize(capacity_limit_);
            data_.resize(capacity_hint() * stride_);
        }

        // --- PUBLIC METADATA ACCESSORS (Big Tech Standard) ---
        uint32_t element_size() const noexcept { return element_size_; }
        DataType data_type() const noexcept { return data_type_; }
        std::string data_type_string() const { return dtype_name(data_type_); }
        size_t dim() const noexcept { return dim_; }
        size_t capacity_hint() const noexcept { return capacity_limit_; }

        void push(uint64_t label, const float *vec)
        {
            if (!vec)
                return;
            std::lock_guard<std::mutex> lock(mutex_);

            if (cursor_ >= capacity_limit_)
            {
                dropped_count_.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            uint32_t idx = cursor_++;
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
        }

        HotBatch swap_and_flush()
        {
            HotBatch batch;
            batch.dim = dim_;
            batch.element_size = element_size_;
            batch.data_type = data_type_;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (cursor_ == 0)
                    return batch;

                size_t actual_vectors = cursor_;
                std::vector<uint64_t> old_labels;
                std::vector<uint8_t> old_data;

                old_labels.resize(actual_vectors);
                std::memcpy(old_labels.data(), labels_.data(), actual_vectors * sizeof(uint64_t));

                old_data.resize(actual_vectors * stride_);
                std::memcpy(old_data.data(), data_.data(), actual_vectors * stride_);

                batch.labels = std::move(old_labels);
                batch.data = std::move(old_data);
                cursor_ = 0;
            }
            return batch;
        }

        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k) const
        {
            if (!query || k == 0)
                return {};
            std::lock_guard<std::mutex> lock(mutex_);
            if (cursor_ == 0)
                return {};

            using ScorePair = std::pair<float, uint64_t>;
            std::priority_queue<ScorePair> pq;
            auto dist_fn = get_pomai_l2sq_kernel();
            std::vector<float> scratch(dim_);

            for (uint32_t i = 0; i < cursor_; ++i)
            {
                const uint8_t *slot = data_.data() + (static_cast<size_t>(i) * stride_);
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
                    pq.push({dist, labels_[i]});
                else if (dist < pq.top().first)
                {
                    pq.pop();
                    pq.push({dist, labels_[i]});
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

        size_t size() const { return cursor_.load(std::memory_order_relaxed); }
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
        alignas(64) std::atomic<uint32_t> cursor_;
        alignas(64) static inline std::atomic<uint64_t> dropped_count_{0};
        mutable std::mutex mutex_;
        std::vector<uint64_t> labels_;
        std::vector<uint8_t> data_;
    };
}