#pragma once
/*
 * src/core/hot_tier.h
 *
 * Generic HotTier that stores raw bytes for different element types
 * (float32, float64, int32, int8, float16).
 *
 * - Push API accepts float* (common in-memory representation).
 * - HotTier converts to the configured storage type on push (hot-path).
 * - swap_and_flush() returns raw bytes + element_size + data_type enum for efficient downstream processing.
 *
 * Performance notes:
 * - data_ is a single contiguous std::vector<uint8_t> (SoA flattened).
 * - push() is zero-alloc hot-path: reserves capacity up-front and writes directly.
 * - swap_and_flush() uses move semantics (O(1)) to hand off buffers to background worker.
 */

#include <vector>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <queue>
#include <cstring>
#include <utility>
#include <thread>
#include <cstdint>
#include <string>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#endif

#include "src/core/config.h"
#include "src/core/types.h"       // DataType enum + helpers
#include "src/core/cpu_kernels.h" // fp16 helpers (fp32_to_fp16 / fp16_to_fp32)

namespace pomai::core
{
    // --- Helper: Adaptive Spinlock ---
    class AdaptiveSpinLock
    {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;

    public:
        void lock()
        {
            if (!flag.test_and_set(std::memory_order_acquire))
                return;
            int spin_count = 0;
            while (flag.test_and_set(std::memory_order_acquire))
            {
                if (spin_count < 16)
                {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
                    _mm_pause();
#elif defined(__aarch64__)
                    asm volatile("isb");
#endif
                    ++spin_count;
                }
                else
                {
                    std::this_thread::yield();
                    spin_count = 0;
                }
            }
        }

        void unlock() { flag.clear(std::memory_order_release); }
    };

    struct ScopedSpinLock
    {
        AdaptiveSpinLock &sl;
        ScopedSpinLock(AdaptiveSpinLock &l) : sl(l) { sl.lock(); }
        ~ScopedSpinLock() { sl.unlock(); }
    };

    // Generic HotBatch (drain result)
    struct HotBatch
    {
        std::vector<uint64_t> labels;
        // raw bytes: flattened vectors, layout = labels.size() * (element_size * dim)
        std::vector<uint8_t> data;
        size_t dim = 0;
        uint32_t element_size = 4;      // bytes per element
        pomai::core::DataType data_type = pomai::core::DataType::FLOAT32;

        bool empty() const { return labels.empty(); }
        size_t count() const { return labels.size(); }
    };

    class HotTier
    {
    public:
        // Constructor accepts DataType enum (preferred) or legacy string.
        HotTier(size_t dim, const pomai::config::HotTierConfig &cfg, pomai::core::DataType dt = pomai::core::DataType::FLOAT32)
            : dim_(dim),
              capacity_hint_(cfg.initial_capacity),
              data_type_(dt)
        {
            // FIX: use dtype_size from src/core/types.h (name unified)
            element_size_ = static_cast<uint32_t>(pomai::core::dtype_size(data_type_));
            labels_.reserve(capacity_hint_);
            data_.reserve(capacity_hint_ * dim_ * element_size_);
        }

        // Backwards-compatible constructor (string)
        HotTier(size_t dim, const pomai::config::HotTierConfig &cfg, const std::string &data_type)
            : HotTier(dim, cfg, str_to_dtype(data_type))
        {
        }

        HotTier(const HotTier &) = delete;
        HotTier &operator=(const HotTier &) = delete;

        // Push API accepts float* (most callers operate in float); conversion to storage type occurs here.
        // This keeps hot path simple for callers while allowing storage of other numeric types.
        void push(uint64_t label, const float *vec)
        {
            if (!vec) return;
            ScopedSpinLock g(lock_);

            labels_.push_back(label);

            size_t vec_bytes = dim_ * element_size_;
            size_t cur = data_.size();
            data_.resize(cur + vec_bytes);

            uint8_t *dst = data_.data() + cur;

            switch (data_type_)
            {
                case DataType::FLOAT32:
                    std::memcpy(dst, vec, dim_ * sizeof(float));
                    break;

                case DataType::FLOAT64:
                {
                    double *dptr = reinterpret_cast<double *>(dst);
                    for (size_t i = 0; i < dim_; ++i)
                        dptr[i] = static_cast<double>(vec[i]);
                    break;
                }

                case DataType::INT32:
                {
                    int32_t *iptr = reinterpret_cast<int32_t *>(dst);
                    for (size_t i = 0; i < dim_; ++i)
                        iptr[i] = static_cast<int32_t>(std::lrintf(vec[i])); // nearest integer
                    break;
                }

                case DataType::INT8:
                {
                    int8_t *iptr = reinterpret_cast<int8_t *>(dst);
                    for (size_t i = 0; i < dim_; ++i)
                    {
                        int32_t v = static_cast<int32_t>(std::lrintf(vec[i]));
                        v = std::max<int32_t>(-128, std::min<int32_t>(127, v));
                        iptr[i] = static_cast<int8_t>(v);
                    }
                    break;
                }

                case DataType::FLOAT16:
                {
                    uint16_t *hptr = reinterpret_cast<uint16_t *>(dst);
                    for (size_t i = 0; i < dim_; ++i)
                        hptr[i] = fp32_to_fp16(vec[i]);
                    break;
                }

                default:
                    // fallback to float32 copy
                    std::memcpy(dst, vec, std::min<size_t>(vec_bytes, dim_ * sizeof(float)));
                    break;
            }
        }

        // Alternative push: accept raw bytes in the target element format (useful if upstream already has matching type)
        void push_raw(uint64_t label, const void *raw_vec)
        {
            if (!raw_vec) return;
            ScopedSpinLock g(lock_);

            labels_.push_back(label);
            size_t vec_bytes = dim_ * element_size_;
            size_t cur = data_.size();
            data_.resize(cur + vec_bytes);
            std::memcpy(data_.data() + cur, raw_vec, vec_bytes);
        }

        // Drain: hand off ownership of buffers (zero-copy)
        HotBatch swap_and_flush()
        {
            HotBatch batch;
            batch.dim = dim_;
            batch.element_size = element_size_;
            batch.data_type = data_type_;

            ScopedSpinLock g(lock_);
            if (labels_.empty())
                return batch;

            batch.labels = std::move(labels_);
            batch.data = std::move(data_);

            // reinit internal buffers
            labels_ = std::vector<uint64_t>();
            data_ = std::vector<uint8_t>();
            labels_.reserve(capacity_hint_);
            data_.reserve(capacity_hint_ * dim_ * element_size_);

            return batch;
        }

        // Brute-force search over hot buffer. Query is float*; conversions applied on the fly.
        std::vector<std::pair<uint64_t, float>> search(const float *query, size_t k) const
        {
            std::vector<std::pair<uint64_t, float>> results;
            if (!query) return results;

            ScopedSpinLock g(lock_);
            size_t n = labels_.size();
            if (n == 0) return results;

            using Entry = std::pair<float, uint64_t>;
            std::priority_queue<Entry> pq;

            const uint8_t *dptr = data_.data();
            const size_t d = dim_;

            // Fast path: stored float32
            if (data_type_ == DataType::FLOAT32 && element_size_ == 4)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    const float *vec_i = reinterpret_cast<const float *>(dptr + i * d * element_size_);
                    float dist = 0.0f;
                    for (size_t j = 0; j < d; ++j)
                    {
                        float diff = query[j] - vec_i[j];
                        dist += diff * diff;
                    }
                    if (pq.size() < k) pq.push({dist, labels_[i]});
                    else if (dist < pq.top().first) { pq.pop(); pq.push({dist, labels_[i]}); }
                }
            }
            else
            {
                std::vector<float> tmp;
                tmp.resize(d);
                for (size_t i = 0; i < n; ++i)
                {
                    const uint8_t *slot = dptr + i * d * element_size_;
                    decode_slot_to_float(slot, tmp.data());
                    float dist = 0.0f;
                    for (size_t j = 0; j < d; ++j)
                    {
                        float diff = query[j] - tmp[j];
                        dist += diff * diff;
                    }
                    if (pq.size() < k) pq.push({dist, labels_[i]});
                    else if (dist < pq.top().first) { pq.pop(); pq.push({dist, labels_[i]}); }
                }
            }

            results.reserve(pq.size());
            while (!pq.empty())
            {
                results.emplace_back(pq.top().second, pq.top().first);
                pq.pop();
            }
            std::reverse(results.begin(), results.end());
            return results;
        }

        size_t size() const
        {
            ScopedSpinLock g(lock_);
            return labels_.size();
        }

        // Accessors
        pomai::core::DataType data_type_enum() const { return data_type_; }
        std::string data_type_string() const { return dtype_to_str(data_type_); }
        uint32_t element_size() const { return element_size_; }
        size_t dim() const { return dim_; }

    private:
        size_t dim_;
        size_t capacity_hint_;
        pomai::core::DataType data_type_;
        uint32_t element_size_;

        std::vector<uint64_t> labels_;
        std::vector<uint8_t> data_;

        mutable AdaptiveSpinLock lock_;

        static pomai::core::DataType str_to_dtype(const std::string &s)
        {
            if (s == "float64" || s == "FLOAT64" || s == "double") return DataType::FLOAT64;
            if (s == "int32" || s == "INT32") return DataType::INT32;
            if (s == "int8" || s == "INT8") return DataType::INT8;
            if (s == "float16" || s == "FLOAT16" || s == "fp16") return DataType::FLOAT16;
            return DataType::FLOAT32;
        }

        static std::string dtype_to_str(pomai::core::DataType dt)
        {
            switch (dt)
            {
                case DataType::FLOAT32: return "float32";
                case DataType::FLOAT64: return "float64";
                case DataType::INT32:   return "int32";
                case DataType::INT8:    return "int8";
                case DataType::FLOAT16: return "float16";
                default: return "float32";
            }
        }

        // Decode one slot (raw bytes at slot) into float buffer out (length dim_)
        void decode_slot_to_float(const uint8_t *slot, float *out) const
        {
            switch (data_type_)
            {
                case DataType::FLOAT32:
                    std::memcpy(out, slot, dim_ * sizeof(float));
                    break;

                case DataType::FLOAT64:
                {
                    const double *dp = reinterpret_cast<const double *>(slot);
                    for (size_t i = 0; i < dim_; ++i) out[i] = static_cast<float>(dp[i]);
                    break;
                }

                case DataType::INT32:
                {
                    const int32_t *ip = reinterpret_cast<const int32_t *>(slot);
                    for (size_t i = 0; i < dim_; ++i) out[i] = static_cast<float>(ip[i]);
                    break;
                }

                case DataType::INT8:
                {
                    const int8_t *ip = reinterpret_cast<const int8_t *>(slot);
                    for (size_t i = 0; i < dim_; ++i) out[i] = static_cast<float>(ip[i]);
                    break;
                }

                case DataType::FLOAT16:
                {
                    const uint16_t *hp = reinterpret_cast<const uint16_t *>(slot);
                    for (size_t i = 0; i < dim_; ++i)
                        out[i] = fp16_to_fp32(hp[i]);
                    break;
                }

                default:
                    std::memcpy(out, slot, std::min<size_t>(dim_ * sizeof(float), dim_ * element_size_));
                    break;
            }
        }
    };

} // namespace pomai::core