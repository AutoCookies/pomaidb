#include "seed.h"
#include "cpu_kernels.h"
#include "fixed_topk.h"
#include "memory_manager.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstring>

namespace pomai
{
    Seed::QuantizeRowFn Seed::quantize_row_impl_ = nullptr;

    std::uint8_t *Seed::AlignedAllocBytes(std::size_t bytes)
    {
        void *p = nullptr;
        if (bytes == 0)
            return nullptr;
        if (posix_memalign(&p, 64, ((bytes + 63) / 64) * 64) != 0)
            return nullptr;
        return reinterpret_cast<std::uint8_t *>(p);
    }

    void Seed::AlignedFreeBytes(std::uint8_t *p)
    {
        if (!p)
            return;
        free(p);
    }

    static void quantize_row_scalar(const float *src, const float *mins, const float *inv_scales, std::uint8_t *dst, std::size_t dim)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            float v = (src[d] - mins[d]) * inv_scales[d];
            int iv = static_cast<int>(std::nearbyint(v));
            if (iv < 0)
                iv = 0;
            if (iv > 255)
                iv = 255;
            dst[d] = static_cast<std::uint8_t>(iv);
        }
    }

    __attribute__((target("avx2"))) static void quantize_row_avx2(const float *src, const float *mins, const float *inv_scales, std::uint8_t *dst, std::size_t dim)
    {
        std::size_t d = 0;
        for (; d + 8 <= dim; d += 8)
        {
            __m256 v = _mm256_loadu_ps(src + d);
            __m256 m = _mm256_loadu_ps(mins + d);
            __m256 s = _mm256_loadu_ps(inv_scales + d);
            __m256 res = _mm256_mul_ps(_mm256_sub_ps(v, m), s);
            __m256i qi = _mm256_cvtps_epi32(_mm256_round_ps(res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
            alignas(32) int32_t tmp[8];
            _mm256_storeu_si256((__m256i *)tmp, qi);
            for (int k = 0; k < 8; ++k)
            {
                int iv = tmp[k];
                if (iv < 0)
                    iv = 0;
                if (iv > 255)
                    iv = 255;
                dst[d + k] = static_cast<std::uint8_t>(iv);
            }
        }
        if (d < dim)
            quantize_row_scalar(src + d, mins + d, inv_scales + d, dst + d, dim - d);
    }

    __attribute__((target("avx512f"))) static void quantize_row_avx512(const float *src, const float *mins, const float *inv_scales, std::uint8_t *dst, std::size_t dim)
    {
        std::size_t d = 0;
        for (; d + 16 <= dim; d += 16)
        {
            __m512 v = _mm512_loadu_ps(src + d);
            __m512 m = _mm512_loadu_ps(mins + d);
            __m512 s = _mm512_loadu_ps(inv_scales + d);
            __m512 res = _mm512_mul_ps(_mm512_sub_ps(v, m), s);
            __m512i qi = _mm512_cvt_roundps_epi32(res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            alignas(64) int32_t tmp[16];
            _mm512_storeu_si512((__m512i *)tmp, qi);
            for (int k = 0; k < 16; ++k)
            {
                int iv = tmp[k];
                if (iv < 0)
                    iv = 0;
                if (iv > 255)
                    iv = 255;
                dst[d + k] = static_cast<std::uint8_t>(iv);
            }
        }
        if (d < dim)
            quantize_row_scalar(src + d, mins + d, inv_scales + d, dst + d, dim - d);
    }

    namespace
    {
        using LocalQuantizeFn = void (*)(const float *, const float *, const float *, std::uint8_t *, std::size_t);
        static LocalQuantizeFn SelectKernelLocal()
        {
#if defined(__GNUC__) || defined(__clang__)
            if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))
                return quantize_row_avx512;
            if (__builtin_cpu_supports("avx2"))
                return quantize_row_avx2;
#endif
            return quantize_row_scalar;
        }
    }

    void Seed::InitQuantizeDispatch()
    {
        if (quantize_row_impl_)
            return;
        LocalQuantizeFn k = SelectKernelLocal();
        quantize_row_impl_ = reinterpret_cast<QuantizeRowFn>(k);
    }

    Seed::Seed(std::size_t dim) : dim_(dim)
    {
        if (dim_ == 0)
            throw std::runtime_error("Seed dim must be > 0");
        ids_.reserve(1024);
        qmins_.assign(dim, std::numeric_limits<float>::infinity());
        qmaxs_.assign(dim, -std::numeric_limits<float>::infinity());
        qscales_.assign(dim, 1.0f);
        qinv_scales_.assign(dim, 1.0f);
        sample_buf_.reserve(sample_threshold_ * dim_);
        sample_ids_.reserve(sample_threshold_);
        InitQuantizeDispatch();
    }

    Seed::Seed(const Seed &other)
        : dim_(other.dim_), ids_(other.ids_), qdata_(other.qdata_), qmins_(other.qmins_), qmaxs_(other.qmaxs_), qscales_(other.qscales_), qinv_scales_(other.qinv_scales_), pos_(other.pos_), accounted_bytes_(other.accounted_bytes_), qrows_(other.qrows_), qcap_(other.qcap_), sample_buf_(other.sample_buf_), sample_ids_(other.sample_ids_), sample_rows_(other.sample_rows_), calibrated_(other.calibrated_), is_fixed_(other.is_fixed_), total_ingested_(other.total_ingested_), fixed_bounds_after_(other.fixed_bounds_after_), out_of_range_rows_(other.out_of_range_rows_.load(std::memory_order_relaxed))
    {
        if (accounted_bytes_ > 0)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Memtable, accounted_bytes_);
    }

    Seed &Seed::operator=(const Seed &other)
    {
        if (this == &other)
            return *this;
        ReleaseMemtableAccounting();
        dim_ = other.dim_;
        ids_ = other.ids_;
        qdata_ = other.qdata_;
        qmins_ = other.qmins_;
        qmaxs_ = other.qmaxs_;
        qscales_ = other.qscales_;
        qinv_scales_ = other.qinv_scales_;
        pos_ = other.pos_;
        accounted_bytes_ = other.accounted_bytes_;
        qrows_ = other.qrows_;
        qcap_ = other.qcap_;
        sample_buf_ = other.sample_buf_;
        sample_ids_ = other.sample_ids_;
        sample_rows_ = other.sample_rows_;
        calibrated_ = other.calibrated_;
        is_fixed_ = other.is_fixed_;
        total_ingested_ = other.total_ingested_;
        fixed_bounds_after_ = other.fixed_bounds_after_;
        out_of_range_rows_.store(other.out_of_range_rows_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        if (accounted_bytes_ > 0)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Memtable, accounted_bytes_);
        return *this;
    }

    Seed::Seed(Seed &&other) noexcept
        : dim_(other.dim_), ids_(std::move(other.ids_)), qdata_(std::move(other.qdata_)), qmins_(std::move(other.qmins_)), qmaxs_(std::move(other.qmaxs_)), qscales_(std::move(other.qscales_)), qinv_scales_(std::move(other.qinv_scales_)), pos_(std::move(other.pos_)), accounted_bytes_(other.accounted_bytes_), qrows_(other.qrows_), qcap_(other.qcap_), sample_buf_(std::move(other.sample_buf_)), sample_ids_(std::move(other.sample_ids_)), sample_rows_(other.sample_rows_), calibrated_(other.calibrated_), is_fixed_(other.is_fixed_), total_ingested_(other.total_ingested_), fixed_bounds_after_(other.fixed_bounds_after_), out_of_range_rows_(other.out_of_range_rows_.load(std::memory_order_relaxed))
    {
        other.accounted_bytes_ = 0;
        other.out_of_range_rows_.store(0, std::memory_order_relaxed);
    }

    Seed &Seed::operator=(Seed &&other) noexcept
    {
        if (this == &other)
            return *this;
        ReleaseMemtableAccounting();
        dim_ = other.dim_;
        ids_ = std::move(other.ids_);
        qdata_ = std::move(other.qdata_);
        qmins_ = std::move(other.qmins_);
        qmaxs_ = std::move(other.qmaxs_);
        qscales_ = std::move(other.qscales_);
        qinv_scales_ = std::move(other.qinv_scales_);
        pos_ = std::move(other.pos_);
        accounted_bytes_ = other.accounted_bytes_;
        qrows_ = other.qrows_;
        qcap_ = other.qcap_;
        sample_buf_ = std::move(other.sample_buf_);
        sample_ids_ = std::move(other.sample_ids_);
        sample_rows_ = other.sample_rows_;
        calibrated_ = other.calibrated_;
        is_fixed_ = other.is_fixed_;
        total_ingested_ = other.total_ingested_;
        fixed_bounds_after_ = other.fixed_bounds_after_;
        out_of_range_rows_.store(other.out_of_range_rows_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.accounted_bytes_ = 0;
        other.out_of_range_rows_.store(0, std::memory_order_relaxed);
        return *this;
    }

    Seed::~Seed()
    {
        ReleaseMemtableAccounting();
    }

    void Seed::ReserveForAppend(std::size_t add_rows)
    {
        const std::size_t need_rows = ids_.size() + add_rows;
        if (ids_.capacity() < need_rows)
            ids_.reserve(std::max<std::size_t>(need_rows, ids_.capacity() + ids_.capacity() / 2 + 1024));
        std::size_t need_qbytes = (qrows_ + add_rows) * dim_;
        if (need_qbytes > qcap_)
        {
            std::size_t newcap = std::max<std::size_t>(need_qbytes, qcap_ + qcap_ / 2 + dim_ * 1024);
            qdata_.reserve((newcap + 63) / 64 * 64);
            qcap_ = newcap;
        }
    }

    void Seed::UpdateMemtableAccounting()
    {
        const std::size_t bytes = ids_.size() * sizeof(Id) + qrows_ * dim_ * sizeof(std::uint8_t) + (qmins_.size() + qscales_.size()) * sizeof(float);
        if (bytes == accounted_bytes_)
            return;
        if (bytes > accounted_bytes_)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Memtable, bytes - accounted_bytes_);
        else
            MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Memtable, accounted_bytes_ - bytes);
        accounted_bytes_ = bytes;
    }

    void Seed::ReleaseMemtableAccounting()
    {
        if (accounted_bytes_ > 0)
            MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Memtable, accounted_bytes_);
        accounted_bytes_ = 0;
    }

    void Seed::EnsureCalibration()
    {
        if (calibrated_)
            return;
        if (is_fixed_)
            return;
        if (sample_rows_ < sample_threshold_)
            return;
        FinalizeCalibrationAndQuantizeSamples();
    }

    void Seed::FinalizeCalibrationAndQuantizeSamples()
    {
        for (std::size_t d = 0; d < dim_; ++d)
        {
            float mn = std::numeric_limits<float>::infinity();
            float mx = -std::numeric_limits<float>::infinity();
            for (std::size_t r = 0; r < sample_rows_; ++r)
            {
                float v = sample_buf_[r * dim_ + d];
                if (v < mn)
                    mn = v;
                if (v > mx)
                    mx = v;
            }
            qmins_[d] = mn;
            qmaxs_[d] = mx;
            float range = mx - mn;
            qscales_[d] = (range > 0.0f) ? (range / 255.0f) : 1.0f;
            qinv_scales_[d] = 1.0f / qscales_[d];
        }
        qrows_ = sample_rows_;
        qdata_.resize(qrows_ * dim_);
        for (std::size_t r = 0; r < qrows_; ++r)
        {
            const float *src = sample_buf_.data() + r * dim_;
            std::uint8_t *dst = qdata_.data() + r * dim_;
            quantize_row_impl_(src, qmins_.data(), qinv_scales_.data(), dst, dim_);
        }
        sample_buf_.clear();
        sample_ids_.clear();
        calibrated_ = true;
        UpdateMemtableAccounting();
    }

    void Seed::RescaleAll(const std::vector<float> &new_mins, const std::vector<float> &new_scales)
    {
        if (qrows_ == 0)
            return;
        std::vector<float> new_inv(dim_);
        for (std::size_t d = 0; d < dim_; ++d)
            new_inv[d] = 1.0f / new_scales[d];
        for (std::size_t r = 0; r < qrows_; ++r)
        {
            std::uint8_t *qptr = qdata_.data() + r * dim_;
            for (std::size_t d = 0; d < dim_; ++d)
            {
                float f = qmins_[d] + qscales_[d] * static_cast<float>(qptr[d]);
                float v = (f - new_mins[d]) * new_inv[d];
                int iv = static_cast<int>(std::nearbyint(v));
                if (iv < 0)
                    iv = 0;
                if (iv > 255)
                    iv = 255;
                qptr[d] = static_cast<std::uint8_t>(iv);
            }
        }
        qmins_ = new_mins;
        qscales_ = new_scales;
        for (std::size_t d = 0; d < dim_; ++d)
            qinv_scales_[d] = 1.0f / qscales_[d];
    }

    void Seed::ApplyUpserts(const std::vector<UpsertRequest> &batch)
    {
        if (batch.empty())
            return;
        InitQuantizeDispatch();
        if (!is_fixed_)
        {
            std::vector<float> new_mins = qmins_;
            std::vector<float> new_maxs = qmaxs_;
            bool changed = false;
            for (const auto &req : batch)
            {
                if (req.vec.data.size() != dim_)
                    continue;
                for (std::size_t d = 0; d < dim_; ++d)
                {
                    float v = req.vec.data[d];
                    if (v < new_mins[d])
                    {
                        new_mins[d] = v;
                        changed = true;
                    }
                    if (v > new_maxs[d])
                    {
                        new_maxs[d] = v;
                        changed = true;
                    }
                }
            }
            if (changed)
            {
                std::vector<float> new_scales(dim_);
                for (std::size_t d = 0; d < dim_; ++d)
                {
                    float range = new_maxs[d] - new_mins[d];
                    new_scales[d] = (range > 0.0f) ? (range / 255.0f) : 1.0f;
                }
                if (qrows_ > 0)
                {
                    RescaleAll(new_mins, new_scales);
                    qmaxs_ = new_maxs;
                }
                else
                {
                    qmins_ = new_mins;
                    qmaxs_ = new_maxs;
                    qscales_ = new_scales;
                    qinv_scales_.resize(dim_);
                    for (std::size_t d = 0; d < dim_; ++d)
                        qinv_scales_[d] = 1.0f / qscales_[d];
                }
            }
        }

        std::size_t valid_rows = 0;
        for (const auto &req : batch)
        {
            if (req.vec.data.size() != dim_)
                continue;
            auto it = pos_.find(req.id);
            std::uint32_t row;
            if (it == pos_.end())
            {
                row = static_cast<std::uint32_t>(ids_.size());
                ids_.push_back(req.id);
                pos_[req.id] = row;
                if (qrows_ <= row)
                {
                    qrows_ = row + 1;
                    qdata_.resize(qrows_ * dim_);
                }
            }
            else
                row = it->second;

            bool out_of_range = false;
            if (is_fixed_)
            {
                for (std::size_t d = 0; d < dim_; ++d)
                {
                    float v = req.vec.data[d];
                    if (v < qmins_[d] || v > qmaxs_[d])
                    {
                        out_of_range = true;
                        break;
                    }
                }
            }
            if (out_of_range)
                out_of_range_rows_.fetch_add(1, std::memory_order_relaxed);

            const float *src = req.vec.data.data();
            std::uint8_t *dst = qdata_.data() + static_cast<std::size_t>(row) * dim_;
            quantize_row_impl_(src, qmins_.data(), qinv_scales_.data(), dst, dim_);
            ++valid_rows;
        }
        if (!is_fixed_ && valid_rows > 0)
        {
            total_ingested_ += valid_rows;
            if (total_ingested_ >= fixed_bounds_after_)
                SetFixedBounds(qmins_, qmaxs_);
        }
        UpdateMemtableAccounting();
    }

    Seed::Snapshot Seed::MakeSnapshot() const
    {
        auto out = std::make_shared<Store>();
        out->dim = dim_;
        out->ids = ids_;
        out->qdata = qdata_;
        out->qmins = qmins_;
        out->qscales = qscales_;
        out->is_quantized.store(true, std::memory_order_release);
        out->accounted_bytes = out->ids.size() * sizeof(Id) + out->qdata.size() * sizeof(std::uint8_t) + (out->qmins.size() + out->qscales.size()) * sizeof(float);
        if (out->accounted_bytes > 0)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Search, out->accounted_bytes);
        return out;
    }

    void Seed::Quantize(Snapshot snap)
    {
        if (!snap)
            return;
        if (snap->is_quantized.load(std::memory_order_acquire))
            return;
        const std::size_t n = snap->ids.size();
        const std::size_t dim = snap->dim;
        if (n == 0)
            return;
        std::vector<float> inv(dim);
        for (std::size_t d = 0; d < dim; ++d)
            inv[d] = 1.0f / snap->qscales[d];
        for (std::size_t r = 0; r < n; ++r)
            quantize_row_impl_(nullptr, snap->qmins.data(), inv.data(), snap->qdata.data() + r * dim, dim);
        snap->is_quantized.store(true, std::memory_order_release);
    }

    void Seed::DequantizeRow(const Snapshot &snap, std::size_t row, float *out)
    {
        const std::uint8_t *src = snap->qdata.data() + row * snap->dim;
        for (std::size_t d = 0; d < snap->dim; ++d)
            out[d] = snap->qmins[d] + snap->qscales[d] * static_cast<float>(src[d]);
    }

    std::vector<float> Seed::DequantizeSnapshot(const Snapshot &snap)
    {
        if (!snap || snap->ids.empty())
            return {};
        const std::size_t n = snap->ids.size();
        const std::size_t dim = snap->dim;
        std::vector<float> out(n * dim);
        for (std::size_t r = 0; r < n; ++r)
            DequantizeRow(snap, r, out.data() + r * dim);
        return out;
    }

    void Seed::SetFixedBounds(const std::vector<float> &mins, const std::vector<float> &maxs)
    {
        if (mins.size() != dim_ || maxs.size() != dim_)
            throw std::runtime_error("SetFixedBounds dim mismatch");
        qmins_ = mins;
        qmaxs_ = maxs;
        qscales_.resize(dim_);
        qinv_scales_.resize(dim_);
        for (std::size_t d = 0; d < dim_; ++d)
        {
            float range = qmaxs_[d] - qmins_[d];
            qscales_[d] = (range > 0.0f) ? (range / 255.0f) : 1.0f;
            qinv_scales_[d] = 1.0f / qscales_[d];
        }
        is_fixed_ = true;
    }

    void Seed::InheritBounds(const Seed &other)
    {
        if (other.dim_ != dim_)
            throw std::runtime_error("InheritBounds dim mismatch");
        qmins_ = other.qmins_;
        qmaxs_ = other.qmaxs_;
        qscales_ = other.qscales_;
        qinv_scales_ = other.qinv_scales_;
        is_fixed_ = other.is_fixed_;
        total_ingested_ = other.total_ingested_;
        fixed_bounds_after_ = other.fixed_bounds_after_;
    }

    void Seed::SetFixedBoundsAfterCount(std::size_t count)
    {
        fixed_bounds_after_ = std::max<std::size_t>(1, count);
    }

    std::uint64_t Seed::ConsumeOutOfRangeCount()
    {
        return out_of_range_rows_.exchange(0, std::memory_order_relaxed);
    }

    SearchResponse Seed::SearchSnapshot(const Snapshot &snap, const SearchRequest &req)
    {
        SearchResponse resp;
        if (!snap || snap->dim == 0 || req.query.data.size() != snap->dim)
            return resp;
        const std::size_t dim = snap->dim;
        const std::size_t n = snap->ids.size();
        if (n == 0)
            return resp;
        std::vector<std::uint8_t> qquant(dim);
        for (std::size_t d = 0; d < dim; ++d)
        {
            float qv = (snap->qscales[d] > 0.0f) ? ((req.query.data[d] - snap->qmins[d]) / snap->qscales[d]) : 0.0f;
            qquant[d] = static_cast<std::uint8_t>(std::clamp<int>(static_cast<int>(std::nearbyint(qv)), 0, 255));
        }
        FixedTopK candidate_topk(128);
        const std::uint8_t *qbase = snap->qdata.data();
        for (std::size_t row = 0; row < n; ++row)
        {
            float d = kernels::L2Sqr_SQ8_AVX2(qbase + row * dim, qquant.data(), dim);
            candidate_topk.Push(-d, static_cast<Id>(row));
        }
        FixedTopK final_topk(req.topk);
        std::vector<float> dequant(dim);
        const auto *candidates = candidate_topk.Data();
        for (std::size_t i = 0; i < candidate_topk.Size(); ++i)
        {
            std::size_t cand_row = static_cast<std::size_t>(candidates[i].id);
            DequantizeRow(snap, cand_row, dequant.data());
            final_topk.Push(-kernels::L2Sqr(dequant.data(), req.query.data.data(), dim), snap->ids[cand_row]);
        }
        final_topk.FillSorted(resp.items);
        return resp;
    }
}
