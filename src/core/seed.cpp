#include "seed.h"
#include "cpu_kernels.h"
#include "fixed_topk.h"
#include "memory_manager.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace pomai
{
    namespace
    {
        using QuantizeFn = void (*)(const float *, const float *, const float *, std::uint8_t *, std::size_t);

        inline bool IsAligned(const void *ptr, std::size_t alignment)
        {
            return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
        }

        void Quantize_Scalar(const float *src, const float *mins, const float *inv_scales, std::uint8_t *dst, std::size_t dim)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                float qv = (src[d] - mins[d]) * inv_scales[d];
                int v = static_cast<int>(std::nearbyint(qv));
                dst[d] = static_cast<std::uint8_t>(std::clamp(v, 0, 255));
            }
        }

        __attribute__((target("avx2"))) void Quantize_AVX2(const float *src, const float *mins, const float *inv_scales, std::uint8_t *dst, std::size_t dim)
        {
            std::size_t d = 0;
            for (; d + 32 <= dim; d += 32)
            {
                alignas(32) std::uint8_t out[32];
                for (int chunk = 0; chunk < 4; ++chunk)
                {
                    const std::size_t off = d + static_cast<std::size_t>(chunk) * 8;
                    __m256 v = _mm256_loadu_ps(src + off);
                    __m256 vmin = _mm256_loadu_ps(mins + off);
                    __m256 vinv = _mm256_loadu_ps(inv_scales + off);
                    __m256 res = _mm256_mul_ps(_mm256_sub_ps(v, vmin), vinv);
                    __m256i qi = _mm256_cvtps_epi32(_mm256_round_ps(res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    alignas(32) int32_t ints[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ints), qi);
                    for (int k = 0; k < 8; ++k)
                        out[chunk * 8 + k] = static_cast<std::uint8_t>(std::clamp(ints[k], 0, 255));
                }
                if (IsAligned(dst + d, 32))
                    _mm256_stream_si256(reinterpret_cast<__m256i *>(dst + d), _mm256_load_si256(reinterpret_cast<const __m256i *>(out)));
                else
                    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + d), _mm256_load_si256(reinterpret_cast<const __m256i *>(out)));
            }
            for (; d < dim; ++d)
            {
                float qv = (src[d] - mins[d]) * inv_scales[d];
                int v = static_cast<int>(std::nearbyint(qv));
                dst[d] = static_cast<std::uint8_t>(std::clamp(v, 0, 255));
            }
        }

        __attribute__((target("avx512f"))) void Quantize_AVX512(const float *src, const float *mins, const float *inv_scales, std::uint8_t *dst, std::size_t dim)
        {
            std::size_t d = 0;
            for (; d + 64 <= dim; d += 64)
            {
                alignas(64) std::uint8_t out[64];
                for (int chunk = 0; chunk < 4; ++chunk)
                {
                    const std::size_t off = d + static_cast<std::size_t>(chunk) * 16;
                    __m512 v = _mm512_loadu_ps(src + off);
                    __m512 vmin = _mm512_loadu_ps(mins + off);
                    __m512 vinv = _mm512_loadu_ps(inv_scales + off);
                    __m512 res = _mm512_mul_ps(_mm512_sub_ps(v, vmin), vinv);
                    __m512i qi = _mm512_cvtps_epi32(_mm512_round_ps(res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                    alignas(64) int32_t ints[16];
                    _mm512_storeu_si512(reinterpret_cast<void *>(ints), qi);
                    for (int k = 0; k < 16; ++k)
                        out[chunk * 16 + k] = static_cast<std::uint8_t>(std::clamp(ints[k], 0, 255));
                }
                if (IsAligned(dst + d, 64))
                    _mm512_stream_si512(reinterpret_cast<void *>(dst + d), _mm512_load_si512(reinterpret_cast<const void *>(out)));
                else
                    _mm512_storeu_si512(reinterpret_cast<void *>(dst + d), _mm512_load_si512(reinterpret_cast<const void *>(out)));
            }
            for (; d < dim; ++d)
            {
                float qv = (src[d] - mins[d]) * inv_scales[d];
                int v = static_cast<int>(std::nearbyint(qv));
                dst[d] = static_cast<std::uint8_t>(std::clamp(v, 0, 255));
            }
        }

        QuantizeFn SelectQuantizeFn()
        {
            if (__builtin_cpu_supports("avx512f"))
                return &Quantize_AVX512;
            if (__builtin_cpu_supports("avx2"))
                return &Quantize_AVX2;
            return &Quantize_Scalar;
        }

        const QuantizeFn kQuantizeFn = SelectQuantizeFn();
    }

    Seed::Seed(std::size_t dim) : dim_(dim)
    {
        if (dim_ == 0)
            throw std::runtime_error("Seed dim must be > 0");
        ids_.reserve(1024);
        qdata_.reserve(1024 * dim_);
        qmins_.assign(dim_, std::numeric_limits<float>::infinity());
        qmaxs_.assign(dim_, -std::numeric_limits<float>::infinity());
        qscales_.assign(dim_, 1.0f);
        qinv_scales_.assign(dim_, 1.0f);
    }

    Seed::Seed(const Seed &other)
        : dim_(other.dim_), ids_(other.ids_), qdata_(other.qdata_), qmins_(other.qmins_), qmaxs_(other.qmaxs_), qscales_(other.qscales_), qinv_scales_(other.qinv_scales_), pos_(other.pos_), accounted_bytes_(other.accounted_bytes_)
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
        if (accounted_bytes_ > 0)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Memtable, accounted_bytes_);
        return *this;
    }

    Seed::Seed(Seed &&other) noexcept
        : dim_(other.dim_), ids_(std::move(other.ids_)), qdata_(std::move(other.qdata_)), qmins_(std::move(other.qmins_)), qmaxs_(std::move(other.qmaxs_)), qscales_(std::move(other.qscales_)), qinv_scales_(std::move(other.qinv_scales_)), pos_(std::move(other.pos_)), accounted_bytes_(other.accounted_bytes_)
    {
        other.accounted_bytes_ = 0;
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
        other.accounted_bytes_ = 0;
        return *this;
    }

    Seed::~Seed() { ReleaseMemtableAccounting(); }

    void Seed::ReserveForAppend(std::size_t add_rows)
    {
        const std::size_t need_rows = ids_.size() + add_rows;
        if (ids_.capacity() < need_rows)
        {
            std::size_t new_cap = std::max<std::size_t>(need_rows, ids_.capacity() + ids_.capacity() / 2 + 1024);
            ids_.reserve(new_cap);
            qdata_.reserve(new_cap * dim_);
        }
    }

    void Seed::ApplyUpserts(const std::vector<UpsertRequest> &batch)
    {
        if (batch.empty())
            return;
        if (qmins_.size() != dim_)
        {
            qmins_.assign(dim_, std::numeric_limits<float>::infinity());
            qmaxs_.assign(dim_, -std::numeric_limits<float>::infinity());
            qscales_.assign(dim_, 1.0f);
            qinv_scales_.assign(dim_, 1.0f);
        }
        std::size_t new_cnt = 0;
        for (const auto &r : batch)
        {
            if (r.vec.data.size() != dim_)
                continue;
            if (pos_.find(r.id) == pos_.end())
                ++new_cnt;
        }
        if (new_cnt)
            ReserveForAppend(new_cnt);
        for (const auto &r : batch)
        {
            if (r.vec.data.size() != dim_)
                continue;
            for (std::size_t d = 0; d < dim_; ++d)
            {
                float v = r.vec.data[d];
                if (v < qmins_[d])
                    qmins_[d] = v;
                if (v > qmaxs_[d])
                    qmaxs_[d] = v;
                float range = qmaxs_[d] - qmins_[d];
                float scale = (range > 0.0f) ? (range / 255.0f) : 1.0f;
                qscales_[d] = scale;
                qinv_scales_[d] = 1.0f / scale;
            }
            auto it = pos_.find(r.id);
            if (it == pos_.end())
            {
                const std::uint32_t row = static_cast<std::uint32_t>(ids_.size());
                ids_.push_back(r.id);
                pos_.emplace(r.id, row);
                const std::size_t base = static_cast<std::size_t>(row) * dim_;
                qdata_.resize(base + dim_);
                kQuantizeFn(r.vec.data.data(), qmins_.data(), qinv_scales_.data(), qdata_.data() + base, dim_);
            }
            else
            {
                const std::uint32_t row = it->second;
                const std::size_t base = static_cast<std::size_t>(row) * dim_;
                kQuantizeFn(r.vec.data.data(), qmins_.data(), qinv_scales_.data(), qdata_.data() + base, dim_);
            }
        }
        UpdateMemtableAccounting();
    }

    Seed::Snapshot Seed::MakeSnapshot() const
    {
        auto out = std::shared_ptr<Store>(new Store(), [](Store *s)
                                          {
            if (s->accounted_bytes > 0) MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Search, s->accounted_bytes);
            delete s; });
        out->dim = dim_;
        out->ids = ids_;
        out->qdata = qdata_;
        out->qmins = qmins_;
        out->qscales = qscales_;
        out->is_quantized.store(true, std::memory_order_release);
        out->accounted_bytes = out->ids.size() * sizeof(Id) + out->qdata.size() * sizeof(std::uint8_t) + out->qmins.size() * sizeof(float) + out->qscales.size() * sizeof(float);
        if (out->accounted_bytes > 0)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Search, out->accounted_bytes);
        return out;
    }

    void Seed::Quantize(Snapshot snap)
    {
        if (!snap)
            return;
        snap->is_quantized.store(true, std::memory_order_release);
    }

    void Seed::DequantizeRow(const Snapshot &snap, std::size_t row, float *out)
    {
        const std::size_t dim = snap->dim;
        const std::uint8_t *src = snap->qdata.data() + row * dim;
        for (std::size_t d = 0; d < dim; ++d)
            out[d] = snap->qmins[d] + snap->qscales[d] * static_cast<float>(src[d]);
    }

    std::vector<float> Seed::DequantizeSnapshot(const Snapshot &snap)
    {
        if (!snap || snap->ids.empty())
            return {};
        const std::size_t n = snap->ids.size();
        const std::size_t dim = snap->dim;
        std::vector<float> out(n * dim);
        for (std::size_t row = 0; row < n; ++row)
        {
            float *dst = out.data() + row * dim;
            DequantizeRow(snap, row, dst);
        }
        return out;
    }

    SearchResponse Seed::SearchSnapshot(const Snapshot &snap, const SearchRequest &req)
    {
        SearchResponse resp;
        if (!snap || snap->dim == 0 || req.query.data.size() != snap->dim || snap->qdata.empty())
            return resp;
        const std::size_t dim = snap->dim;
        const std::size_t n = snap->ids.size();
        if (n == 0)
            return resp;

        constexpr std::size_t kOversample = 128;
        const float *q = req.query.data.data();

        std::vector<std::uint8_t> qquant(dim);
        for (std::size_t d = 0; d < dim; ++d)
        {
            float qv = (snap->qscales[d] > 0.0f) ? ((q[d] - snap->qmins[d]) / snap->qscales[d]) : 0.0f;
            qquant[d] = static_cast<std::uint8_t>(std::clamp<int>(std::nearbyint(qv), 0, 255));
        }

        FixedTopK candidate_topk(kOversample);
        const std::uint8_t *qbase = snap->qdata.data();
        for (std::size_t row = 0; row < n; ++row)
        {
            float d = kernels::L2Sqr_SQ8_AVX2(qbase + row * dim, qquant, dim);
            candidate_topk.Push(-d, static_cast<Id>(row));
        }

        FixedTopK final_topk(req.topk);
        std::vector<float> dequant(dim);
        const auto *candidates = candidate_topk.Data();
        for (std::size_t i = 0; i < candidate_topk.Size(); ++i)
        {
            std::size_t cand_row = static_cast<std::size_t>(candidates[i].id);
            DequantizeRow(snap, cand_row, dequant.data());
            final_topk.Push(-kernels::L2Sqr(dequant.data(), q, dim), snap->ids[cand_row]);
        }
        final_topk.FillSorted(resp.items);
        return resp;
    }

    void Seed::UpdateMemtableAccounting()
    {
        const std::size_t bytes = ids_.size() * sizeof(Id) + qdata_.size() * sizeof(std::uint8_t) + qmins_.size() * sizeof(float) + qmaxs_.size() * sizeof(float) + qscales_.size() * sizeof(float) + qinv_scales_.size() * sizeof(float);
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
        if (accounted_bytes_ == 0)
            return;
        MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Memtable, accounted_bytes_);
        accounted_bytes_ = 0;
    }
}
