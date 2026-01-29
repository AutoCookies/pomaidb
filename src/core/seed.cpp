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

namespace pomai
{
    Seed::Seed(std::size_t dim) : dim_(dim)
    {
        if (dim_ == 0)
            throw std::runtime_error("Seed dim must be > 0");
        ids_.reserve(1024);
        data_.reserve(1024 * dim_);
    }

    Seed::Seed(const Seed &other)
        : dim_(other.dim_), ids_(other.ids_), data_(other.data_), pos_(other.pos_), accounted_bytes_(other.accounted_bytes_)
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
        data_ = other.data_;
        pos_ = other.pos_;
        accounted_bytes_ = other.accounted_bytes_;
        if (accounted_bytes_ > 0)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Memtable, accounted_bytes_);
        return *this;
    }

    Seed::Seed(Seed &&other) noexcept
        : dim_(other.dim_), ids_(std::move(other.ids_)), data_(std::move(other.data_)), pos_(std::move(other.pos_)), accounted_bytes_(other.accounted_bytes_)
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
        data_ = std::move(other.data_);
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
            data_.reserve(new_cap * dim_);
        }
    }

    void Seed::ApplyUpserts(const std::vector<UpsertRequest> &batch)
    {
        if (batch.empty())
            return;
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
            auto it = pos_.find(r.id);
            if (it == pos_.end())
            {
                const std::uint32_t row = static_cast<std::uint32_t>(ids_.size());
                ids_.push_back(r.id);
                pos_.emplace(r.id, row);
                const std::size_t base = static_cast<std::size_t>(row) * dim_;
                data_.resize(base + dim_);
                std::copy(r.vec.data.begin(), r.vec.data.end(), data_.begin() + base);
            }
            else
            {
                const std::uint32_t row = it->second;
                const std::size_t base = static_cast<std::size_t>(row) * dim_;
                std::copy(r.vec.data.begin(), r.vec.data.end(), data_.begin() + base);
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
        out->data = data_;
        out->accounted_bytes = out->ids.size() * sizeof(Id) + out->data.size() * sizeof(float);
        if (out->accounted_bytes > 0)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Search, out->accounted_bytes);
        return out;
    }

    void Seed::Quantize(Snapshot snap)
    {
        if (snap->is_quantized.exchange(true))
            return;
        const std::size_t n = snap->ids.size();
        const std::size_t dim = snap->dim;
        if (n == 0)
            return;

        snap->qmins.assign(dim, std::numeric_limits<float>::infinity());
        snap->qscales.assign(dim, 0.0f);
        std::vector<float> qmaxs(dim, -std::numeric_limits<float>::infinity());

        for (std::size_t d = 0; d < dim; d += 8)
        {
            __m256 vmin = _mm256_set1_ps(std::numeric_limits<float>::infinity());
            __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
            std::size_t limit = std::min(d + 8, dim);

            for (std::size_t row = 0; row < n; ++row)
            {
                const float *ptr = snap->data.data() + row * dim + d;
                __m256 v;
                if (d + 8 <= dim)
                    v = _mm256_loadu_ps(ptr);
                else
                {
                    alignas(32) float tmp[8] = {0};
                    for (std::size_t k = 0; k < limit - d; ++k)
                        tmp[k] = ptr[k];
                    v = _mm256_loadu_ps(tmp);
                }
                vmin = _mm256_min_ps(vmin, v);
                vmax = _mm256_max_ps(vmax, v);
            }

            alignas(32) float mins_out[8], maxs_out[8];
            _mm256_storeu_ps(mins_out, vmin);
            _mm256_storeu_ps(maxs_out, vmax);
            for (std::size_t k = 0; k < limit - d; ++k)
            {
                snap->qmins[d + k] = mins_out[k];
                qmaxs[d + k] = maxs_out[k];
            }
        }

        std::vector<float> inv_scales(dim);
        for (std::size_t d = 0; d < dim; ++d)
        {
            float range = qmaxs[d] - snap->qmins[d];
            snap->qscales[d] = (range > 0.0f) ? (range / 255.0f) : 1.0f;
            inv_scales[d] = 1.0f / snap->qscales[d];
        }

        snap->qdata.resize(n * dim);
        for (std::size_t row = 0; row < n; ++row)
        {
            const float *src = snap->data.data() + row * dim;
            std::uint8_t *dst = snap->qdata.data() + row * dim;
            for (std::size_t d = 0; d < dim; d += 8)
            {
                std::size_t limit = std::min(d + 8, dim);
                __m256 v = _mm256_loadu_ps(src + d);
                __m256 v_min = _mm256_loadu_ps(snap->qmins.data() + d);
                __m256 v_inv_scale = _mm256_loadu_ps(inv_scales.data() + d);
                __m256 res = _mm256_mul_ps(_mm256_sub_ps(v, v_min), v_inv_scale);
                __m256i qi = _mm256_cvtps_epi32(_mm256_round_ps(res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

                alignas(32) int32_t ints[8];
                _mm256_storeu_si256((__m256i *)ints, qi);
                for (std::size_t k = 0; k < limit - d; ++k)
                {
                    dst[d + k] = static_cast<std::uint8_t>(std::clamp(ints[k], 0, 255));
                }
            }
        }

        std::size_t extra_bytes = snap->qdata.size() * sizeof(std::uint8_t) + snap->qmins.size() * sizeof(float) + snap->qscales.size() * sizeof(float);
        snap->accounted_bytes += extra_bytes;
        MemoryManager::Instance().AddUsage(MemoryManager::Pool::Search, extra_bytes);
    }

    bool Seed::TryDetachSnapshot(Snapshot &snap, std::vector<float> &data, std::vector<Id> &ids)
    {
        if (!snap || snap.use_count() > 2)
            return false;
        if (snap->accounted_bytes > 0)
            MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Search, snap->accounted_bytes);
        snap->accounted_bytes = 0;
        ids = std::move(snap->ids);
        data = std::move(snap->data);
        return true;
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

        if (!snap->is_quantized.load(std::memory_order_acquire))
        {
            FixedTopK float_topk(req.topk);
            const float *base = snap->data.data();
            for (std::size_t row = 0; row < n; ++row)
                float_topk.Push(-kernels::L2Sqr(base + row * dim, req.query.data.data(), dim), snap->ids[row]);
            float_topk.FillSorted(resp.items);
            return resp;
        }

        constexpr std::size_t kOversample = 128;
        const std::size_t k = std::min({req.topk, kOversample, n});
        const float *q = req.query.data.data();

        alignas(32) std::uint8_t qquant[1024];
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
        const float *base = snap->data.data();
        const auto *candidates = candidate_topk.Data();
        for (std::size_t i = 0; i < candidate_topk.Size(); ++i)
        {
            std::size_t cand_row = static_cast<std::size_t>(candidates[i].id);
            final_topk.Push(-kernels::L2Sqr(base + cand_row * dim, q, dim), snap->ids[cand_row]);
        }
        final_topk.FillSorted(resp.items);
        return resp;
    }

    void Seed::UpdateMemtableAccounting()
    {
        const std::size_t bytes = ids_.size() * sizeof(Id) + data_.size() * sizeof(float);
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