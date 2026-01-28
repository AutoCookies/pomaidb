#include "seed.h"
#include "cpu_kernels.h"
#include "fixed_topk.h"
#include "memory_manager.h"

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
        // Reasonable initial reserve to reduce early reallocs; tune later.
        ids_.reserve(1024);
        data_.reserve(1024 * dim_);
        pos_.reserve(1024);
    }

    Seed::Seed(const Seed &other)
        : dim_(other.dim_),
          ids_(other.ids_),
          data_(other.data_),
          pos_(other.pos_),
          accounted_bytes_(other.accounted_bytes_)
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
        : dim_(other.dim_),
          ids_(std::move(other.ids_)),
          data_(std::move(other.data_)),
          pos_(std::move(other.pos_)),
          accounted_bytes_(other.accounted_bytes_)
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

    Seed::~Seed()
    {
        ReleaseMemtableAccounting();
    }

    void Seed::ReserveForAppend(std::size_t add_rows)
    {
        const std::size_t need_rows = ids_.size() + add_rows;
        if (ids_.capacity() < need_rows)
        {
            // Growth policy: 1.5x to reduce realloc frequency.
            std::size_t new_cap = std::max<std::size_t>(need_rows, ids_.capacity() + ids_.capacity() / 2 + 1024);
            ids_.reserve(new_cap);
            data_.reserve(new_cap * dim_);
            pos_.reserve(new_cap);
        }
    }

    void Seed::ApplyUpserts(const std::vector<UpsertRequest> &batch)
    {
        if (batch.empty())
            return;

        // Count how many are new to reserve once.
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
                // Append new row
                const std::uint32_t row = static_cast<std::uint32_t>(ids_.size());
                ids_.push_back(r.id);
                pos_.emplace(r.id, row);

                const std::size_t base = static_cast<std::size_t>(row) * dim_;
                data_.resize(base + dim_);
                std::copy(r.vec.data.begin(), r.vec.data.end(), data_.begin() + base);
            }
            else
            {
                // Overwrite existing row
                const std::uint32_t row = it->second;
                const std::size_t base = static_cast<std::size_t>(row) * dim_;
                std::copy(r.vec.data.begin(), r.vec.data.end(), data_.begin() + base);
            }
        }

        UpdateMemtableAccounting();
    }

    Seed::Snapshot Seed::MakeSnapshot() const
    {
        // Immutable snapshot: copy contiguous buffers (much cheaper than copying a hashmap of vectors).
        auto out = std::shared_ptr<Store>(
            new Store(),
            [](Store *s)
            {
                if (s->accounted_bytes > 0)
                    MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Search, s->accounted_bytes);
                delete s;
            });
        out->dim = dim_;
        out->ids = ids_;
        out->data = data_;

        const std::size_t n = out->ids.size();
        out->qmins.assign(dim_, std::numeric_limits<float>::infinity());
        out->qscales.assign(dim_, 0.0f);
        std::vector<float> qmaxs(dim_, -std::numeric_limits<float>::infinity());

        if (n > 0)
        {
            for (std::size_t row = 0; row < n; ++row)
            {
                const float *src = out->data.data() + row * dim_;
                for (std::size_t d = 0; d < dim_; ++d)
                {
                    float v = src[d];
                    out->qmins[d] = std::min(out->qmins[d], v);
                    qmaxs[d] = std::max(qmaxs[d], v);
                }
            }

            for (std::size_t d = 0; d < dim_; ++d)
            {
                float range = qmaxs[d] - out->qmins[d];
                out->qscales[d] = (range > 0.0f) ? (range / 255.0f) : 1.0f;
            }

            out->qdata.resize(n * dim_);
            for (std::size_t row = 0; row < n; ++row)
            {
                const float *src = out->data.data() + row * dim_;
                std::uint8_t *dst = out->qdata.data() + row * dim_;
                for (std::size_t d = 0; d < dim_; ++d)
                {
                    float scale = out->qscales[d];
                    float q = (scale > 0.0f) ? ((src[d] - out->qmins[d]) / scale) : 0.0f;
                    int qi = static_cast<int>(std::nearbyint(q));
                    qi = std::min(255, std::max(0, qi));
                    dst[d] = static_cast<std::uint8_t>(qi);
                }
            }
        }

        out->accounted_bytes = out->ids.size() * sizeof(Id) +
                               out->data.size() * sizeof(float) +
                               out->qdata.size() * sizeof(std::uint8_t) +
                               out->qmins.size() * sizeof(float) +
                               out->qscales.size() * sizeof(float);
        if (out->accounted_bytes > 0)
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Search, out->accounted_bytes);
        return out;
    }

    bool Seed::TryDetachSnapshot(Snapshot &snap, std::vector<float> &data, std::vector<Id> &ids)
    {
        if (!snap)
            return false;
        if (!snap.unique())
            return false;

        auto *store = const_cast<Store *>(snap.get());
        if (store->accounted_bytes > 0)
        {
            MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Search, store->accounted_bytes);
            store->accounted_bytes = 0;
        }

        ids = std::move(store->ids);
        data = std::move(store->data);
        store->qdata.clear();
        store->qdata.shrink_to_fit();
        store->qmins.clear();
        store->qmins.shrink_to_fit();
        store->qscales.clear();
        store->qscales.shrink_to_fit();
        return true;
    }

    SearchResponse Seed::SearchSnapshot(const Snapshot &snap, const SearchRequest &req)
    {
        SearchResponse resp;
        if (!snap)
            return resp;
        if (snap->dim == 0)
            return resp;
        if (req.query.data.size() != snap->dim)
            return resp;

        const std::size_t dim = snap->dim;
        const std::size_t n = snap->ids.size();
        if (n == 0)
            return resp;

        const std::size_t k = std::min<std::size_t>(req.topk, n);
        if (k == 0)
            return resp;

        const float *q = req.query.data.data();

        constexpr std::size_t kOversample = 128;
        const std::size_t candidate_k = std::min<std::size_t>(kOversample, n);

        constexpr std::size_t kUnroll = 8;
        constexpr std::size_t kPrefetchDistance = 4;

        if (snap->qdata.size() != n * dim || snap->qmins.size() != dim || snap->qscales.size() != dim)
        {
            FixedTopK float_topk(k);
            const float *base = snap->data.data();
            constexpr std::size_t kBlock = 16;
            std::array<float, kBlock> distances{};
            for (std::size_t row = 0; row < n; row += kBlock)
            {
                const std::size_t count = std::min(kBlock, n - row);
                kernels::ScanBucketAVX2(base + row * dim, q, dim, count, distances.data());
                for (std::size_t i = 0; i < count; ++i)
                {
                    float score = -distances[i];
                    float_topk.Push(score, snap->ids[row + i]);
                }
            }

            float_topk.FillSorted(resp.items);
            return resp;
        }

        const std::uint8_t *qbase = snap->qdata.data();
        const float *mins = snap->qmins.data();
        const float *scales = snap->qscales.data();
        thread_local std::unique_ptr<std::uint8_t[]> qquant_buf;
        thread_local std::size_t qquant_cap = 0;
        if (qquant_cap < dim)
        {
            qquant_buf.reset(new std::uint8_t[dim]);
            qquant_cap = dim;
        }
        std::uint8_t *qquant = qquant_buf.get();
        for (std::size_t d = 0; d < dim; ++d)
        {
            float scale = scales[d];
            float qv = (scale > 0.0f) ? ((q[d] - mins[d]) / scale) : 0.0f;
            int qi = static_cast<int>(std::nearbyint(qv));
            qi = std::min(255, std::max(0, qi));
            qquant[d] = static_cast<std::uint8_t>(qi);
        }

        thread_local std::unique_ptr<FixedTopK> candidate_topk;
        if (!candidate_topk)
            candidate_topk = std::make_unique<FixedTopK>(candidate_k);
        candidate_topk->Reset(candidate_k);

        std::size_t row = 0;
        for (; row + kUnroll <= n; row += kUnroll)
        {
            std::size_t prefetch_row = row + kUnroll;
            for (std::size_t p = 0; p < kPrefetchDistance; ++p)
            {
                std::size_t next_row = prefetch_row + p;
                if (next_row < n)
                    _mm_prefetch(reinterpret_cast<const char *>(qbase + next_row * dim), _MM_HINT_T0);
            }

            float d0 = kernels::L2Sqr_SQ8_AVX2(qbase + (row + 0) * dim, qquant, dim);
            float d1 = kernels::L2Sqr_SQ8_AVX2(qbase + (row + 1) * dim, qquant, dim);
            float d2 = kernels::L2Sqr_SQ8_AVX2(qbase + (row + 2) * dim, qquant, dim);
            float d3 = kernels::L2Sqr_SQ8_AVX2(qbase + (row + 3) * dim, qquant, dim);
            float d4 = kernels::L2Sqr_SQ8_AVX2(qbase + (row + 4) * dim, qquant, dim);
            float d5 = kernels::L2Sqr_SQ8_AVX2(qbase + (row + 5) * dim, qquant, dim);
            float d6 = kernels::L2Sqr_SQ8_AVX2(qbase + (row + 6) * dim, qquant, dim);
            float d7 = kernels::L2Sqr_SQ8_AVX2(qbase + (row + 7) * dim, qquant, dim);

            candidate_topk->Push(-d0, static_cast<Id>(row + 0));
            candidate_topk->Push(-d1, static_cast<Id>(row + 1));
            candidate_topk->Push(-d2, static_cast<Id>(row + 2));
            candidate_topk->Push(-d3, static_cast<Id>(row + 3));
            candidate_topk->Push(-d4, static_cast<Id>(row + 4));
            candidate_topk->Push(-d5, static_cast<Id>(row + 5));
            candidate_topk->Push(-d6, static_cast<Id>(row + 6));
            candidate_topk->Push(-d7, static_cast<Id>(row + 7));
        }

        for (; row < n; ++row)
        {
            float d = kernels::L2Sqr_SQ8_AVX2(qbase + row * dim, qquant, dim);
            candidate_topk->Push(-d, static_cast<Id>(row));
        }

        thread_local std::unique_ptr<FixedTopK> final_topk;
        if (!final_topk)
            final_topk = std::make_unique<FixedTopK>(k);
        final_topk->Reset(k);
        const float *base = snap->data.data();
        const auto *candidates = candidate_topk->Data();
        const std::size_t candidate_count = candidate_topk->Size();
        for (std::size_t i = 0; i < candidate_count; ++i)
        {
            std::size_t cand_row = static_cast<std::size_t>(candidates[i].id);
            float d = kernels::L2Sqr(base + cand_row * dim, q, dim);
            final_topk->Push(-d, snap->ids[cand_row]);
        }
        final_topk->FillSorted(resp.items);
        return resp;
    }

    void Seed::UpdateMemtableAccounting()
    {
        const std::size_t bytes =
            ids_.size() * sizeof(Id) + data_.size() * sizeof(float);
        if (bytes == accounted_bytes_)
            return;
        if (bytes > accounted_bytes_)
        {
            MemoryManager::Instance().AddUsage(MemoryManager::Pool::Memtable, bytes - accounted_bytes_);
        }
        else
        {
            MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Memtable, accounted_bytes_ - bytes);
        }
        accounted_bytes_ = bytes;
    }

    void Seed::ReleaseMemtableAccounting()
    {
        if (accounted_bytes_ == 0)
            return;
        MemoryManager::Instance().ReleaseUsage(MemoryManager::Pool::Memtable, accounted_bytes_);
        accounted_bytes_ = 0;
    }

} // namespace pomai
