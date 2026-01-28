#include "seed.h"
#include "cpu_kernels.h"
#include "memory_manager.h"

#include <algorithm>
#include <queue>
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
        out->accounted_bytes = out->ids.size() * sizeof(Id) + out->data.size() * sizeof(float);
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

        struct Node
        {
            float score; // -dist
            Id id;
        };
        // Min-heap by score (worst element on top), so we can pop/replace.
        auto cmp = [](const Node &a, const Node &b)
        { return a.score > b.score; };
        std::priority_queue<Node, std::vector<Node>, decltype(cmp)> heap(cmp);

        const float *base = snap->data.data();

        for (std::size_t row = 0; row < n; ++row)
        {
            const float *v = base + row * dim;

            // Use AVX2 L2 squared
            float dist = pomai::kernels::L2Sqr(v, q, dim);
            float score = -dist;

            if (heap.size() < k)
            {
                heap.push(Node{score, snap->ids[row]});
            }
            else if (score > heap.top().score)
            {
                heap.pop();
                heap.push(Node{score, snap->ids[row]});
            }
        }

        resp.items.reserve(heap.size());
        while (!heap.empty())
        {
            resp.items.push_back(SearchResultItem{heap.top().id, heap.top().score});
            heap.pop();
        }
        std::reverse(resp.items.begin(), resp.items.end());
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
