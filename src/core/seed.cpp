#include "seed.h"
#include "cpu_kernels.h" // Sửa include path cho phù hợp với CMake (src/core/)

#include <cmath>
#include <queue>
#include <algorithm>
#include <mutex>
#include <cstring>

namespace pomai
{

    Seed::Seed(std::size_t dim) : dim_(dim)
    {
        ids_.reserve(4096);
        id_to_loc_.reserve(4096);
        chunks_.push_back(std::make_shared<Chunk>());
        chunks_.back()->reserve(kVectorsPerChunk * dim_);
    }

    void Seed::ApplyUpserts(const std::vector<UpsertRequest> &batch)
    {
        std::unique_lock<std::shared_mutex> lock(mu_);

        for (const auto &req : batch)
        {
            if (req.vec.data.size() != dim_)
                continue;

            auto it = id_to_loc_.find(req.id);
            if (it != id_to_loc_.end())
            {
                std::uint32_t c_idx = it->second.first;
                std::uint32_t offset = it->second.second;

                Chunk &chunk = *chunks_[c_idx];
                std::memcpy(chunk.data() + offset, req.vec.data.data(), dim_ * sizeof(float));
            }
            else
            {
                Chunk *current_chunk = chunks_.back().get();

                if (current_chunk->size() >= kVectorsPerChunk * dim_)
                {
                    chunks_.push_back(std::make_shared<Chunk>());
                    current_chunk = chunks_.back().get();
                    current_chunk->reserve(kVectorsPerChunk * dim_);
                }

                std::size_t offset = current_chunk->size();
                current_chunk->insert(current_chunk->end(), req.vec.data.begin(), req.vec.data.end());

                std::uint32_t c_idx = static_cast<std::uint32_t>(chunks_.size() - 1);

                ids_.push_back(req.id);
                id_to_loc_[req.id] = {c_idx, static_cast<std::uint32_t>(offset)};
            }
        }
    }

    std::shared_ptr<const SeedSnapshot> Seed::MakeSnapshot() const
    {
        std::shared_lock<std::shared_mutex> lock(mu_);

        auto snap = std::make_shared<SeedSnapshot>();
        snap->dim = dim_;
        snap->ids = ids_;
        snap->chunks = chunks_;

        return snap;
    }

    std::size_t Seed::Count() const
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        return ids_.size();
    }

    std::vector<float> Seed::GetFlatData() const
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        std::vector<float> flat;
        flat.reserve(ids_.size() * dim_);
        for (const auto &chunk : chunks_)
        {
            flat.insert(flat.end(), chunk->begin(), chunk->end());
        }
        return flat;
    }

    std::vector<Id> Seed::GetFlatIds() const
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        return ids_;
    }

    SearchResponse Seed::SearchSnapshot(const std::shared_ptr<const SeedSnapshot> &snap, const SearchRequest &req)
    {
        SearchResponse resp;
        if (!snap || snap->ids.empty() || req.query.data.empty())
            return resp;

        const std::size_t dim = snap->dim;
        const float *query_ptr = req.query.data.data();

        using Pair = std::pair<float, Id>;
        auto cmp = [](const Pair &a, const Pair &b)
        { return a.first > b.first; };
        std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> min_heap(cmp);

        std::size_t global_idx = 0;
        const std::size_t total_ids = snap->ids.size();

        for (const auto &chunk_ptr : snap->chunks)
        {
            const std::vector<float> &chunk = *chunk_ptr;
            const float *chunk_data = chunk.data();

            // FIX: Sử dụng '.' thay vì '->' vì chunk là reference
            const std::size_t num_vecs_in_chunk = chunk.size() / dim;

            for (std::size_t k = 0; k < num_vecs_in_chunk; ++k)
            {
                if (global_idx >= total_ids)
                    break;

                const float *vec_ptr = chunk_data + (k * dim);
                float dist_sq = kernels::L2Sqr(vec_ptr, query_ptr, dim);
                float score = -dist_sq;

                if (min_heap.size() < req.topk)
                {
                    min_heap.push({score, snap->ids[global_idx]});
                }
                else if (score > min_heap.top().first)
                {
                    min_heap.pop();
                    min_heap.push({score, snap->ids[global_idx]});
                }

                global_idx++;
            }
        }

        resp.items.resize(min_heap.size());
        for (int i = static_cast<int>(min_heap.size()) - 1; i >= 0; --i)
        {
            resp.items[i] = {min_heap.top().second, min_heap.top().first};
            min_heap.pop();
        }

        return resp;
    }

}