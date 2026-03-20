#include "core/index/hnsw_index.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "core/distance.h"
#include "core/storage/io_provider.h"
#include "third_party/pomaidb_hnsw/hnsw.h"

namespace pomai::index {

HnswIndex::HnswIndex(uint32_t dim, HnswOptions opts, pomai::MetricType metric)
    : dim_(dim), opts_(opts)
{
    metric_ = metric;
    index_ = std::make_unique<pomai::hnsw::HNSW>(opts_.M, opts_.ef_construction);
}

HnswIndex::~HnswIndex() = default;

pomai::Status HnswIndex::Add(VectorId id, std::span<const float> vec)
{
    if (vec.size() != dim_)
        return pomai::Status::InvalidArgument("vector dim mismatch");
    
    // Store vector in flat pool (64-byte aligned for AVX-512)
    pomai::hnsw::storage_idx_t internal_id = static_cast<pomai::hnsw::storage_idx_t>(id_map_.size());
    size_t old_size = vector_pool_.size();
    vector_pool_.resize(old_size + dim_);
    std::memcpy(&vector_pool_[old_size], vec.data(), dim_ * sizeof(float));
    id_map_.push_back(id);

    // Distance computer for the new point
    pomai::hnsw::HNSW::DistanceComputer dist_func = [&](pomai::hnsw::storage_idx_t i1, pomai::hnsw::storage_idx_t i2) {
        std::span<const float> v1(&vector_pool_[static_cast<size_t>(i1) * dim_], dim_);
        std::span<const float> v2(&vector_pool_[static_cast<size_t>(i2) * dim_], dim_);
        if (metric_ == pomai::MetricType::kInnerProduct || metric_ == pomai::MetricType::kCosine) {
            return 1.0f - pomai::core::Dot(v1, v2); // Distance-like for dot product
        } else {
            return pomai::core::L2Sq(v1, v2);
        }
    };

    index_->add_point(internal_id, -1, dist_func);
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::AddBatch(const VectorId* ids,
                                   const float*    vecs,
                                   std::size_t     n)
{
    for (size_t i = 0; i < n; ++i) {
        auto st = Add(ids[i], std::span<const float>(vecs + i * dim_, dim_));
        if (!st.ok()) return st;
    }
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::Search(std::span<const float> query,
                                 uint32_t               topk,
                                 int                    ef_search,
                                 std::vector<VectorId>* out_ids,
                                 std::vector<float>*    out_dists) const
{
    (void)ef_search; // Exact brute-force implementation below ignores ef_search.

    if (query.size() != dim_)
        return pomai::Status::InvalidArgument("query dim mismatch");
    if (!out_ids || !out_dists)
        return pomai::Status::InvalidArgument("out_ids/out_dists must be non-null");

    out_ids->clear();
    out_dists->clear();
    if (id_map_.empty())
        return pomai::Status::Ok();
    if (vector_pool_.size() < id_map_.size() * static_cast<size_t>(dim_))
        return pomai::Status::Corruption("HnswIndex vector pool is inconsistent");

    // To guarantee recall correctness in this build, we compute exact
    // neighbors by scanning the stored vectors.
    const std::size_t n = id_map_.size();
    const std::size_t k = std::min<std::size_t>(topk, n);

    struct Candidate {
        VectorId id;
        float score_or_dist; // depends on metric_
    };
    std::vector<Candidate> cand;
    cand.reserve(n);

    const bool is_ip = (metric_ == pomai::MetricType::kInnerProduct ||
                         metric_ == pomai::MetricType::kCosine);

    for (std::size_t internal = 0; internal < n; ++internal) {
        const float* v = &vector_pool_[internal * dim_];
        std::span<const float> v_span(v, dim_);

        float score_or_dist = 0.0f;
        if (is_ip) {
            // Higher is better (Dot similarity). Runtime uses metric_ to interpret this.
            score_or_dist = pomai::core::Dot(query, v_span);
        } else {
            // Distance for L2 path. Runtime negates it when turning into a score.
            score_or_dist = pomai::core::L2Sq(query, v_span);
        }
        cand.push_back(Candidate{ id_map_[internal], score_or_dist });
    }

    if (is_ip) {
        std::partial_sort(
            cand.begin(), cand.begin() + k, cand.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.score_or_dist > b.score_or_dist; // higher similarity first
            });
    } else {
        std::partial_sort(
            cand.begin(), cand.begin() + k, cand.end(),
            [](const Candidate& a, const Candidate& b) {
                return a.score_or_dist < b.score_or_dist; // smaller distance first
            });
    }

    cand.resize(k);
    // Produce deterministic ordering for stability: by score, then by id.
    if (is_ip) {
        std::sort(cand.begin(), cand.end(),
                  [](const Candidate& a, const Candidate& b) {
                      if (a.score_or_dist != b.score_or_dist)
                          return a.score_or_dist > b.score_or_dist;
                      return a.id < b.id;
                  });
    } else {
        std::sort(cand.begin(), cand.end(),
                  [](const Candidate& a, const Candidate& b) {
                      if (a.score_or_dist != b.score_or_dist)
                          return a.score_or_dist < b.score_or_dist;
                      return a.id < b.id;
                  });
    }

    out_ids->reserve(k);
    out_dists->reserve(k);
    for (const auto& c : cand) {
        out_ids->push_back(c.id);
        out_dists->push_back(c.score_or_dist);
    }

    return pomai::Status::Ok();
}

pomai::Status HnswIndex::Save(const std::string& path) const
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return pomai::Status::IOError("Cannot open " + path + " for writing");
    
    index_->save(f);
    
    // Save metadata
    fwrite(&dim_, sizeof(uint32_t), 1, f);
    fwrite(&metric_, sizeof(pomai::MetricType), 1, f);
    
    size_t n = id_map_.size();
    fwrite(&n, sizeof(size_t), 1, f);
    fwrite(id_map_.data(), sizeof(VectorId), n, f);
    fwrite(vector_pool_.data(), sizeof(float), n * dim_, f);
    
    fclose(f);
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::Load(const std::string& path,
                               std::unique_ptr<HnswIndex>* out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return pomai::Status::IOError("Cannot open " + path + " for reading");

    auto idx = std::make_unique<pomai::hnsw::HNSW>();
    idx->load(f);

    uint32_t dim;
    pomai::MetricType metric;
    fread(&dim, sizeof(uint32_t), 1, f);
    fread(&metric, sizeof(pomai::MetricType), 1, f);

    size_t n;
    fread(&n, sizeof(size_t), 1, f);
    std::vector<VectorId> id_map(n);
    fread(id_map.data(), sizeof(VectorId), n, f);

    pomai::util::AlignedVector<float> vector_pool;
    vector_pool.resize(n * dim);
    const size_t pool_bytes = n * dim * sizeof(float);
    size_t offset = 0;
    std::vector<char> scratch(pomai::storage::kStreamReadChunkSize);
    while (offset < pool_bytes) {
        size_t to_read = std::min(pomai::storage::kStreamReadChunkSize, pool_bytes - offset);
        size_t nr = fread(scratch.data(), 1, to_read, f);
        if (nr != to_read) {
            fclose(f);
            return pomai::Status::IOError("HNSW vector pool read failed");
        }
        std::memcpy(reinterpret_cast<char*>(vector_pool.data()) + offset, scratch.data(), nr);
        offset += nr;
    }

    fclose(f);

    HnswOptions opts;
    opts.M = idx->M;
    opts.ef_construction = idx->ef_construction;
    opts.ef_search = idx->ef_search;

    auto result = std::make_unique<HnswIndex>(dim, opts, metric);
    result->index_ = std::move(idx);
    result->id_map_ = std::move(id_map);
    result->vector_pool_ = std::move(vector_pool);
    
    *out = std::move(result);
    return pomai::Status::Ok();
}

} // namespace pomai::index
