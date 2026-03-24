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
    if (query.size() != dim_)
        return pomai::Status::InvalidArgument("query dim mismatch");
    if (!out_ids || !out_dists)
        return pomai::Status::InvalidArgument("out_ids/out_dists must be non-null");

    out_ids->clear();
    out_dists->clear();
    if (topk == 0)
        return pomai::Status::Ok();
    if (id_map_.empty())
        return pomai::Status::Ok();
    if (vector_pool_.size() < id_map_.size() * static_cast<size_t>(dim_))
        return pomai::Status::Corruption("HnswIndex vector pool is inconsistent");

    const std::size_t n = id_map_.size();
    const std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(topk), n);

    const bool is_ip = (metric_ == pomai::MetricType::kInnerProduct ||
                        metric_ == pomai::MetricType::kCosine);

    // Graph ANN search (same distance semantics as Add()).
    pomai::hnsw::HNSW::QueryDistanceComputer qdis = [&](pomai::hnsw::storage_idx_t internal_id) -> float {
        const float* v = &vector_pool_[static_cast<size_t>(internal_id) * dim_];
        std::span<const float> v_span(v, dim_);
        if (is_ip) {
            return 1.0f - pomai::core::Dot(query, v_span);
        }
        return pomai::core::L2Sq(query, v_span);
    };

    const int ef_cfg = ef_search > 0 ? ef_search : opts_.ef_search;
    const int ef_use = std::max(ef_cfg, static_cast<int>(k));

    std::vector<pomai::hnsw::storage_idx_t> internal_ids;
    std::vector<float> internal_dists;
    index_->search(qdis, static_cast<int>(k), ef_use, internal_ids, internal_dists);

    if (internal_ids.empty()) {
        // Extremely small graphs / edge cases: fall back to exact top-k.
        struct Candidate {
            VectorId id;
            float score_or_dist;
        };
        std::vector<Candidate> cand;
        cand.reserve(n);
        for (std::size_t internal = 0; internal < n; ++internal) {
            const float* v = &vector_pool_[internal * dim_];
            std::span<const float> v_span(v, dim_);
            float score_or_dist = is_ip ? pomai::core::Dot(query, v_span)
                                        : pomai::core::L2Sq(query, v_span);
            cand.push_back(Candidate{ id_map_[internal], score_or_dist });
        }
        if (is_ip) {
            std::partial_sort(
                cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(k), cand.end(),
                [](const Candidate& a, const Candidate& b) {
                    return a.score_or_dist > b.score_or_dist;
                });
        } else {
            std::partial_sort(
                cand.begin(), cand.begin() + static_cast<std::ptrdiff_t>(k), cand.end(),
                [](const Candidate& a, const Candidate& b) {
                    return a.score_or_dist < b.score_or_dist;
                });
        }
        cand.resize(k);
        out_ids->reserve(k);
        out_dists->reserve(k);
        for (const auto& c : cand) {
            out_ids->push_back(c.id);
            out_dists->push_back(c.score_or_dist);
        }
        return pomai::Status::Ok();
    }

    out_ids->reserve(internal_ids.size());
    out_dists->reserve(internal_dists.size());
    for (size_t i = 0; i < internal_ids.size(); ++i) {
        const auto ii = internal_ids[i];
        if (ii < 0 || static_cast<size_t>(ii) >= id_map_.size()) {
            continue;
        }
        out_ids->push_back(id_map_[static_cast<size_t>(ii)]);
        // Runtime expects dot similarity for IP/Cosine, and L2^2 for L2 (negated to score).
        if (is_ip) {
            out_dists->push_back(1.0f - internal_dists[i]);
        } else {
            out_dists->push_back(internal_dists[i]);
        }
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
