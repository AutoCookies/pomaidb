#include "core/index/hnsw_index.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include "third_party/pomaidb_hnsw/hnsw.h"
#include "core/distance.h"

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
    
    // Store vector in flat pool
    pomai::hnsw::storage_idx_t internal_id = static_cast<pomai::hnsw::storage_idx_t>(id_map_.size());
    vector_pool_.insert(vector_pool_.end(), vec.begin(), vec.end());
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
    if (id_map_.empty())
        return pomai::Status::Ok();

    int ef = (ef_search > 0) ? ef_search : opts_.ef_search;

    pomai::hnsw::HNSW::QueryDistanceComputer qdis = [&](pomai::hnsw::storage_idx_t target_id) {
        std::span<const float> v_target(&vector_pool_[static_cast<size_t>(target_id) * dim_], dim_);
        if (metric_ == pomai::MetricType::kInnerProduct || metric_ == pomai::MetricType::kCosine) {
            return 1.0f - pomai::core::Dot(query, v_target);
        } else {
            return pomai::core::L2Sq(query, v_target);
        }
    };

    std::vector<pomai::hnsw::storage_idx_t> internal_ids;
    std::vector<float> distances;
    index_->search(qdis, static_cast<int>(topk), ef, internal_ids, distances);

    out_ids->clear();
    out_dists->clear();
    for (size_t i = 0; i < internal_ids.size(); ++i) {
        out_ids->push_back(id_map_[static_cast<size_t>(internal_ids[i])]);
        out_dists->push_back(distances[i]);
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
    
    std::vector<float> vector_pool(n * dim);
    fread(vector_pool.data(), sizeof(float), n * dim, f);
    
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
