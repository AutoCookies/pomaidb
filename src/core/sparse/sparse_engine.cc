#include "core/sparse/sparse_engine.h"

#include "core/simd/simd_dispatch.h"

namespace pomai::core {

Status SparseEngine::Put(std::uint64_t id, const pomai::SparseEntry& entry) {
    if (entry.indices.size() != entry.weights.size()) return Status::InvalidArgument("sparse index/weight size mismatch");
    if (max_entries_ > 0 && sparse_.size() >= max_entries_ && sparse_.find(id) == sparse_.end()) sparse_.erase(sparse_.begin());
    sparse_[id] = entry;
    return Status::Ok();
}

Status SparseEngine::Dot(std::uint64_t a, std::uint64_t b, double* out) const {
    if (!out) return Status::InvalidArgument("sparse out null");
    auto ia = sparse_.find(a);
    auto ib = sparse_.find(b);
    if (ia == sparse_.end() || ib == sparse_.end()) return Status::NotFound("sparse vector not found");
    *out = simd::SparseDotU32F32(ia->second.indices.data(), ia->second.weights.data(), ia->second.indices.size(),
                                 ib->second.indices.data(), ib->second.weights.data(), ib->second.indices.size());
    return Status::Ok();
}

Status SparseEngine::Intersect(std::uint64_t a, std::uint64_t b, std::uint32_t* out) const {
    if (!out) return Status::InvalidArgument("sparse out null");
    auto ia = sparse_.find(a);
    auto ib = sparse_.find(b);
    if (ia == sparse_.end() || ib == sparse_.end()) return Status::NotFound("sparse vector not found");
    *out = simd::SparseIntersectU32(ia->second.indices.data(), ia->second.indices.size(),
                                    ib->second.indices.data(), ib->second.indices.size());
    return Status::Ok();
}

Status SparseEngine::Jaccard(std::uint64_t a, std::uint64_t b, double* out) const {
    if (!out) return Status::InvalidArgument("sparse out null");
    std::uint32_t inter = 0;
    auto st = Intersect(a, b, &inter);
    if (!st.ok()) return st;
    const auto& A = sparse_.at(a).indices;
    const auto& B = sparse_.at(b).indices;
    const std::uint32_t uni = static_cast<std::uint32_t>(A.size() + B.size() - inter);
    *out = (uni == 0) ? 1.0 : static_cast<double>(inter) / static_cast<double>(uni);
    return Status::Ok();
}

void SparseEngine::ForEach(const std::function<void(std::uint64_t id, std::size_t nnz)>& fn) const {
    for (const auto& [id, e] : sparse_) fn(id, e.indices.size());
}

} // namespace pomai::core

