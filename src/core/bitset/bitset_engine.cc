#include "core/bitset/bitset_engine.h"

#include "core/simd/simd_dispatch.h"

namespace pomai::core {

Status BitsetEngine::Put(std::uint64_t id, std::span<const std::uint8_t> bits) {
    std::size_t old = 0;
    auto it = bitsets_.find(id);
    if (it != bitsets_.end()) old = it->second.size();
    if (max_bytes_ > 0 && cur_bytes_ - old + bits.size() > max_bytes_) return Status::ResourceExhausted("bitset cap exceeded");
    bitsets_[id] = std::vector<std::uint8_t>(bits.begin(), bits.end());
    cur_bytes_ = cur_bytes_ - old + bits.size();
    return Status::Ok();
}

Status BitsetEngine::BinaryOp(std::uint64_t a, std::uint64_t b, std::vector<std::uint8_t>* out, std::uint8_t op) const {
    if (!out) return Status::InvalidArgument("bitset out null");
    auto ia = bitsets_.find(a);
    auto ib = bitsets_.find(b);
    if (ia == bitsets_.end() || ib == bitsets_.end()) return Status::NotFound("bitset not found");
    if (ia->second.size() != ib->second.size()) return Status::InvalidArgument("bitset size mismatch");
    out->resize(ia->second.size());
    for (std::size_t i = 0; i < ia->second.size(); ++i) {
        if (op == 0) (*out)[i] = ia->second[i] & ib->second[i];
        else if (op == 1) (*out)[i] = ia->second[i] | ib->second[i];
        else (*out)[i] = ia->second[i] ^ ib->second[i];
    }
    return Status::Ok();
}

Status BitsetEngine::And(std::uint64_t a, std::uint64_t b, std::vector<std::uint8_t>* out) const { return BinaryOp(a, b, out, 0); }
Status BitsetEngine::Or(std::uint64_t a, std::uint64_t b, std::vector<std::uint8_t>* out) const { return BinaryOp(a, b, out, 1); }
Status BitsetEngine::Xor(std::uint64_t a, std::uint64_t b, std::vector<std::uint8_t>* out) const { return BinaryOp(a, b, out, 2); }

Status BitsetEngine::Hamming(std::uint64_t a, std::uint64_t b, double* out) const {
    if (!out) return Status::InvalidArgument("bitset out null");
    auto ia = bitsets_.find(a);
    auto ib = bitsets_.find(b);
    if (ia == bitsets_.end() || ib == bitsets_.end()) return Status::NotFound("bitset not found");
    if (ia->second.size() != ib->second.size()) return Status::InvalidArgument("bitset size mismatch");
    *out = simd::BitsetHamming(ia->second.data(), ib->second.data(), ia->second.size());
    return Status::Ok();
}

Status BitsetEngine::Jaccard(std::uint64_t a, std::uint64_t b, double* out) const {
    if (!out) return Status::InvalidArgument("bitset out null");
    auto ia = bitsets_.find(a);
    auto ib = bitsets_.find(b);
    if (ia == bitsets_.end() || ib == bitsets_.end()) return Status::NotFound("bitset not found");
    if (ia->second.size() != ib->second.size()) return Status::InvalidArgument("bitset size mismatch");
    *out = simd::BitsetJaccard(ia->second.data(), ib->second.data(), ia->second.size());
    return Status::Ok();
}

void BitsetEngine::ForEach(const std::function<void(std::uint64_t id, std::size_t nbytes)>& fn) const {
    for (const auto& [id, b] : bitsets_) fn(id, b.size());
}

} // namespace pomai::core

