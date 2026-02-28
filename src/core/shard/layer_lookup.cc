#include "core/shard/layer_lookup.h"

#include <utility>

#include "table/segment.h"

namespace pomai::core {

LookupResult LookupById(const std::shared_ptr<table::MemTable>& active,
                        const std::shared_ptr<ShardSnapshot>& snapshot,
                        pomai::VectorId id,
                        std::uint32_t dim) {
    if (active) {
        if (active->IsTombstone(id)) {
            return {.state = LookupState::kTombstone, .vec = {}, .decoded_vec = {}, .meta = {}, .pinnable_vec = {}};
        }

        const float* vec_ptr = nullptr;
        pomai::Metadata meta;
        const auto st = active->Get(id, &vec_ptr, &meta);
        if (st.ok() && vec_ptr != nullptr) {
            return {.state = LookupState::kFound,
                    .vec = std::span<const float>(vec_ptr, static_cast<std::size_t>(dim)),
                    .decoded_vec = {},
                    .meta = std::move(meta),
                    .pinnable_vec = {}};
        }
    }

    if (!snapshot) {
        return {};
    }

    for (auto it = snapshot->frozen_memtables.rbegin(); it != snapshot->frozen_memtables.rend(); ++it) {
        if ((*it)->IsTombstone(id)) {
            return {.state = LookupState::kTombstone, .vec = {}, .decoded_vec = {}, .meta = {}, .pinnable_vec = {}};
        }

        const float* vec_ptr = nullptr;
        pomai::Metadata meta;
        const auto st = (*it)->Get(id, &vec_ptr, &meta);
        if (st.ok() && vec_ptr != nullptr) {
            return {.state = LookupState::kFound,
                    .vec = std::span<const float>(vec_ptr, static_cast<std::size_t>(dim)),
                    .decoded_vec = {},
                    .meta = std::move(meta),
                    .pinnable_vec = {}};
        }
    }

    for (const auto& segment : snapshot->segments) {
        LookupResult res;
        const auto st = segment->Get(id, &res.pinnable_vec, &res.meta);
        if (st.ok()) {
            res.state = LookupState::kFound;
            // Map PinnableSlice data back to span<float> for existing consumers
            if (segment->GetQuantType() == pomai::QuantizationType::kNone) {
                res.vec = std::span<const float>(reinterpret_cast<const float*>(res.pinnable_vec.data()), static_cast<std::size_t>(dim));
            } else {
                // If quantized, we still need to decode for the 'vec' span if requested.
                // However, for pure zero-copy distillation, we prioritize the raw pinned data.
                size_t bytes = dim;
                if (segment->GetQuantType() == pomai::QuantizationType::kFp16) bytes *= 2;
                res.decoded_vec = segment->GetQuantizer()->Decode(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(res.pinnable_vec.data()), bytes));
                res.vec = res.decoded_vec;
            }
            return res;
        }
        if (st.code() == ErrorCode::kNotFound && std::string_view(st.message()) == "tombstone") {
            return {.state = LookupState::kTombstone, .vec = {}, .decoded_vec = {}, .meta = {}, .pinnable_vec = {}};
        }
    }

    return {.state = LookupState::kNotFound, .vec = {}, .decoded_vec = {}, .meta = {}, .pinnable_vec = {}};
}

}  // namespace pomai::core
