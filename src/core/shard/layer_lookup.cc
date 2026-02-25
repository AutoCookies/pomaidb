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
            return {.state = LookupState::kTombstone};
        }

        const float* vec_ptr = nullptr;
        pomai::Metadata meta;
        const auto st = active->Get(id, &vec_ptr, &meta);
        if (st.ok() && vec_ptr != nullptr) {
            return {.state = LookupState::kFound,
                    .vec = std::span<const float>(vec_ptr, dim),
                    .meta = std::move(meta)};
        }
    }

    if (!snapshot) {
        return {};
    }

    for (auto it = snapshot->frozen_memtables.rbegin(); it != snapshot->frozen_memtables.rend(); ++it) {
        if ((*it)->IsTombstone(id)) {
            return {.state = LookupState::kTombstone};
        }

        const float* vec_ptr = nullptr;
        pomai::Metadata meta;
        const auto st = (*it)->Get(id, &vec_ptr, &meta);
        if (st.ok() && vec_ptr != nullptr) {
            return {.state = LookupState::kFound,
                    .vec = std::span<const float>(vec_ptr, dim),
                    .meta = std::move(meta)};
        }
    }

    for (const auto& segment : snapshot->segments) {
        std::span<const float> seg_vec;
        std::vector<float> seg_decoded;
        pomai::Metadata seg_meta;
        const auto find = segment->FindAndDecode(id, &seg_vec, &seg_decoded, &seg_meta);
        if (find == table::SegmentReader::FindResult::kFoundTombstone) {
            return {.state = LookupState::kTombstone};
        }
        if (find == table::SegmentReader::FindResult::kFound) {
            return {.state = LookupState::kFound, .vec = seg_vec, .decoded_vec = std::move(seg_decoded), .meta = std::move(seg_meta)};
        }
    }

    return {};
}

}  // namespace pomai::core
