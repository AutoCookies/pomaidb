#include "core/shard/shard.h"

#include <utility>

namespace pomai::core
{
    Shard::Shard(std::unique_ptr<ShardRuntime> rt) : rt_(std::move(rt)) {}
    Shard::~Shard() = default;

    pomai::Status Shard::Start() { return rt_->Start(); }

    pomai::Status Shard::Put(pomai::VectorId id, std::span<const float> vec)
    {
        return rt_->Put(id, vec);
    }

    pomai::Status Shard::Delete(pomai::VectorId id)
    {
        return rt_->Delete(id);
    }

    pomai::Status Shard::Flush()
    {
        return rt_->Flush();
    }

    pomai::Status Shard::SearchLocal(std::span<const float> query,
                                     std::uint32_t topk,
                                     std::vector<pomai::SearchHit> *out) const
    {
        return rt_->Search(query, topk, out);
    }

} // namespace pomai::core
