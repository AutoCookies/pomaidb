#include "core/shard/shard.h"

#include <utility>
#include "pomai/iterator.h"  // For SnapshotIterator

namespace pomai::core
{
    Shard::Shard(std::unique_ptr<ShardRuntime> rt) : rt_(std::move(rt)) {}
    Shard::~Shard() = default;

    pomai::Status Shard::Start() { return rt_->Start(); }

    pomai::Status Shard::Put(pomai::VectorId id, std::span<const float> vec)
    {
        return rt_->Put(id, vec);
    }

    pomai::Status Shard::Put(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata& meta)
    {
        return rt_->Put(id, vec, meta);
    }

    pomai::Status Shard::PutBatch(const std::vector<pomai::VectorId>& ids,
                                  const std::vector<std::span<const float>>& vectors)
    {
        return rt_->PutBatch(ids, vectors);
    }

    pomai::Status Shard::Get(pomai::VectorId id, std::vector<float> *out)
    {
        return rt_->Get(id, out, nullptr);
    }

    pomai::Status Shard::Get(pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta)
    {
        return rt_->Get(id, out, out_meta);
    }

    pomai::Status Shard::Exists(pomai::VectorId id, bool *exists)
    {
        return rt_->Exists(id, exists);
    }

    pomai::Status Shard::Delete(pomai::VectorId id)
    {
        return rt_->Delete(id);
    }

    pomai::Status Shard::GetSemanticPointer(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id, pomai::SemanticPointer* out)
    {
        return rt_->GetSemanticPointer(std::move(snap), id, out);
    }

    pomai::Status Shard::Flush()
    {
        return rt_->Flush();
    }

    pomai::Status Shard::SearchLocal(std::span<const float> q, std::uint32_t k,
                              std::vector<pomai::SearchHit> *out) const
    {
        return rt_->Search(q, k, SearchOptions{}, out);
    }

    pomai::Status Shard::SearchLocal(std::span<const float> q, std::uint32_t k,
                              const SearchOptions& opts, std::vector<pomai::SearchHit> *out) const
    {
        return rt_->Search(q, k, opts, out);
    }

    Status Shard::Freeze() { return rt_->Freeze(); }
    Status Shard::Compact() { return rt_->Compact(); }
    Status Shard::NewIterator(std::unique_ptr<pomai::SnapshotIterator> *out) { return rt_->NewIterator(out); }
    Status Shard::NewIterator(std::shared_ptr<ShardSnapshot> snap, std::unique_ptr<pomai::SnapshotIterator> *out) {
        return rt_->NewIterator(std::move(snap), out); 
    }

    std::shared_ptr<ShardSnapshot> Shard::GetSnapshot() {
        return rt_->GetSnapshot();
    }

} // namespace pomai::core
