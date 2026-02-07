#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/shard/runtime.h" // make ShardRuntime complete here
#include "pomai/metadata.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::core
{
    class Shard
    {
    public:
        explicit Shard(std::unique_ptr<ShardRuntime> rt);
        ~Shard();

        Shard(const Shard &) = delete;
        Shard &operator=(const Shard &) = delete;

        Status Start();

        Status Put(VectorId id, std::span<const float> vec);
        Status Put(VectorId id, std::span<const float> vec, const Metadata& meta); // Overload
        Status PutBatch(const std::vector<VectorId>& ids,
                        const std::vector<std::span<const float>>& vectors);
        Status Get(VectorId id, std::vector<float> *out);
        Status Get(VectorId id, std::vector<float> *out, pomai::Metadata* out_meta); // Added
        Status Exists(VectorId id, bool *exists);
        Status Delete(VectorId id);
        Status Flush();

        Status SearchLocal(std::span<const float> q, std::uint32_t k,
                           std::vector<pomai::SearchHit> *out) const;
        Status SearchLocal(std::span<const float> q, std::uint32_t k,
                           const SearchOptions& opts, std::vector<pomai::SearchHit> *out) const;

        Status Freeze();
        Status Compact();
        Status NewIterator(std::unique_ptr<pomai::SnapshotIterator> *out);
        Status NewIterator(std::shared_ptr<ShardSnapshot> snap, std::unique_ptr<pomai::SnapshotIterator> *out);
        std::shared_ptr<ShardSnapshot> GetSnapshot();
        std::uint64_t LastQueryCandidatesScanned() const { return rt_->LastQueryCandidatesScanned(); }

    private:
        std::unique_ptr<ShardRuntime> rt_;
    };
} // namespace pomai::core
