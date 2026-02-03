#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/shard/runtime.h" // make ShardRuntime complete here
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
        Status Get(VectorId id, std::vector<float> *out);
        Status Exists(VectorId id, bool *exists);
        Status Delete(VectorId id);
        Status Flush();

        Status SearchLocal(std::span<const float> q, std::uint32_t k,
                           std::vector<pomai::SearchHit> *out) const;

        Status Freeze();
        Status Compact();

    private:
        std::unique_ptr<ShardRuntime> rt_;
    };
} // namespace pomai::core
