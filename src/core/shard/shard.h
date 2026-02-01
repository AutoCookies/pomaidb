#pragma once

#include <memory>
#include <span>
#include <vector>

#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::core
{

    class ShardRuntime;

    class Shard
    {
    public:
        explicit Shard(std::unique_ptr<ShardRuntime> rt);
        ~Shard();

        pomai::Status Start();

        pomai::Status Put(VectorId id, std::span<const float> vec);
        pomai::Status Delete(VectorId id);
        pomai::Status Flush();

        pomai::Status Search(std::span<const float> query,
                             std::uint32_t topk,
                             std::vector<pomai::SearchHit> *out);

    private:
        std::unique_ptr<ShardRuntime> rt_;
    };

} // namespace pomai::core
