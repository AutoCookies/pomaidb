#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "core/shard/shard.h"
#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::core
{

    class Engine
    {
    public:
        explicit Engine(pomai::DBOptions opt);

        pomai::Status Open();
        pomai::Status Close();

        pomai::Status Put(VectorId id, std::span<const float> vec);
        pomai::Status Delete(VectorId id);
        pomai::Status Flush();

        pomai::Status Search(std::span<const float> query,
                             std::uint32_t topk,
                             std::vector<pomai::SearchHit> *out);

    private:
        std::uint32_t ShardOf(VectorId id) const noexcept;

        pomai::DBOptions opt_;
        std::vector<std::unique_ptr<Shard>> shards_;
    };

} // namespace pomai::core
