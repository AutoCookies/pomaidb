#pragma once
#include <cstdint>
#include <span>
#include <unordered_map>
#include "pomai/status.h"
#include "pomai/types.h"
#include "table/arena.h"

namespace pomai::table
{

    class MemTable
    {
    public:
        MemTable(std::uint32_t dim, std::size_t arena_block_bytes);

        pomai::Status Put(pomai::VectorId id, std::span<const float> vec);
        pomai::Status Delete(pomai::VectorId id);

        template <class Fn>
        void ForEach(Fn &&fn) const
        {
            for (const auto &[id, ptr] : map_)
            {
                if (ptr == nullptr)
                    continue;
                fn(id, std::span<const float>{ptr, dim_});
            }
        }

    private:
        std::uint32_t dim_;
        Arena arena_;
        std::unordered_map<pomai::VectorId, float *> map_;
    };

} // namespace pomai::table
