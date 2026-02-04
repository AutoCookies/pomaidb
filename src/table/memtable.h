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
        
        // Batch put: Optimized for inserting multiple vectors at once
        pomai::Status PutBatch(const std::vector<pomai::VectorId>& ids,
                               const std::vector<std::span<const float>>& vectors);
        
        pomai::Status Get(pomai::VectorId id, const float** out_vec) const;
        pomai::Status Delete(pomai::VectorId id);

        size_t GetCount() const { return map_.size(); }
        void Clear();

        const float *GetPtr(pomai::VectorId id) const
        {
            auto it = map_.find(id);
            if (it == map_.end())
                return nullptr;
            return it->second;
        }

        bool IsTombstone(pomai::VectorId id) const
        {
            auto it = map_.find(id);
            return (it != map_.end()) && (it->second == nullptr);
        }


        template <class Fn>
        void IterateWithStatus(Fn &&fn) const
        {
            for (const auto &[id, ptr] : map_)
            {
                bool is_deleted = (ptr == nullptr);
                // If deleted, vec is empty span.
                std::span<const float> vec;
                if (!is_deleted) {
                    vec = std::span<const float>{ptr, dim_};
                }
                fn(id, vec, is_deleted);
            }
        }

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
