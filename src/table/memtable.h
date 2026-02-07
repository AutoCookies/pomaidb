#pragma once
#include <cstdint>
#include <span>
#include <unordered_map>
#include <shared_mutex>
#include "pomai/metadata.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "table/arena.h"

namespace pomai::table
{

    class MemTable
    {
    public:
        MemTable(std::uint32_t dim, std::size_t arena_block_bytes);

        pomai::Status Put(pomai::VectorId id, pomai::VectorView vec);
        // Overload with metadata
        pomai::Status Put(pomai::VectorId id, pomai::VectorView vec, const pomai::Metadata& meta);
        
        // Batch put: Optimized for inserting multiple vectors at once
        pomai::Status PutBatch(const std::vector<pomai::VectorId>& ids,
                               const std::vector<pomai::VectorView>& vectors);
        
        pomai::Status Get(pomai::VectorId id, const float** out_vec) const;
        // Get metadata for an ID (returns empty/default if not found or no metadata)
        pomai::Status Get(pomai::VectorId id, const float** out_vec, pomai::Metadata* out_meta) const;
        
        pomai::Status Delete(pomai::VectorId id);

        size_t GetCount() const {
            std::shared_lock lock(mutex_);
            return map_.size();
        }
        void Clear();

        struct CursorEntry {
            pomai::VectorId id;
            std::span<const float> vec;
            bool is_deleted;
            const pomai::Metadata* meta;
        };

        class Cursor {
        public:
            bool Next(CursorEntry* out);

        private:
            friend class MemTable;
            using MapIter = std::unordered_map<pomai::VectorId, float *>::const_iterator;
            Cursor(const MemTable* mem, MapIter it, MapIter end) : mem_(mem), it_(it), end_(end) {}

            const MemTable* mem_;
            MapIter it_;
            MapIter end_;
        };

        Cursor CreateCursor() const;

        const float *GetPtr(pomai::VectorId id) const
        {
            std::shared_lock lock(mutex_);
            auto it = map_.find(id);
            if (it == map_.end())
                return nullptr;
            return it->second;
        }

        bool IsTombstone(pomai::VectorId id) const
        {
            std::shared_lock lock(mutex_);
            auto it = map_.find(id);
            return (it != map_.end()) && (it->second == nullptr);
        }


        template <class Fn>
        void IterateWithStatus(Fn &&fn) const
        {
            std::shared_lock lock(mutex_);
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
        void IterateWithMetadata(Fn &&fn) const
        {
            std::shared_lock lock(mutex_);
            for (const auto &[id, ptr] : map_)
            {
                bool is_deleted = (ptr == nullptr);
                std::span<const float> vec;
                if (!is_deleted) {
                    vec = std::span<const float>{ptr, dim_};
                }
                
                const pomai::Metadata* meta_ptr = nullptr;
                if (!is_deleted) {
                     auto it = metadata_.find(id);
                     if (it != metadata_.end()) {
                         meta_ptr = &it->second;
                     }
                }
                fn(id, vec, is_deleted, meta_ptr);
            }
        }

        template <class Fn>
        void ForEach(Fn &&fn) const
        {
            std::shared_lock lock(mutex_);
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
        std::unordered_map<pomai::VectorId, pomai::Metadata> metadata_;
        mutable std::shared_mutex mutex_;
    };

} // namespace pomai::table
