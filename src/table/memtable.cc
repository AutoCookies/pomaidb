#include "table/memtable.h"
#include <cstring>
#include <mutex>

namespace pomai::table
{

    static std::size_t AlignUp(std::size_t x, std::size_t a)
    {
        return (x + (a - 1)) & ~(a - 1);
    }

    void *Arena::Allocate(std::size_t n, std::size_t align)
    {
        if (blocks_.empty() || AlignUp(blocks_.back().used, align) + n > block_bytes_)
        {
            Block b;
            b.mem = std::make_unique<std::byte[]>(block_bytes_);
            b.used = 0;
            blocks_.push_back(std::move(b));
        }
        auto &blk = blocks_.back();
        blk.used = AlignUp(blk.used, align);
        void *p = blk.mem.get() + blk.used;
        blk.used += n;
        return p;
    }

    MemTable::MemTable(std::uint32_t dim, std::size_t arena_block_bytes)
        : dim_(dim), arena_(arena_block_bytes)
    {
        map_.reserve(1u << 20);
    }

    pomai::Status MemTable::Put(pomai::VectorId id, std::span<const float> vec)
    {
        return Put(id, vec, pomai::Metadata());
    }

    pomai::Status MemTable::Put(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata& meta)
    {
        std::unique_lock lock(mutex_);
        if (vec.size() != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");
        float *dst = static_cast<float *>(arena_.Allocate(vec.size_bytes(), alignof(float)));
        std::memcpy(dst, vec.data(), vec.size_bytes());
        map_[id] = dst;
        
        // Store metadata if not empty
        // For MVP with just 'tenant', check if tenant is empty
        if (!meta.tenant.empty()) {
            metadata_[id] = meta;
        } else {
            // Ensure no metadata exists if empty (overwrite case)
            metadata_.erase(id);
        }
        
        return pomai::Status::Ok();
    }

    pomai::Status MemTable::PutBatch(const std::vector<pomai::VectorId>& ids,
                                      const std::vector<std::span<const float>>& vectors)
    {
        std::unique_lock lock(mutex_);
        // Validation
        if (ids.size() != vectors.size())
            return pomai::Status::InvalidArgument("ids and vectors size mismatch");
        if (ids.empty())
            return pomai::Status::Ok();
        
        // Validate dimensions for all vectors
        for (const auto& vec : vectors) {
            if (vec.size() != dim_)
                return pomai::Status::InvalidArgument("dim mismatch");
        }
        
        // Batch insert: allocate and copy all vectors
        for (std::size_t i = 0; i < ids.size(); ++i) {
            float *dst = static_cast<float *>(arena_.Allocate(vectors[i].size_bytes(), alignof(float)));
            std::memcpy(dst, vectors[i].data(), vectors[i].size_bytes());
            map_[ids[i]] = dst;
        }
        
        return pomai::Status::Ok();
    }

    pomai::Status MemTable::Delete(pomai::VectorId id)
    {
        std::unique_lock lock(mutex_);
        map_[id] = nullptr; // Tombstone
        metadata_.erase(id); // Clear metadata
        return pomai::Status::Ok();
    }

    pomai::Status MemTable::Get(pomai::VectorId id, const float** out_vec) const {
        std::shared_lock lock(mutex_);
        if (!out_vec) return Status::InvalidArgument("out_vec is null");
        auto it = map_.find(id);
        if (it == map_.end() || it->second == nullptr) {
            *out_vec = nullptr;
            return Status::NotFound("vector not found");
        }
        *out_vec = it->second;
        return Status::Ok();
    }

    pomai::Status MemTable::Get(pomai::VectorId id, const float** out_vec, pomai::Metadata* out_meta) const {
        std::shared_lock lock(mutex_);
        if (!out_vec) return Status::InvalidArgument("out_vec is null");
        auto it = map_.find(id);
        if (it == map_.end() || it->second == nullptr) {
            *out_vec = nullptr;
            return Status::NotFound("vector not found");
        }
        *out_vec = it->second;
        
        if (out_meta) {
            auto meta_it = metadata_.find(id);
            if (meta_it != metadata_.end()) {
                *out_meta = meta_it->second;
            } else {
                *out_meta = pomai::Metadata(); // Default/Empty
            }
        }
        return Status::Ok();
    }

    void MemTable::Clear()
    {
        std::unique_lock lock(mutex_);
        map_.clear();
        metadata_.clear();
        arena_.Clear();
    }



} // namespace pomai::table
