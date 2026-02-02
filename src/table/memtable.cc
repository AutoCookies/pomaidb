#include "table/memtable.h"
#include <cstring>

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
        if (vec.size() != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");
        float *dst = static_cast<float *>(arena_.Allocate(vec.size_bytes(), alignof(float)));
        std::memcpy(dst, vec.data(), vec.size_bytes());
        map_[id] = dst;
        return pomai::Status::Ok();
    }

    pomai::Status MemTable::Delete(pomai::VectorId id)
    {
        map_[id] = nullptr;
        return pomai::Status::Ok();
    }

    pomai::Status MemTable::Get(pomai::VectorId id, const float** out_vec) const {
        if (!out_vec) return Status::InvalidArgument("out_vec is null");
        auto it = map_.find(id);
        if (it == map_.end() || it->second == nullptr) {
            *out_vec = nullptr;
            return Status::NotFound("vector not found");
        }
        *out_vec = it->second;
        return Status::Ok();
    }



} // namespace pomai::table
