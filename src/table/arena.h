#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "palloc_compat.h"

namespace pomai::table
{

    class Arena
    {
    public:
        explicit Arena(std::size_t block_bytes, palloc_heap_t* heap = nullptr)
            : block_bytes_(block_bytes), heap_(heap) {}

        ~Arena() { Clear(); }

        void *Allocate(std::size_t n, std::size_t align);
        void Clear();

    private:
        struct Block
        {
            std::byte* mem = nullptr;
            std::size_t used = 0;
        };

        std::size_t block_bytes_;
        palloc_heap_t* heap_;
        std::vector<Block> blocks_;
    };

} // namespace pomai::table
