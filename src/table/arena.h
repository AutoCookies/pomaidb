#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace pomai::table
{

    class Arena
    {
    public:
        explicit Arena(std::size_t block_bytes) : block_bytes_(block_bytes) {}

        void *Allocate(std::size_t n, std::size_t align);

    private:
        struct Block
        {
            std::unique_ptr<std::byte[]> mem;
            std::size_t used = 0;
        };

        std::size_t block_bytes_;
        std::vector<Block> blocks_;
    };

} // namespace pomai::table
