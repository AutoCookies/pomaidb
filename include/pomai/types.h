#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace pomai
{

    using VectorId = std::uint64_t;

    struct Slice
    {
        const std::byte *data = nullptr;
        std::size_t size = 0;

        constexpr bool empty() const noexcept { return size == 0; }
    };

    struct FloatSpan
    {
        const float *data = nullptr;
        std::uint32_t dim = 0;

        constexpr std::span<const float> span() const noexcept { return {data, dim}; }
    };

} // namespace pomai
