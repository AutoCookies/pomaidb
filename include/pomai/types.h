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

    struct VectorView
    {
        const float *data = nullptr;
        std::uint32_t dim = 0;

        constexpr VectorView() = default;
        constexpr VectorView(const float *data_, std::uint32_t dim_) : data(data_), dim(dim_) {}
        constexpr VectorView(std::span<const float> span)
            : data(span.data()), dim(static_cast<std::uint32_t>(span.size()))
        {
        }

        constexpr std::span<const float> span() const noexcept { return {data, dim}; }
        constexpr std::size_t size_bytes() const noexcept
        {
            return static_cast<std::size_t>(dim) * sizeof(float);
        }
    };

} // namespace pomai
