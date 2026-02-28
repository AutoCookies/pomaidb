#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include "slice.h"

namespace pomai
{

    using VectorId = std::uint64_t;

    /**
     * VectorView: Borrowed view of float vectors.
     * Enforces or assumes NEON-compatible alignment (16-byte) where required by kernels.
     */
    struct alignas(16) VectorView
    {
        const float *data = nullptr;
        std::uint32_t dim = 0;

        constexpr VectorView() = default;
        constexpr VectorView(const float *data_, std::uint32_t dim_) : data(data_), dim(dim_) {}
        /* implicit */ constexpr VectorView(std::span<const float> s) 
            : data(s.data()), dim(static_cast<std::uint32_t>(s.size())) {}
        
        constexpr std::span<const float> span() const noexcept { return {data, dim}; }
        
        constexpr std::size_t size_bytes() const noexcept
        {
            return static_cast<std::size_t>(dim) * sizeof(float);
        }
    };

} // namespace pomai
