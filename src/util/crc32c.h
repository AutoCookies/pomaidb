#pragma once
#include <cstddef>
#include <cstdint>

namespace pomai::util
{

    // CRC32C (Castagnoli) software implementation.
    std::uint32_t Crc32c(const void *data, std::size_t n, std::uint32_t seed = 0);

} // namespace pomai::util