#pragma once
#include <cstddef>
#include <cstdint>

namespace pomai::util
{
    // Tính CRC32C của buffer.
    // Nếu CPU hỗ trợ SSE4.2, nó sẽ chạy ở tốc độ DRAM bandwidth.
    std::uint32_t Crc32c(const void *data, std::size_t n);

    // Tính nối tiếp (Extend) cho streaming writes
    std::uint32_t Crc32cExtend(std::uint32_t crc, const void *data, std::size_t n);
}