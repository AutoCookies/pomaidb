#include "util/crc32c.h"

namespace pomai::util
{

    static constexpr std::uint32_t kPoly = 0x82F63B78u; // reversed Castagnoli

    static std::uint32_t TableAt(std::uint32_t i)
    {
        std::uint32_t crc = i;
        for (int k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (kPoly & (~(crc & 1u) + 1u));
        return crc;
    }

    std::uint32_t Crc32c(const void *data, std::size_t n, std::uint32_t seed)
    {
        static std::uint32_t table[256];
        static bool inited = false;
        if (!inited)
        {
            for (std::uint32_t i = 0; i < 256; ++i)
                table[i] = TableAt(i);
            inited = true;
        }

        std::uint32_t crc = ~seed;
        const std::uint8_t *p = static_cast<const std::uint8_t *>(data);
        for (std::size_t i = 0; i < n; ++i)
        {
            crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
        }
        return ~crc;
    }

} // namespace pomai::util
