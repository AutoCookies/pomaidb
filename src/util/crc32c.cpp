#include "pomai/util/crc32c.h"

// Check for SSE4.2 support via compiler macros
#if defined(__SSE4_2__) || defined(__CRC32__) || (defined(_M_X64) && !defined(__ARM_ARCH))
#include <nmmintrin.h>
#define POMAI_HAVE_SSE42 1
#endif

// Fallback table (Sẽ được khởi tạo lazy hoặc hardcoded)
// Để ngắn gọn cho bản production này, tôi dùng giải thuật software compact
static const std::uint32_t kCrc32cPoly = 0x82f63b78;

static std::uint32_t Crc32cSoftware(std::uint32_t crc, const void *data, std::size_t n)
{
    const std::uint8_t *p = static_cast<const std::uint8_t *>(data);
    crc = ~crc;
    while (n--)
    {
        crc ^= *p++;
        for (int k = 0; k < 8; k++)
            crc = crc & 1 ? (crc >> 1) ^ kCrc32cPoly : crc >> 1;
    }
    return ~crc;
}

namespace pomai::util
{
    std::uint32_t Crc32cExtend(std::uint32_t crc, const void *data, std::size_t n)
    {
#if defined(POMAI_HAVE_SSE42)
        // Hardware acceleration implementation
        // Xử lý alignment để đạt hiệu năng tối đa
        const std::uint8_t *p = static_cast<const std::uint8_t *>(data);
        std::size_t len = n;

        // Process byte-by-byte until aligned to 8 bytes (on 64-bit)
        // (Simplified version: process 64-bit chunks directly if possible)
        // Lưu ý: _mm_crc32_u64 yêu cầu hệ 64-bit.

#if defined(__x86_64__) || defined(_M_X64)
        const std::size_t kStep = 8;
        while (len >= kStep)
        {
            std::uint64_t chunk;
            // Copy để tránh lỗi unaligned access (dù x86 cho phép nhưng chậm)
            __builtin_memcpy(&chunk, p, sizeof(chunk));
            crc = static_cast<std::uint32_t>(
                _mm_crc32_u64(static_cast<std::uint64_t>(crc), chunk));
            p += kStep;
            len -= kStep;
        }
#endif

        // Process remaining bytes
        while (len > 0)
        {
            crc = _mm_crc32_u8(crc, *p++);
            len--;
        }
        return crc;
#else
        return Crc32cSoftware(crc, data, n);
#endif
    }

    std::uint32_t Crc32c(const void *data, std::size_t n)
    {
        return Crc32cExtend(0, data, n);
    }
}