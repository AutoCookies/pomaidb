#include "tests/common/test_main.h"
#include <cstdint>
#include <cstring>
#include <string>

#include "util/crc32c.h"

POMAI_TEST(Crc32c_KnownVectors)
{
    // CRC32C(Castagnoli) of "123456789" is 0xE3069283.
    const std::string s = "123456789";
    std::uint32_t crc = pomai::util::Crc32c(s.data(), s.size(), /*seed*/ 0u);
    POMAI_EXPECT_EQ(crc, 0xE3069283u);

    // Empty input should yield seed.
    std::uint32_t seed = 0xDEADBEEFu;
    std::uint32_t crc2 = pomai::util::Crc32c(nullptr, 0, seed);
    POMAI_EXPECT_EQ(crc2, seed);
}

POMAI_TEST(Crc32c_IncrementalMatchesOneShot)
{
    const std::string a = "hello ";
    const std::string b = "world";

    std::uint32_t oneshot = pomai::util::Crc32c((a + b).data(), a.size() + b.size(), 0u);
    std::uint32_t inc = pomai::util::Crc32c(a.data(), a.size(), 0u);
    inc = pomai::util::Crc32c(b.data(), b.size(), inc);

    POMAI_EXPECT_EQ(oneshot, inc);
}
