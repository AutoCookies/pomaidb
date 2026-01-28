#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace pomai::server
{

    // Frame: u32 len (LE) + payload[len]
    // payload starts with u8 op

    enum class Op : std::uint8_t
    {
        PING = 1,
        CREATE_COLLECTION = 2,
        UPSERT_BATCH = 3,
        SEARCH = 4
    };

    struct Buf
    {
        std::vector<std::uint8_t> data;
        void clear() { data.clear(); }
    };

    void PutU8(Buf &b, std::uint8_t v);
    void PutU16(Buf &b, std::uint16_t v);
    void PutU32(Buf &b, std::uint32_t v);
    void PutU64(Buf &b, std::uint64_t v);
    void PutF32(Buf &b, float v);
    void PutBytes(Buf &b, const void *p, std::size_t n);
    void PutString(Buf &b, const std::string &s); // u16 len + bytes

    struct Reader
    {
        const std::uint8_t *p{nullptr};
        const std::uint8_t *e{nullptr};

        bool ReadU8(std::uint8_t &v);
        bool ReadU16(std::uint16_t &v);
        bool ReadU32(std::uint32_t &v);
        bool ReadU64(std::uint64_t &v);
        bool ReadF32(float &v);
        bool ReadBytes(void *out, std::size_t n);
        bool ReadString(std::string &s);
    };

    // utilities
    std::uint32_t LoadU32LE(const std::uint8_t *p);
    void StoreU32LE(std::uint8_t *p, std::uint32_t v);

} // namespace pomai::server
