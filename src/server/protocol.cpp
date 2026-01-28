#include "server/protocol.h"
#include <cstring>

namespace pomai::server
{

    static inline void PutLE(void *dst, std::uint64_t v, std::size_t n)
    {
        auto *p = reinterpret_cast<std::uint8_t *>(dst);
        for (std::size_t i = 0; i < n; ++i)
            p[i] = (std::uint8_t)((v >> (8 * i)) & 0xFF);
    }
    static inline std::uint64_t GetLE(const std::uint8_t *p, std::size_t n)
    {
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < n; ++i)
            v |= (std::uint64_t)p[i] << (8 * i);
        return v;
    }

    std::uint32_t LoadU32LE(const std::uint8_t *p)
    {
        return (std::uint32_t)GetLE(p, 4);
    }
    void StoreU32LE(std::uint8_t *p, std::uint32_t v)
    {
        PutLE(p, v, 4);
    }

    void PutBytes(Buf &b, const void *p, std::size_t n)
    {
        auto *x = reinterpret_cast<const std::uint8_t *>(p);
        b.data.insert(b.data.end(), x, x + n);
    }
    void PutU8(Buf &b, std::uint8_t v) { b.data.push_back(v); }
    void PutU16(Buf &b, std::uint16_t v)
    {
        std::uint8_t tmp[2];
        PutLE(tmp, v, 2);
        PutBytes(b, tmp, 2);
    }
    void PutU32(Buf &b, std::uint32_t v)
    {
        std::uint8_t tmp[4];
        PutLE(tmp, v, 4);
        PutBytes(b, tmp, 4);
    }
    void PutU64(Buf &b, std::uint64_t v)
    {
        std::uint8_t tmp[8];
        PutLE(tmp, v, 8);
        PutBytes(b, tmp, 8);
    }
    void PutF32(Buf &b, float v)
    {
        static_assert(sizeof(float) == 4);
        std::uint8_t tmp[4];
        std::memcpy(tmp, &v, 4);
        PutBytes(b, tmp, 4);
    }
    void PutString(Buf &b, const std::string &s)
    {
        if (s.size() > 65535)
        {
            PutU16(b, 0);
            return;
        }
        PutU16(b, (std::uint16_t)s.size());
        PutBytes(b, s.data(), s.size());
    }

    bool Reader::ReadBytes(void *out, std::size_t n)
    {
        if ((std::size_t)(e - p) < n)
            return false;
        std::memcpy(out, p, n);
        p += n;
        return true;
    }
    bool Reader::ReadU8(std::uint8_t &v)
    {
        if (p >= e)
            return false;
        v = *p++;
        return true;
    }
    bool Reader::ReadU16(std::uint16_t &v)
    {
        if ((e - p) < 2)
            return false;
        v = (std::uint16_t)GetLE(p, 2);
        p += 2;
        return true;
    }
    bool Reader::ReadU32(std::uint32_t &v)
    {
        if ((e - p) < 4)
            return false;
        v = (std::uint32_t)GetLE(p, 4);
        p += 4;
        return true;
    }
    bool Reader::ReadU64(std::uint64_t &v)
    {
        if ((e - p) < 8)
            return false;
        v = (std::uint64_t)GetLE(p, 8);
        p += 8;
        return true;
    }
    bool Reader::ReadF32(float &v)
    {
        if ((e - p) < 4)
            return false;
        std::memcpy(&v, p, 4);
        p += 4;
        return true;
    }
    bool Reader::ReadString(std::string &s)
    {
        std::uint16_t len = 0;
        if (!ReadU16(len))
            return false;
        if ((std::size_t)(e - p) < len)
            return false;
        s.assign(reinterpret_cast<const char *>(p), reinterpret_cast<const char *>(p + len));
        p += len;
        return true;
    }

} // namespace pomai::server
