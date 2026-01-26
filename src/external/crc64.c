#include "src/external/crc64.h"
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

static const uint64_t POLY = UINT64_C(0xad93d23594c935a9);

static uint64_t crc64_table[256];
static pthread_once_t crc64_once = PTHREAD_ONCE_INIT;

static void crc64_build_table(void)
{
    for (int i = 0; i < 256; ++i)
    {
        uint64_t crc = (uint64_t)i;
        for (int j = 0; j < 8; ++j)
        {
            if (crc & 1)
                crc = (crc >> 1) ^ POLY;
            else
                crc >>= 1;
        }
        crc64_table[i] = crc;
    }
}

void crc64_init(void)
{
    pthread_once(&crc64_once, crc64_build_table);
}

uint64_t crc64(uint64_t crc, const unsigned char *s, uint64_t l)
{
    if (s == NULL || l == 0)
    {
        return crc;
    }

    pthread_once(&crc64_once, crc64_build_table);

    uint64_t c = ~crc;
    for (uint64_t i = 0; i < l; ++i)
    {
        uint8_t idx = (uint8_t)(c ^ (uint64_t)s[i]);
        c = crc64_table[idx] ^ (c >> 8);
    }
    return ~c;
}