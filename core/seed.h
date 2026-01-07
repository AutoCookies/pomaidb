// core/seed.h
#pragma once
#include <cstdint>
#include <cstring>
#include <atomic>
#include <cstddef>
#include "core/config.h"

/*
 * Seed layout (cache line aligned)
 *
 * header (atomic<uint64_t>):
 *   bits 0..31   : expiry timestamp (seconds since epoch) (optional)
 *   bits 32..47  : key length  (uint16_t)
 *   bits 48..63  : value length if stored inline (uint16_t)
 *
 * Lightweight fields:
 *   uint8_t type;   // object type (OBJ_STRING, OBJ_VECTOR, ...)
 *   uint8_t flags;  // FLAG_INDIRECT etc.
 *
 * payload[48]:
 *   key bytes immediately followed by inline value bytes OR
 *   key bytes followed by a uint64_t offset that points to a blob header in Arena.
 *
 * Concurrency:
 * - Writers must write payload/type/flags before storing header (header.store(..., release)).
 * - Readers should load header with acquire semantics and then read dependent fields (type, flags, payload).
 */

struct alignas(pomai::config::SEED_ALIGNMENT) Seed
{
    std::atomic<uint64_t> header; // expiry (low32) | klen (32..47) | vlen (48..63)

    uint32_t entropy;
    uint32_t checksum;

    // type & flags
    uint8_t type;
    uint8_t flags;
    uint8_t reserved[6]; // padding to align payload on 8 bytes

    // payload area (key + inline value OR key + offset)
    char payload[pomai::config::SEED_PAYLOAD_BYTES];

    // Preserve backwards-compatible names by mapping to centralised config values.
    static constexpr uint8_t OBJ_STRING = pomai::config::SEED_OBJ_STRING;
    static constexpr uint8_t OBJ_VECTOR = 1; // new type for stored vectors (float32[])
    static constexpr uint8_t FLAG_INLINE = pomai::config::SEED_FLAG_INLINE;
    static constexpr uint8_t FLAG_INDIRECT = pomai::config::SEED_FLAG_INDIRECT;

    // Preserve expiry bits when updating meta
    void set_meta(uint16_t klen, uint16_t vlen)
    {
        uint64_t old = header.load(std::memory_order_relaxed);
        uint64_t expiry = old & 0xFFFFFFFFULL;
        uint64_t new_hdr = expiry | (static_cast<uint64_t>(klen) << 32) | (static_cast<uint64_t>(vlen) << 48);
        header.store(new_hdr, std::memory_order_release);
    }

    void set_expiry(uint32_t expiry_seconds_since_epoch)
    {
        uint64_t old = header.load(std::memory_order_relaxed);
        uint64_t upper = old & (~0xFFFFFFFFULL);
        uint64_t new_hdr = upper | static_cast<uint64_t>(expiry_seconds_since_epoch);
        header.store(new_hdr, std::memory_order_release);
    }

    uint32_t get_expiry() const { return static_cast<uint32_t>(header.load(std::memory_order_acquire) & 0xFFFFFFFFULL); }
    uint16_t get_klen() const { return static_cast<uint16_t>((header.load(std::memory_order_acquire) >> 32) & 0xFFFF); }
    uint16_t get_vlen() const { return static_cast<uint16_t>((header.load(std::memory_order_acquire) >> 48) & 0xFFFF); }

    bool is_initialized() const { return header.load(std::memory_order_acquire) != 0ULL; }

    // Key match helper used by hash table (compares only key bytes)
    bool key_match(const char *key, size_t len) const
    {
        if (get_klen() != len)
            return false;
        return memcmp(payload, key, len) == 0;
    }
};