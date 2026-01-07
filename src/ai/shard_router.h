#pragma once
// src/sharding/router.h
// Small router helper: map key -> shard id. Uses std::hash for portability.
// If you later vendor xxhash, you can replace implementation here.

#include <string_view>
#include <cstdint>
#include <functional>

namespace pomai::shard
{

// Compute shard id for a key (klen bytes) given shard_count.
// If shard_count is power-of-two, uses mask for speed; otherwise uses modulo.
static inline uint32_t shard_for_key(const char *key, size_t klen, uint32_t shard_count)
{
    if (shard_count == 0)
        return 0;
    std::hash<std::string_view> h;
    uint64_t hv = h(std::string_view(key, klen));
    // fast path: power of two
    if ((shard_count & (shard_count - 1)) == 0)
        return static_cast<uint32_t>(hv & (shard_count - 1));
    return static_cast<uint32_t>(hv % shard_count);
}

} // namespace pomai::shard