#pragma once
// ai/vector_index.h
// Simple linear-scan Vector Index (top-K) using PomaiMap + PomaiArena
//
// - Expects vectors stored as blob: [uint32_t byte_len][float32 ...]
// - Returns binary result buffer: sequence of [4B keylen][key bytes][4B score_bits(network order)]
//
// This is intentionally scalar and simple so it's easy to replace with SIMD kernel later.

#include <vector>
#include <cstdint>
#include "src/core/map.h"
#include "src/memory/arena.h"

class VectorIndex
{
public:
    explicit VectorIndex(PomaiMap *map);

    // Search for topk nearest vectors to 'query' (length dim floats).
    // Returns binary result buffer (ready to be sent as response body).
    // Format: repeated entries [4B keylen][key bytes][4B score_f32_bits (network order)]
    std::vector<char> search(const float *query, size_t dim, size_t topk) const;

private:
    PomaiMap *map_;
    PomaiArena *arena_;
};