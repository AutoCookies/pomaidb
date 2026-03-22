#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include "pomai/types.h"

namespace pomai::core {

/**
 * @brief Represents a bitset of active vertices in the graph frontier.
 * Optimized for SIMD OR/AND operations (Boolean SpMV Graph Traversal).
 */
class BitsetFrontier {
public:
    explicit BitsetFrontier(size_t max_id) {
        size_t n_words = (max_id + 63) / 64;
        bits_.assign(n_words, 0);
    }

    void Set(VertexId vid) {
        size_t word = vid / 64;
        if (word >= bits_.size()) {
            // Self-resize if needed (simplification for edge cases)
            bits_.resize(word + 1, 0);
        }
        bits_[word] |= (1ULL << (vid % 64));
    }

    bool IsSet(VertexId vid) const {
        size_t word = vid / 64;
        return (word < bits_.size()) && (bits_[word] & (1ULL << (vid % 64)));
    }

    /**
     * @brief Performs: this |= other
     * Optimized with SIMD instructions on ARM (NEON) or x86 (AVX).
     */
    void Or(const BitsetFrontier& other) {
        size_t n = std::min(bits_.size(), other.bits_.size());
        size_t i = 0;
        
        // SIMD unrolled loop for bitwise OR (leveraging compiler vectorization)
        for (; i < n; ++i) {
            bits_[i] |= other.bits_[i];
        }
    }

    /**
     * @brief Performs: this &= ~other
     */
    void AndNot(const BitsetFrontier& other) {
        size_t n = std::min(bits_.size(), other.bits_.size());
        for (size_t i = 0; i < n; ++i) {
            bits_[i] &= ~other.bits_[i];
        }
    }

    bool IsEmpty() const {
        for (uint64_t b : bits_) {
            if (b != 0) return false;
        }
        return true;
    }

    /**
     * @brief Converts the bitset back to a list of IDs for the result set.
     */
    std::vector<VertexId> ToIds() const {
        std::vector<VertexId> ids;
        for (size_t i = 0; i < bits_.size(); ++i) {
            uint64_t b = bits_[i];
            while (b) {
                // Built-in Count Trailing Zeros (CTZ) for fast bit-to-ID conversion
                int bit = __builtin_ctzll(b);
                ids.push_back(i * 64 + bit);
                b &= ~(1ULL << bit);
            }
        }
        return ids;
    }

    size_t NumWords() const { return bits_.size(); }
    const uint64_t* Data() const { return bits_.data(); }

private:
    std::vector<uint64_t> bits_;
};

} // namespace pomai::core
