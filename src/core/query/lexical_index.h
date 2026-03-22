#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include "pomai/types.h"

namespace pomai::core {

struct LexicalHit {
    VectorId id;
    float score;
};

/**
 * @brief LexicalIndex provides keyword-based searching (Full-text).
 * Designed to be lightweight for edge devices.
 */
class LexicalIndex {
public:
    LexicalIndex() = default;

    /**
     * @brief Adds a document to the index.
     */
    void Add(VectorId id, const std::string& text);

    /**
     * @brief Searches for documents matching the query.
     */
    void Search(const std::string& query, uint32_t topk, std::vector<LexicalHit>* out);

    /**
     * @brief Removes a document from the index.
     */
    void Remove(VectorId id);

    void Clear();

private:
    std::vector<std::string> Tokenize(const std::string& text);
    
    // Inverted index: word -> list of (id, term_frequency)
    std::unordered_map<std::string, std::vector<std::pair<VectorId, uint32_t>>> inverted_index_;
    
    // Document statistics for BM25
    std::unordered_map<VectorId, uint32_t> doc_lengths_;
    std::uint64_t total_docs_ = 0;
    double avg_doc_length_ = 0.0;
};

} // namespace pomai::core
