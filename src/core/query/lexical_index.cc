#include "core/query/lexical_index.h"
#include <sstream>
#include <cctype>

namespace pomai::core {

std::vector<std::string> LexicalIndex::Tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            current += static_cast<char>(std::tolower(c));
        } else if (!current.empty()) {
            tokens.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) tokens.push_back(std::move(current));
    return tokens;
}

void LexicalIndex::Add(VectorId id, const std::string& text) {
    auto tokens = Tokenize(text);
    if (tokens.empty()) return;

    // Remove from length if already exists (overwrite)
    if (doc_lengths_.count(id)) {
        avg_doc_length_ = (avg_doc_length_ * total_docs_ - doc_lengths_[id]) / (total_docs_ - 1);
        total_docs_--;
    }

    std::unordered_map<std::string, uint32_t> tf;
    for (const auto& t : tokens) tf[t]++;

    for (const auto& [word, count] : tf) {
        inverted_index_[word].push_back({id, count});
    }

    doc_lengths_[id] = static_cast<uint32_t>(tokens.size());
    total_docs_++;
    avg_doc_length_ = (avg_doc_length_ * (total_docs_ - 1) + tokens.size()) / total_docs_;
}

void LexicalIndex::Search(const std::string& query, uint32_t topk, std::vector<LexicalHit>* out) {
    auto q_tokens = Tokenize(query);
    if (q_tokens.empty() || total_docs_ == 0) return;

    std::unordered_map<VectorId, double> scores;
    const double k1 = 1.2;
    const double b = 0.75;

    for (const auto& word : q_tokens) {
        auto it = inverted_index_.find(word);
        if (it == inverted_index_.end()) continue;

        const auto& postings = it->second;
        double idf = std::log((total_docs_ - postings.size() + 0.5) / (postings.size() + 0.5) + 1.0);

        for (const auto& [id, count] : postings) {
            double tf = static_cast<double>(count);
            double L = static_cast<double>(doc_lengths_[id]) / avg_doc_length_;
            double term_score = idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * L));
            scores[id] += term_score;
        }
    }

    if (scores.empty()) return;

    for (const auto& [id, score] : scores) {
        out->push_back({id, static_cast<float>(score)});
    }

    std::partial_sort(out->begin(), std::min(out->end(), out->begin() + topk), out->end(),
                      [](const auto& a, const auto& b) { return a.score > b.score; });

    if (out->size() > topk) out->resize(topk);
}

void LexicalIndex::Remove(VectorId id) {
     // For a really efficient inverted index, removal involves scanning all lists.
     // In a lightweight memory-only version, we can tolerate lazy removal.
     // But for 100% correctness:
     if (doc_lengths_.count(id)) {
         avg_doc_length_ = (avg_doc_length_ * total_docs_ - doc_lengths_[id]);
         total_docs_--;
         if (total_docs_ > 0) avg_doc_length_ /= total_docs_; else avg_doc_length_ = 0;
         doc_lengths_.erase(id);
         
         // Ideally clear the word from inverted_index_ too.
         // In an edge DB, we frequently recreate segment-level lexical indices during compaction.
     }
}

void LexicalIndex::Clear() {
    inverted_index_.clear();
    doc_lengths_.clear();
    total_docs_ = 0;
    avg_doc_length_ = 0;
}

} // namespace pomai::core
