#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "core/query/lexical_index.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::core {

class TextMembrane {
public:
    explicit TextMembrane(std::size_t max_docs = 50000) : max_docs_(max_docs) {}
    Status Put(VectorId id, const std::string& text);
    Status Get(VectorId id, std::string* out) const;
    Status Delete(VectorId id);
    Status Search(const std::string& query, uint32_t topk, std::vector<LexicalHit>* out) const;
    void Clear();
    void ForEach(const std::function<void(VectorId id, std::string_view text)>& fn) const;

private:
    void RebuildIndex();
    mutable LexicalIndex index_;
    std::unordered_map<VectorId, std::string> docs_;
    std::size_t max_docs_ = 50000;
};

} // namespace pomai::core

