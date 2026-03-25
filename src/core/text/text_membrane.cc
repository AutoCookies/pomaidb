#include "core/text/text_membrane.h"

namespace pomai::core {

Status TextMembrane::Get(VectorId id, std::string* out) const {
    if (!out) return Status::InvalidArgument("text get out is null");
    const auto it = docs_.find(id);
    if (it == docs_.end()) return Status::NotFound("text doc not found");
    *out = it->second;
    return Status::Ok();
}

Status TextMembrane::Put(VectorId id, const std::string& text) {
    if (max_docs_ > 0 && docs_.size() >= max_docs_ && docs_.find(id) == docs_.end()) {
        // Strict memory cap for edge devices.
        auto it = docs_.begin();
        if (it != docs_.end()) {
            docs_.erase(it);
            RebuildIndex();
        }
    }
    docs_[id] = text;
    index_.Add(id, text);
    return Status::Ok();
}

Status TextMembrane::Delete(VectorId id) {
    docs_.erase(id);
    index_.Remove(id);
    return Status::Ok();
}

Status TextMembrane::Search(const std::string& query, uint32_t topk, std::vector<LexicalHit>* out) const {
    if (!out) return Status::InvalidArgument("text membrane out is null");
    out->clear();
    index_.Search(query, topk, out);
    return Status::Ok();
}

void TextMembrane::Clear() {
    docs_.clear();
    index_.Clear();
}

void TextMembrane::ForEach(const std::function<void(VectorId id, std::string_view text)>& fn) const {
    for (const auto& [id, t] : docs_) fn(id, t);
}

void TextMembrane::RebuildIndex() {
    index_.Clear();
    for (const auto& kv : docs_) {
        index_.Add(kv.first, kv.second);
    }
}

} // namespace pomai::core

