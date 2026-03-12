// Mock embedding provider: deterministic hash-based vectors for tests and default pipeline.

#include "pomai/rag/embedding_provider.h"
#include <cstdint>
#include <functional>
#include <numeric>

namespace pomai {

namespace {

// Simple deterministic hash over string view to seed per-dimension values.
uint32_t HashString(std::string_view s) {
  uint32_t h = 0x811c9dc5u;
  for (unsigned char c : s) {
    h ^= static_cast<uint32_t>(c);
    h *= 0x01000193u;
  }
  return h;
}

}  // namespace

MockEmbeddingProvider::MockEmbeddingProvider(uint32_t dim) : dim_(dim) {
  if (dim_ == 0) dim_ = 1;
}

Status MockEmbeddingProvider::Embed(std::span<const std::string_view> texts,
                                   std::vector<std::vector<float>>* out,
                                   uint32_t dim) {
  if (out == nullptr) return Status::InvalidArgument("out is null");
  if (dim != dim_) return Status::InvalidArgument("dim mismatch");
  if (texts.empty()) return Status::InvalidArgument("empty texts");

  out->resize(texts.size());
  for (size_t i = 0; i < texts.size(); ++i) {
    std::vector<float>& v = (*out)[i];
    v.resize(dim);
    uint32_t h = HashString(texts[i]);
    for (uint32_t d = 0; d < dim; ++d) {
      h = h * 31u + 17u + static_cast<uint32_t>(d);
      // Normalize to [-1, 1] for cosine similarity
      v[d] = (static_cast<float>(h % 10000u) / 5000.0f) - 1.0f;
    }
  }
  return Status::Ok();
}

}  // namespace pomai
