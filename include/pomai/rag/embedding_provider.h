// pomai/rag/embedding_provider.h — Local embedding interface for Edge RAG.
// Implementations can be mock (tests), or a future GGML/llama.cpp-style backend.

#pragma once

#include "pomai/status.h"
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace pomai {

/**
 * Abstract embedding provider: embed one or many texts into fixed-dimension vectors.
 * Batch embedding is preferred for efficiency; callers can pass a single text.
 * Implementations must be thread-safe for concurrent Embed() calls if used from
 * multiple threads (pipeline may call from a single thread only).
 */
class EmbeddingProvider {
 public:
  virtual ~EmbeddingProvider() = default;

  /** Embed a batch of texts. out[i] has dimension Dim() for each i. */
  virtual Status Embed(std::span<const std::string_view> texts,
                      std::vector<std::vector<float>>* out,
                      uint32_t dim) = 0;

  /** Embedding dimension (e.g. 384, 768). */
  virtual uint32_t Dim() const = 0;
};

/**
 * Mock provider: deterministic, hash-based vectors for tests.
 * No external model required. Same text always yields the same vector.
 */
class MockEmbeddingProvider : public EmbeddingProvider {
 public:
  explicit MockEmbeddingProvider(uint32_t dim);

  Status Embed(std::span<const std::string_view> texts,
              std::vector<std::vector<float>>* out,
              uint32_t dim) override;

  uint32_t Dim() const override { return dim_; }

 private:
  uint32_t dim_;
};

}  // namespace pomai
