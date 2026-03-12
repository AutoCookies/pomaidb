// pomai/rag/chunking.h — Zero-copy text chunking for Edge RAG.
// Uses std::string_view; no allocation of document text copies.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace pomai {

/** View over a single chunk; refers into the original document buffer. */
struct ChunkView {
  std::string_view text;
  uint64_t start_offset = 0;
};

/** Parameters for recursive character splitter. */
struct RecursiveChunkOptions {
  size_t max_chunk_bytes = 512;
  size_t overlap_bytes = 0;
  /** Preferred separators in order (e.g. "\n\n", "\n", " "). */
  std::vector<std::string_view> separators;
};

/** Recursive character splitter: zero-copy chunks over document. */
class RecursiveChunkSplitter {
 public:
  explicit RecursiveChunkSplitter(RecursiveChunkOptions options);

  /** Produce chunk views into \a document. Document must outlive the returned views. */
  void Chunk(std::string_view document, std::vector<ChunkView>* out) const;

 private:
  RecursiveChunkOptions opts_;
};

/** Sliding window: fixed-size chunks with optional overlap. */
struct SlidingWindowOptions {
  size_t chunk_bytes = 256;
  size_t stride_bytes = 0;  // 0 => stride = chunk_bytes (no overlap)
};

class SlidingWindowChunker {
 public:
  explicit SlidingWindowChunker(SlidingWindowOptions options);

  void Chunk(std::string_view document, std::vector<ChunkView>* out) const;

 private:
  SlidingWindowOptions opts_;
};

}  // namespace pomai
