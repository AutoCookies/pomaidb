// Zero-copy sliding-window chunker for Edge RAG.

#include "pomai/rag/chunking.h"
#include <algorithm>
#include <cstddef>

namespace pomai {

SlidingWindowChunker::SlidingWindowChunker(SlidingWindowOptions options)
    : opts_(std::move(options)) {
  if (opts_.chunk_bytes == 0) opts_.chunk_bytes = 256;
  if (opts_.stride_bytes == 0) opts_.stride_bytes = opts_.chunk_bytes;
}

void SlidingWindowChunker::Chunk(std::string_view document, std::vector<ChunkView>* out) const {
  out->clear();
  size_t chunk_size = opts_.chunk_bytes;
  size_t stride = opts_.stride_bytes;
  if (stride == 0) stride = chunk_size;

  uint64_t offset = 0;
  while (offset < document.size()) {
    size_t len = (std::min)(chunk_size, document.size() - offset);
    out->push_back(ChunkView{document.substr(offset, len), offset});
    if (offset + stride >= document.size()) break;
    offset += stride;
  }
}

}  // namespace pomai
