// Zero-copy recursive character splitter for Edge RAG.

#include "pomai/rag/chunking.h"
#include <algorithm>
#include <string_view>

namespace pomai {

namespace {

void SplitRecursive(std::string_view doc, size_t max_bytes, size_t overlap_bytes,
                    const std::vector<std::string_view>& separators, size_t sep_index,
                    uint64_t base_offset, std::vector<ChunkView>* out) {
  if (doc.empty()) return;

  if (sep_index >= separators.size()) {
    // No more separators: emit remainder as one chunk (may exceed max; plan allows it for last)
    out->push_back(ChunkView{doc, base_offset});
    return;
  }

  std::string_view sep = separators[sep_index];
  if (sep.empty()) {
    SplitRecursive(doc, max_bytes, overlap_bytes, separators, sep_index + 1, base_offset, out);
    return;
  }

  uint64_t offset = base_offset;
  std::string_view rest = doc;

  while (!rest.empty()) {
    if (rest.size() <= max_bytes) {
      // Fits in one chunk; trim leading/trailing separators and emit
      std::string_view trimmed = rest;
      while (trimmed.size() >= sep.size() && trimmed.substr(0, sep.size()) == sep)
        trimmed.remove_prefix(sep.size());
      if (!trimmed.empty())
        out->push_back(ChunkView{trimmed, offset});
      return;
    }

    // Find last occurrence of sep in the first max_bytes (so we split on separator)
    std::string_view window = rest.substr(0, max_bytes);
    size_t last_sep = std::string_view::npos;
    size_t pos = 0;
    for (;;) {
      size_t i = window.find(sep, pos);
      if (i == std::string_view::npos) break;
      last_sep = i;
      pos = i + sep.size();
    }

    if (last_sep != std::string_view::npos) {
      std::string_view chunk = rest.substr(0, last_sep);
      // Trim trailing sep and leading/trailing whitespace if desired; we keep simple
      out->push_back(ChunkView{chunk, offset});
      size_t advance = last_sep + sep.size();
      if (overlap_bytes > 0 && advance > overlap_bytes)
        advance -= overlap_bytes;
      else
        advance = last_sep + sep.size();
      rest.remove_prefix(advance);
      offset += advance;
    } else {
      // No separator in window: try next separator level on this window
      SplitRecursive(window, max_bytes, overlap_bytes, separators, sep_index + 1, offset, out);
      rest.remove_prefix(window.size());
      offset += window.size();
    }
  }
}

}  // namespace

RecursiveChunkSplitter::RecursiveChunkSplitter(RecursiveChunkOptions options)
    : opts_(std::move(options)) {
  if (opts_.max_chunk_bytes == 0) opts_.max_chunk_bytes = 512;
}

void RecursiveChunkSplitter::Chunk(std::string_view document, std::vector<ChunkView>* out) const {
  out->clear();
  if (document.empty()) return;

  std::vector<std::string_view> seps = opts_.separators;
  if (seps.empty()) {
    seps = {"\n\n", "\n", " "};
  }
  SplitRecursive(document, opts_.max_chunk_bytes, opts_.overlap_bytes,
                 seps, 0, 0, out);
}

}  // namespace pomai
