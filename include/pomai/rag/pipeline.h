// pomai/rag/pipeline.h — Edge RAG pipeline: IngestDocument + RetrieveContext.

#pragma once

#include "pomai/rag/chunking.h"
#include "pomai/rag/embedding_provider.h"
#include "pomai/status.h"
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace pomai {

class DB;

/** Options for the RAG pipeline (memory and chunking limits). */
struct RagPipelineOptions {
  size_t max_chunk_bytes = 512;
  size_t max_doc_bytes = 4 * 1024 * 1024;   // 4 MiB per document
  size_t max_chunks_per_batch = 32;
  size_t overlap_bytes = 0;
};

/**
 * Edge RAG pipeline: ingest documents (chunk → embed → tokenize → PutChunk)
 * and retrieve context (embed query → SearchRag → format from chunk text).
 * Uses zero-copy chunking and a pluggable EmbeddingProvider.
 */
class RagPipeline {
 public:
  /**
   * Constructs a pipeline. Caller must have created and opened the RAG membrane.
   * @param db non-null DB with RAG support
   * @param membrane_name name of the open RAG membrane
   * @param embedding_dim must match provider->Dim() and membrane dim
   * @param provider non-null (e.g. MockEmbeddingProvider for tests)
   * @param options chunking and memory limits
   */
  RagPipeline(DB* db,
              std::string_view membrane_name,
              uint32_t embedding_dim,
              EmbeddingProvider* provider,
              RagPipelineOptions options = {});

  ~RagPipeline();

  RagPipeline(const RagPipeline&) = delete;
  RagPipeline& operator=(const RagPipeline&) = delete;

  /**
   * Ingest one document: chunk text, embed chunks, tokenize, store with optional chunk text.
   * Document buffer is processed in one pass; if doc size > max_doc_bytes returns ResourceExhausted.
   */
  Status IngestDocument(uint64_t doc_id, std::string_view text);

  /**
   * Retrieve context for a query: embed query, search RAG, format context from stored chunk text.
   * Returns concatenated chunk texts of top_k hits, separated by newlines.
   */
  Status RetrieveContext(std::string_view query,
                         uint32_t top_k,
                         std::string* out_context);

 private:
  DB* db_;
  std::string membrane_name_;
  uint32_t embedding_dim_;
  EmbeddingProvider* provider_;
  RagPipelineOptions options_;
  RecursiveChunkSplitter chunker_;
  uint64_t next_chunk_id_ = 1;  // simple monotonic counter per pipeline instance
};

}  // namespace pomai
