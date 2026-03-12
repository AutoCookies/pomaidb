// Edge RAG pipeline: IngestDocument and RetrieveContext.

#include "pomai/rag/pipeline.h"
#include "pomai/pomai.h"
#include "pomai/rag.h"
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace pomai {

namespace {

// Simple offline tokenizer: whitespace split, hash to TokenId (no external vocab).
void Tokenize(std::string_view text, std::vector<TokenId>* out_tokens) {
  out_tokens->clear();
  if (text.empty()) return;
  auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && is_space(text[i])) ++i;
    if (i >= text.size()) break;
    size_t start = i;
    while (i < text.size() && !is_space(text[i])) ++i;
    std::string_view word = text.substr(start, i - start);
    // FNV-1a style hash -> uint32_t
    uint32_t h = 2166136261u;
    for (unsigned char c : word) {
      h ^= static_cast<uint32_t>(c);
      h *= 16777619u;
    }
    out_tokens->push_back(static_cast<TokenId>(h));
  }
}

}  // namespace

RagPipeline::RagPipeline(DB* db,
                         std::string_view membrane_name,
                         uint32_t embedding_dim,
                         EmbeddingProvider* provider,
                         RagPipelineOptions options)
    : db_(db),
      membrane_name_(membrane_name),
      embedding_dim_(embedding_dim),
      provider_(provider),
      options_(std::move(options)),
      chunker_([] (const RagPipelineOptions& opts) {
        RecursiveChunkOptions o;
        o.max_chunk_bytes = opts.max_chunk_bytes;
        o.overlap_bytes = opts.overlap_bytes;
        return o;
      }(options_)) {}

RagPipeline::~RagPipeline() = default;

Status RagPipeline::IngestDocument(uint64_t doc_id, std::string_view text) {
  if (text.size() > options_.max_doc_bytes)
    return Status::ResourceExhausted("document exceeds max_doc_bytes");

  std::vector<ChunkView> chunks;
  chunker_.Chunk(text, &chunks);
  if (chunks.empty()) return Status::Ok();

  const size_t batch_size = options_.max_chunks_per_batch;
  for (size_t b = 0; b < chunks.size(); b += batch_size) {
    size_t end = (std::min)(b + batch_size, chunks.size());

    std::vector<std::string_view> batch_texts;
    batch_texts.reserve(end - b);
    for (size_t i = b; i < end; ++i)
      batch_texts.push_back(chunks[i].text);

    std::vector<std::vector<float>> embeddings;
    Status st = provider_->Embed(batch_texts, &embeddings, embedding_dim_);
    if (!st.ok()) return st;

    for (size_t i = b; i < end; ++i) {
      size_t j = i - b;
      RagChunk chunk;
      chunk.chunk_id = next_chunk_id_++;
      chunk.doc_id = doc_id;
      Tokenize(chunks[i].text, &chunk.tokens);
      if (chunk.tokens.empty())
        continue;  // skip empty-token chunks (RagEngine requires non-empty tokens)
      chunk.vec = VectorView(embeddings[j].data(), static_cast<uint32_t>(embeddings[j].size()));
      chunk.chunk_text = std::string(chunks[i].text);

      st = db_->PutChunk(membrane_name_, chunk);
      if (!st.ok()) return st;
    }
  }
  return Status::Ok();
}

Status RagPipeline::RetrieveContext(std::string_view query,
                                    uint32_t top_k,
                                    std::string* out_context) {
  if (out_context == nullptr) return Status::InvalidArgument("out_context is null");
  out_context->clear();

  std::vector<std::string_view> q_texts = {query};
  std::vector<std::vector<float>> q_embeddings;
  Status st = provider_->Embed(q_texts, &q_embeddings, embedding_dim_);
  if (!st.ok()) return st;

  std::vector<TokenId> query_tokens;
  Tokenize(query, &query_tokens);

  RagQuery rag_query;
  rag_query.tokens = std::span<const TokenId>(query_tokens);
  rag_query.vec = VectorView(q_embeddings[0].data(), embedding_dim_);
  rag_query.topk = top_k;

  RagSearchOptions opts;
  opts.candidate_budget = 500;
  opts.enable_vector_rerank = true;

  RagSearchResult result;
  st = db_->SearchRag(membrane_name_, rag_query, opts, &result);
  if (!st.ok()) return st;

  std::ostringstream oss;
  for (size_t i = 0; i < result.hits.size(); ++i) {
    if (i > 0) oss << "\n";
    oss << result.hits[i].chunk_text;
  }
  *out_context = oss.str();
  return Status::Ok();
}

}  // namespace pomai
